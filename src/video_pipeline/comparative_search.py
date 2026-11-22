"""Unified Comparative Search Engine (CLIP vs SigLIP).

Streamlined to ONE canonical retrieval algorithm designed specifically for
controlled, rigorous benchmarking across diverse query taxonomies:

1. Canonical Pipeline:
   - Parallel text encoding (CLIP 512D unit-norm vs SigLIP 768D unit-norm).
   - Exact Inner Product similarity search via FAISS IndexFlatIP.
   - Near-duplicate temporal suppression (optional, for video keyframes).

2. Automated Query Taxonomy Profiler:
   - Language classification: English vs Vietnamese / Multilingual.
   - Length / Granularity: Short Entity (<= 3 words), Phrase (4-8 words), Complex Compositional (> 8 words).
   - Relational / Spatial cues: Prepositions, action verbs, color attributes.

3. Query-Level Benchmark Telemetry:
   - Inference latency (encode + search ms).
   - Score distribution & separation margin (Top-1 vs Mean Top-2..5).
   - Inter-model rank agreement (Jaccard Overlap @ K, Rank-1 agreement).
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..storage.vector_store import VectorStore
from ..models.clip_wrapper import CLIPWrapper
from ..models.siglip_wrapper import SigLIPWrapper
from .query_translator import QueryTranslator, detect_query_language


def profile_query(query_text: str) -> Dict[str, Any]:
    """Analyzes query linguistic features to support benchmark categorization."""
    text = query_text.strip()
    words = text.split()
    word_count = len(words)

    language = detect_query_language(text)

    # Granularity classification
    if word_count <= 3:
        granularity = "short_entity"
    elif word_count <= 8:
        granularity = "phrase"
    else:
        granularity = "complex_compositional"

    # Semantic attributes presence
    relational_cues = ["on", "in", "under", "next to", "near", "with", "behind", "front", "trên", "dưới", "trong", "cạnh"]
    has_relational = any(cue in text.lower().split() for cue in relational_cues)

    return {
        "word_count": word_count,
        "char_count": len(text),
        "language": language,
        "granularity": granularity,
        "has_relational_cue": has_relational,
    }


SUPPORTED_DEDUP_MODES = ("video_max_pool", "temporal_nms", "none")


class ComparativeSearchEngine:
    """The canonical dual-model comparative search engine for CLIP vs SigLIP."""

    def __init__(
        self,
        clip_store: Optional[VectorStore] = None,
        siglip_store: Optional[VectorStore] = None,
        device: Optional[str] = None,
        translator: Optional[QueryTranslator] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_store = clip_store
        self.siglip_store = siglip_store
        self.translator = translator or QueryTranslator()

        self._clip_model: Optional[CLIPWrapper] = None
        self._siglip_model: Optional[SigLIPWrapper] = None

    @property
    def clip_model(self) -> CLIPWrapper:
        if self._clip_model is None:
            self._clip_model = CLIPWrapper(device=self.device)
        return self._clip_model

    @property
    def siglip_model(self) -> SigLIPWrapper:
        if self._siglip_model is None:
            self._siglip_model = SigLIPWrapper(device=self.device)
        return self._siglip_model

    def set_stores(self, clip_store: Optional[VectorStore], siglip_store: Optional[VectorStore]) -> None:
        self.clip_store = clip_store
        self.siglip_store = siglip_store

    def _resolve_dedup_mode(
        self, dedup_mode: Optional[str], filter_duplicates: bool
    ) -> str:
        if dedup_mode is None:
            return "temporal_nms" if filter_duplicates else "none"
        mode = dedup_mode.strip().lower()
        if mode not in SUPPORTED_DEDUP_MODES:
            raise ValueError(
                f"Invalid dedup_mode '{dedup_mode}'. Must be one of {SUPPORTED_DEDUP_MODES}."
            )
        return mode

    def _search_store_with_dedup(
        self,
        store: VectorStore,
        query_vec: np.ndarray,
        k: int,
        dedup_mode: str,
        min_time_gap_sec: float = 5.0,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Executes FAISS search with adaptive candidate expansion for video/temporal deduplication."""
        if dedup_mode == "none":
            t0 = time.perf_counter()
            items = store.search_with_metadata(query_vec, k=k)
            return items[:k], (time.perf_counter() - t0) * 1000.0

        total_items = store.count
        max_fetch = min(total_items, 10000)
        if dedup_mode == "video_max_pool":
            fetch_k = min(max_fetch, max(k * 20, 300))
        else:
            fetch_k = min(max_fetch, max(k * 6, 80))

        total_search_ms = 0.0
        filtered: List[Dict[str, Any]] = []
        while True:
            t0 = time.perf_counter()
            raw_items = store.search_with_metadata(query_vec, k=fetch_k)
            total_search_ms += (time.perf_counter() - t0) * 1000.0

            if dedup_mode == "video_max_pool":
                filtered = self._dedup_video_max_pool(raw_items)
            else:
                filtered = self._filter_near_duplicates(raw_items, min_time_gap_sec)

            if len(filtered) >= k or fetch_k >= max_fetch or len(raw_items) < fetch_k:
                break
            fetch_k = min(max_fetch, fetch_k * 4)

        return filtered[:k], total_search_ms

    def search_text(
        self,
        query_text: str,
        k: int = 10,
        filter_duplicates: bool = False,
        min_time_gap_sec: float = 5.0,
        auto_translate: bool = True,
        dedup_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Canonical semantic retrieval comparing CLIP and SigLIP on the exact same query.

        Args:
            query_text: Input query string (English, Vietnamese, or Multilingual).
            k: Top-K results to retrieve.
            filter_duplicates: Legacy flag; maps to 'temporal_nms' when dedup_mode is omitted.
            min_time_gap_sec: Minimum temporal distance for frames from the same video (temporal_nms).
            auto_translate: Whether to automatically translate non-English queries to English
                before encoding with English-only CLIP and SigLIP models.
            dedup_mode: Deduplication/aggregation mode:
                - 'video_max_pool': 1 peak-scoring keyframe per distinct video.
                - 'temporal_nms': keep multiple scenes per video separated by >= min_time_gap_sec.
                - 'none': raw frame-level FAISS top-K without deduplication.
        """
        query_text = query_text.strip()
        if not query_text:
            raise ValueError("Query text cannot be empty.")

        resolved_dedup = self._resolve_dedup_mode(dedup_mode, filter_duplicates)
        query_profile = profile_query(query_text)
        if auto_translate and query_profile["language"] != "en":
            translation_info = self.translator.translate_to_english(
                query_text,
                source_lang=query_profile["language"],
            )
            effective_query = translation_info["translated_query"]
        else:
            translation_info = {
                "applied": False,
                "source_lang": query_profile["language"],
                "target_lang": "en",
                "original_query": query_text,
                "translated_query": query_text,
                "provider": "disabled" if (not auto_translate and query_profile["language"] != "en") else "none",
                "latency_ms": 0.0,
            }
            effective_query = query_text

        # 1. CLIP Retrieval (512D)
        clip_res = []
        clip_metrics = {}
        if self.clip_store is not None and self.clip_store.count > 0:
            t0 = time.perf_counter()
            with torch.no_grad():
                clip_feat = self.clip_model.get_text_embeddings([effective_query]).cpu().numpy()
            t_enc = (time.perf_counter() - t0) * 1000.0

            clip_res, t_search = self._search_store_with_dedup(
                self.clip_store,
                clip_feat[0],
                k=k,
                dedup_mode=resolved_dedup,
                min_time_gap_sec=min_time_gap_sec,
            )

            scores = [r["score"] for r in clip_res]
            top1 = scores[0] if scores else 0.0
            top_rest = np.mean(scores[1:5]) if len(scores) > 1 else top1

            clip_metrics = {
                "encode_latency_ms": round(t_enc, 2),
                "search_latency_ms": round(t_search, 2),
                "total_latency_ms": round(t_enc + t_search, 2),
                "top1_score": round(float(top1), 4),
                "mean_top5_score": round(float(np.mean(scores[:5])), 4) if scores else 0.0,
                "separation_margin": round(float(top1 - top_rest), 4),
            }

        # 2. SigLIP Retrieval (768D)
        siglip_res = []
        siglip_metrics = {}
        if self.siglip_store is not None and self.siglip_store.count > 0:
            t0 = time.perf_counter()
            with torch.no_grad():
                siglip_feat = self.siglip_model.get_text_embeddings([effective_query]).cpu().numpy()
            t_enc = (time.perf_counter() - t0) * 1000.0

            siglip_res, t_search = self._search_store_with_dedup(
                self.siglip_store,
                siglip_feat[0],
                k=k,
                dedup_mode=resolved_dedup,
                min_time_gap_sec=min_time_gap_sec,
            )

            scores = [r["score"] for r in siglip_res]
            top1 = scores[0] if scores else 0.0
            top_rest = np.mean(scores[1:5]) if len(scores) > 1 else top1

            siglip_metrics = {
                "encode_latency_ms": round(t_enc, 2),
                "search_latency_ms": round(t_search, 2),
                "total_latency_ms": round(t_enc + t_search, 2),
                "top1_score": round(float(top1), 4),
                "mean_top5_score": round(float(np.mean(scores[:5])), 4) if scores else 0.0,
                "separation_margin": round(float(top1 - top_rest), 4),
            }

        # 3. Inter-Model Agreement
        agreement = self._compute_agreement(clip_res, siglip_res, k=k, dedup_mode=resolved_dedup)

        # 4. Grouped View
        video_groups = self._aggregate_video_groups(siglip_res if siglip_res else clip_res)

        return {
            "query": query_text,
            "effective_query": effective_query,
            "translation": translation_info,
            "dedup_mode": resolved_dedup,
            "min_time_gap_sec": min_time_gap_sec,
            "query_type": "text",
            "query_profile": query_profile,
            "k": k,
            "clip_results": clip_res,
            "siglip_results": siglip_res,
            "video_groups": video_groups,
            "benchmark": {
                "clip": clip_metrics,
                "siglip": siglip_metrics,
                "agreement": agreement,
                "query_profile": query_profile,
                "translation": translation_info,
                "dedup_mode": resolved_dedup,
            },
        }

    def search_image(
        self,
        query_image: Image.Image,
        k: int = 10,
        dedup_mode: str = "video_max_pool",
        min_time_gap_sec: float = 5.0,
    ) -> Dict[str, Any]:
        """Canonical visual-to-visual search for reverse image retrieval."""
        if not isinstance(query_image, Image.Image):
            raise TypeError("query_image must be a PIL Image instance.")

        resolved_dedup = self._resolve_dedup_mode(dedup_mode, filter_duplicates=False)

        clip_res = []
        clip_metrics = {}
        if self.clip_store is not None and self.clip_store.count > 0:
            t0 = time.perf_counter()
            with torch.no_grad():
                clip_feat = self.clip_model.get_image_embeddings([query_image]).cpu().numpy()
            t_enc = (time.perf_counter() - t0) * 1000.0

            clip_res, t_search = self._search_store_with_dedup(
                self.clip_store,
                clip_feat[0],
                k=k,
                dedup_mode=resolved_dedup,
                min_time_gap_sec=min_time_gap_sec,
            )

            scores = [r["score"] for r in clip_res]
            clip_metrics = {
                "encode_latency_ms": round(t_enc, 2),
                "search_latency_ms": round(t_search, 2),
                "total_latency_ms": round(t_enc + t_search, 2),
                "top1_score": round(float(scores[0]), 4) if scores else 0.0,
            }

        siglip_res = []
        siglip_metrics = {}
        if self.siglip_store is not None and self.siglip_store.count > 0:
            t0 = time.perf_counter()
            with torch.no_grad():
                siglip_feat = self.siglip_model.get_image_embeddings([query_image]).cpu().numpy()
            t_enc = (time.perf_counter() - t0) * 1000.0

            siglip_res, t_search = self._search_store_with_dedup(
                self.siglip_store,
                siglip_feat[0],
                k=k,
                dedup_mode=resolved_dedup,
                min_time_gap_sec=min_time_gap_sec,
            )

            scores = [r["score"] for r in siglip_res]
            siglip_metrics = {
                "encode_latency_ms": round(t_enc, 2),
                "search_latency_ms": round(t_search, 2),
                "total_latency_ms": round(t_enc + t_search, 2),
                "top1_score": round(float(scores[0]), 4) if scores else 0.0,
            }

        agreement = self._compute_agreement(clip_res, siglip_res, k=k, dedup_mode=resolved_dedup)
        video_groups = self._aggregate_video_groups(siglip_res if siglip_res else clip_res)

        return {
            "query": "[Uploaded Image]",
            "query_type": "image",
            "dedup_mode": resolved_dedup,
            "k": k,
            "clip_results": clip_res,
            "siglip_results": siglip_res,
            "video_groups": video_groups,
            "benchmark": {
                "clip": clip_metrics,
                "siglip": siglip_metrics,
                "agreement": agreement,
                "dedup_mode": resolved_dedup,
            },
        }

    def _dedup_video_max_pool(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Video-Wise Max-Pooling: keeps the single highest-scoring keyframe per distinct video."""
        kept: List[Dict[str, Any]] = []
        seen_idx: Dict[str, int] = {}

        for item in items:
            vid = item.get("video_id")
            if not vid:
                entry = dict(item)
                entry["rank"] = len(kept) + 1
                kept.append(entry)
                continue

            if vid not in seen_idx:
                entry = dict(item)
                entry["matched_frame_count"] = 1
                entry["rank"] = len(kept) + 1
                seen_idx[vid] = len(kept)
                kept.append(entry)
            else:
                kept[seen_idx[vid]]["matched_frame_count"] += 1

        return kept

    def _filter_near_duplicates(self, items: List[Dict[str, Any]], min_time_gap_sec: float) -> List[Dict[str, Any]]:
        """True Temporal NMS: suppresses frames within min_time_gap_sec of ANY kept frame in the same video."""
        kept: List[Dict[str, Any]] = []
        kept_times_by_video: Dict[str, List[float]] = {}

        for item in items:
            vid = item.get("video_id")
            if not vid:
                entry = dict(item)
                entry["rank"] = len(kept) + 1
                kept.append(entry)
                continue

            ts = float(item.get("timestamp_sec", 0.0))
            prev_times = kept_times_by_video.get(vid, [])
            if any(abs(ts - prev_ts) < min_time_gap_sec for prev_ts in prev_times):
                continue

            entry = dict(item)
            entry["rank"] = len(kept) + 1
            kept.append(entry)
            kept_times_by_video.setdefault(vid, []).append(ts)

        return kept

    def _aggregate_video_groups(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            vid = item.get("video_id") or item.get("id", "single_item")
            grouped.setdefault(vid, []).append(item)

        video_list = []
        for vid, kfs in grouped.items():
            kfs.sort(key=lambda x: float(x.get("timestamp_sec", 0.0)))
            max_score = max(float(x.get("score", 0.0)) for x in kfs)
            first = kfs[0]
            video_list.append({
                "video_id": vid,
                "video_title": first.get("video_title", vid),
                "language": first.get("language", "en"),
                "max_score": round(max_score, 4),
                "frame_count": len(kfs),
                "keyframes": kfs,
            })

        video_list.sort(key=lambda v: v["max_score"], reverse=True)
        return video_list

    def _compute_agreement(
        self,
        clip_res: List[Dict[str, Any]],
        siglip_res: List[Dict[str, Any]],
        k: int = 10,
        dedup_mode: str = "none",
    ) -> Dict[str, Any]:
        use_video_level = dedup_mode == "video_max_pool"
        if use_video_level:
            clip_ids = [r.get("video_id") or r.get("id") or r.get("filename") or r.get("relative_path") for r in clip_res]
            siglip_ids = [r.get("video_id") or r.get("id") or r.get("filename") or r.get("relative_path") for r in siglip_res]
        else:
            clip_ids = [r.get("id") or r.get("filename") or r.get("relative_path") for r in clip_res]
            siglip_ids = [r.get("id") or r.get("filename") or r.get("relative_path") for r in siglip_res]

        distinct_videos_clip = len({r.get("video_id") for r in clip_res if r.get("video_id")})
        distinct_videos_siglip = len({r.get("video_id") for r in siglip_res if r.get("video_id")})

        set_clip = set(filter(None, clip_ids))
        set_siglip = set(filter(None, siglip_ids))

        if not set_clip or not set_siglip:
            return {
                "jaccard_overlap_pct": 0.0,
                "intersection_count": 0,
                "top5_overlap": 0,
                "rank_1_match": False,
                "agreement_level": "video" if use_video_level else "frame",
                "distinct_videos_clip": distinct_videos_clip,
                "distinct_videos_siglip": distinct_videos_siglip,
            }

        intersection = set_clip.intersection(set_siglip)
        union = set_clip.union(set_siglip)
        jaccard = (len(intersection) / len(union)) * 100.0 if union else 0.0
        top5_clip = set(filter(None, clip_ids[:5]))
        top5_siglip = set(filter(None, siglip_ids[:5]))
        top5_overlap = len(top5_clip.intersection(top5_siglip))
        rank_1_match = bool(clip_ids and siglip_ids and clip_ids[0] == siglip_ids[0])

        return {
            "jaccard_overlap_pct": round(float(jaccard), 2),
            "intersection_count": len(intersection),
            "top5_overlap": top5_overlap,
            "rank_1_match": rank_1_match,
            "agreement_level": "video" if use_video_level else "frame",
            "distinct_videos_clip": distinct_videos_clip,
            "distinct_videos_siglip": distinct_videos_siglip,
        }
