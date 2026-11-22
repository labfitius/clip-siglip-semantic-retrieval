"""Comprehensive verification suite for Vietnamese Cross-Modal Semantic Retrieval.

Tests:
1. Language detection (accented VI, unaccented VI, English, multilingual non-ASCII).
2. Multi-tier QueryTranslator (Taxonomy Bank, LRU cache, Google NMT, Offline Opus-MT, graceful fallback).
3. End-to-end CLIP & SigLIP retrieval quality on NewsSeek (164,476 keyframes) & COCO (5,000 images):
   compares raw Vietnamese (auto_translate=False) vs auto-translated VI -> EN (auto_translate=True).
4. FastAPI server endpoints (/, /api/status, /api/datasets, /api/search).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient

import server
from src.video_pipeline.comparative_search import ComparativeSearchEngine
from src.video_pipeline.query_translator import QueryTranslator, detect_query_language


class TestDeduplicationModesUnit(unittest.TestCase):
    def test_video_max_pool_keeps_peak_frame_per_video(self):
        eng = ComparativeSearchEngine()
        items = [
            {"id": "f1", "video_id": "vA", "timestamp_sec": 10.0, "score": 0.90},
            {"id": "f2", "video_id": "vA", "timestamp_sec": 11.0, "score": 0.88},
            {"id": "f3", "video_id": "vB", "timestamp_sec": 20.0, "score": 0.85},
            {"id": "f4", "video_id": "vA", "timestamp_sec": 40.0, "score": 0.80},
            {"id": "f5", "video_id": "vC", "timestamp_sec": 5.0, "score": 0.75},
        ]
        deduped = eng._dedup_video_max_pool(items)
        self.assertEqual([x["video_id"] for x in deduped], ["vA", "vB", "vC"])
        self.assertEqual(deduped[0]["id"], "f1")
        self.assertEqual(deduped[0]["matched_frame_count"], 3)
        self.assertEqual(deduped[1]["matched_frame_count"], 1)

    def test_temporal_nms_suppresses_against_all_kept_timestamps(self):
        eng = ComparativeSearchEngine()
        # Frame f3 (10.5s) is close to f1 (10.0s) even though f2 (50.0s) intervened!
        items = [
            {"id": "f1", "video_id": "vA", "timestamp_sec": 10.0, "score": 0.90},
            {"id": "f2", "video_id": "vA", "timestamp_sec": 50.0, "score": 0.85},
            {"id": "f3", "video_id": "vA", "timestamp_sec": 10.5, "score": 0.84},
            {"id": "f4", "video_id": "vA", "timestamp_sec": 52.0, "score": 0.83},
            {"id": "f5", "video_id": "vA", "timestamp_sec": 25.0, "score": 0.82},
        ]
        kept = eng._filter_near_duplicates(items, min_time_gap_sec=5.0)
        self.assertEqual([x["id"] for x in kept], ["f1", "f2", "f5"])


class TestLanguageDetection(unittest.TestCase):
    def test_vietnamese_with_diacritics(self):
        self.assertEqual(detect_query_language("Người đi xe máy trên đường phố đông đúc"), "vi")
        self.assertEqual(detect_query_language("Phà chở khách bốc cháy nghi ngút khói đen"), "vi")
        self.assertEqual(detect_query_language("Động đất đổ nát"), "vi")

    def test_vietnamese_unaccented(self):
        self.assertEqual(detect_query_language("xe cuu hoa dap lua"), "vi")
        self.assertEqual(detect_query_language("nguoi di xe may tren duong pho"), "vi")
        self.assertEqual(detect_query_language("chay rung khoi den"), "vi")

    def test_english_no_false_positives(self):
        english_queries = [
            "A person riding a motorcycle on a crowded street",
            "Wildfire smoke plumes over forest mountain",
            "Firefighter spraying high pressure water hose at house fire",
            "Luigi Mangione state court trial hearing",
            "Space launch rocket on the launchpad",
            "Gold bullion bars stacked inside bank vault",
            "cat on mat",
            "red car",
        ]
        for q in english_queries:
            with self.subTest(query=q):
                self.assertEqual(detect_query_language(q), "en")

    def test_non_ascii_multilingual(self):
        self.assertEqual(detect_query_language("三峡大坝光伏发电"), "multilingual")
        self.assertEqual(detect_query_language("ロケット打ち上げ"), "multilingual")


class TestQueryTranslatorTiers(unittest.TestCase):
    def test_english_passthrough(self):
        tr = QueryTranslator(enable_online=False, enable_offline_fallback=False)
        res = tr.translate_to_english("A dog catching a frisbee on the beach")
        self.assertFalse(res["applied"])
        self.assertEqual(res["provider"], "none")
        self.assertEqual(res["translated_query"], "A dog catching a frisbee on the beach")

    def test_tier1_taxonomy_bank_and_cache(self):
        tr = QueryTranslator(enable_online=False, enable_offline_fallback=False)
        vi_q = "Phà chở khách bốc cháy nghi ngút khói đen trong khi các hành khách mặc áo phao màu cam lên thuyền cứu hộ"
        res1 = tr.translate_to_english(vi_q)
        self.assertTrue(res1["applied"])
        self.assertEqual(res1["provider"], "taxonomy_bank")
        self.assertIn("ferry", res1["translated_query"].lower())

        res2 = tr.translate_to_english(vi_q)
        self.assertTrue(res2["applied"])
        self.assertEqual(res2["provider"], "cache:taxonomy_bank")
        self.assertEqual(res2["translated_query"], res1["translated_query"])

    def test_tier2_online_google_nmt_freeform(self):
        tr = QueryTranslator(enable_online=True, enable_offline_fallback=False)
        res = tr.translate_to_english("Người đi xe máy trên đường phố đông đúc")
        self.assertTrue(res["applied"])
        self.assertIn(res["provider"], ("google_nmt", "mymemory"))
        low = res["translated_query"].lower()
        self.assertTrue(any(w in low for w in ("motorbike", "motorcycle", "street", "road")), msg=res)

    def test_tier3_offline_marian_mt(self):
        tr = QueryTranslator(enable_online=False, enable_offline_fallback=True)
        res = tr.translate_to_english("Cảnh sát bắt giữ người biểu tình trên đường phố")
        self.assertTrue(res["applied"])
        self.assertEqual(res["provider"], "opus_mt_offline")
        low = res["translated_query"].lower()
        self.assertTrue("police" in low or "protest" in low, msg=res)

    def test_graceful_fallback_when_all_disabled(self):
        tr = QueryTranslator(enable_online=False, enable_offline_fallback=False)
        raw = "Một câu truy vấn hoàn toàn mới không có trong từ điển"
        res = tr.translate_to_english(raw)
        self.assertFalse(res["applied"])
        self.assertEqual(res["provider"], "fallback_raw")
        self.assertEqual(res["translated_query"], raw)


class TestEndToEndRetrievalAndAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)

    def test_frontend_and_status_endpoints(self):
        r_root = self.client.get("/")
        self.assertEqual(r_root.status_code, 200)
        self.assertIn("html", r_root.text.lower())

        r_status = self.client.get("/api/status")
        self.assertEqual(r_status.status_code, 200)
        st = r_status.json()
        self.assertTrue(st["datasets"]["newsseek"]["ready"])
        self.assertEqual(st["datasets"]["newsseek"]["clip_vectors"], 164476)
        self.assertEqual(st["datasets"]["newsseek"]["siglip_vectors"], 164476)

        r_ds = self.client.get("/api/datasets")
        self.assertEqual(r_ds.status_code, 200)
        ns_ds = next(d for d in r_ds.json()["datasets"] if d["id"] == "newsseek")
        self.assertGreaterEqual(len(ns_ds["presets"]), 5)

    def test_empty_query_returns_400(self):
        r = self.client.post("/api/search", json={"query": "   ", "dataset": "newsseek", "k": 6})
        self.assertEqual(r.status_code, 400)

    def test_vietnamese_freeform_search_dramatically_improves_relevance(self):
        q = "Người đi xe máy trên đường phố đông đúc"

        # With auto_translate=True -> should retrieve Vietnam motorcycle documentary (kIRP3hZJuG8)
        r_on = self.client.post(
            "/api/search",
            json={"query": q, "dataset": "newsseek", "k": 6, "auto_translate": True},
        )
        self.assertEqual(r_on.status_code, 200)
        d_on = r_on.json()
        self.assertTrue(d_on["translation"]["applied"])
        self.assertNotEqual(d_on["effective_query"], q)
        self.assertIn("top5_overlap", d_on["benchmark"]["agreement"])

        clip_vids_on = [x["video_id"] for x in d_on["clip_results"]]
        siglip_vids_on = [x["video_id"] for x in d_on["siglip_results"]]
        self.assertIn("kIRP3hZJuG8", clip_vids_on[:3], msg=f"CLIP top results: {clip_vids_on}")
        self.assertIn("kIRP3hZJuG8", siglip_vids_on[:3], msg=f"SigLIP top results: {siglip_vids_on}")

        # With auto_translate=False -> translation disabled
        r_off = self.client.post(
            "/api/search",
            json={"query": q, "dataset": "newsseek", "k": 6, "auto_translate": False},
        )
        self.assertEqual(r_off.status_code, 200)
        d_off = r_off.json()
        self.assertFalse(d_off["translation"]["applied"])
        self.assertEqual(d_off["translation"]["provider"], "disabled")
        self.assertEqual(d_off["effective_query"], q)
        # Verify raw Vietnamese misses kIRP3hZJuG8
        self.assertNotIn("kIRP3hZJuG8", [x["video_id"] for x in d_off["siglip_results"][:3]])

    def test_vietnamese_ferry_fire_and_earthquake_freeform(self):
        # Free-form Ferry fire query
        r_ferry = self.client.post(
            "/api/search",
            json={
                "query": "Phà chở khách bốc cháy nghi ngút khói đen trên biển",
                "dataset": "newsseek",
                "k": 6,
                "auto_translate": True,
            },
        )
        self.assertEqual(r_ferry.status_code, 200)
        d_ferry = r_ferry.json()
        self.assertTrue(d_ferry["translation"]["applied"])
        siglip_titles = " | ".join(x.get("video_title", "").lower() for x in d_ferry["siglip_results"][:5])
        self.assertIn("ferry", siglip_titles, msg=f"SigLIP ferry titles: {siglip_titles}")

        # Free-form Earthquake rescue query
        r_eq = self.client.post(
            "/api/search",
            json={
                "query": "Đội cứu hộ đào bới đống đổ nát sau động đất",
                "dataset": "newsseek",
                "k": 6,
                "auto_translate": True,
            },
        )
        self.assertEqual(r_eq.status_code, 200)
        d_eq = r_eq.json()
        self.assertTrue(d_eq["translation"]["applied"])
        eq_titles = " | ".join(x.get("video_title", "").lower() for x in d_eq["siglip_results"][:5])
        self.assertTrue(
            "quake" in eq_titles or "rescue" in eq_titles or "collapse" in eq_titles,
            msg=f"SigLIP earthquake titles: {eq_titles}",
        )

    def test_unaccented_vietnamese_query(self):
        r = self.client.post(
            "/api/search",
            json={"query": "xe cuu hoa dap lua", "dataset": "newsseek", "k": 6, "auto_translate": True},
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["query_profile"]["language"], "vi")
        self.assertTrue(data["translation"]["applied"])
        self.assertIn("fire", data["effective_query"].lower())

    def test_grounded_kis_vietnamese_hit_rate_vs_raw(self):
        bm_path = ROOT_DIR / "storage" / "newsseek" / "vlm_grounded_queries_benchmark.json"
        with open(bm_path, "r", encoding="utf-8") as f:
            bm = json.load(f)

        kis_items = bm.get("textual_kis", [])
        hits_on = 0
        hits_off = 0
        for item in kis_items:
            target = item["target_video_id"]
            q_vi = item["query_vi"]

            d_on = self.client.post(
                "/api/search",
                json={"query": q_vi, "dataset": "newsseek", "k": 10, "auto_translate": True},
            ).json()
            vids_on = {x["video_id"] for x in d_on["clip_results"] + d_on["siglip_results"]}
            if target in vids_on:
                hits_on += 1

            d_off = self.client.post(
                "/api/search",
                json={"query": q_vi, "dataset": "newsseek", "k": 10, "auto_translate": False},
            ).json()
            vids_off = {x["video_id"] for x in d_off["clip_results"] + d_off["siglip_results"]}
            if target in vids_off:
                hits_off += 1

        print(f"\n[KIS 12-Query VI Benchmark @ Top-10] Raw VI hits: {hits_off}/{len(kis_items)} -> Auto-Translate VI->EN hits: {hits_on}/{len(kis_items)}")
        self.assertGreater(hits_on, hits_off)
        self.assertGreaterEqual(hits_on, 7)

    def test_vietnamese_search_on_coco(self):
        r = self.client.post(
            "/api/search",
            json={
                "query": "Một con chó đang bắt đĩa bay trên bãi biển",
                "dataset": "coco",
                "k": 5,
                "auto_translate": True,
            },
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["translation"]["applied"])
        top_captions = " ".join(
            " ".join(x.get("captions", [])) for x in data["siglip_results"][:3]
        ).lower()
        self.assertIn("dog", top_captions)
        self.assertIn("frisbee", top_captions)

    def test_dedup_modes_end_to_end_api(self):
        q = "Người đi xe máy trên đường phố đông đúc"

        # 1. Default mode = video_max_pool -> k=12 must return 12 distinct videos!
        r_vmp = self.client.post(
            "/api/search",
            json={"query": q, "dataset": "newsseek", "k": 12, "auto_translate": True},
        )
        self.assertEqual(r_vmp.status_code, 200)
        d_vmp = r_vmp.json()
        self.assertEqual(d_vmp["dedup_mode"], "video_max_pool")
        self.assertEqual(len(d_vmp["clip_results"]), 12)
        self.assertEqual(len(d_vmp["siglip_results"]), 12)
        self.assertEqual(len({x["video_id"] for x in d_vmp["clip_results"]}), 12)
        self.assertEqual(len({x["video_id"] for x in d_vmp["siglip_results"]}), 12)
        self.assertEqual(d_vmp["benchmark"]["agreement"]["agreement_level"], "video")
        self.assertEqual(d_vmp["benchmark"]["agreement"]["distinct_videos_siglip"], 12)
        self.assertEqual(d_vmp["clip_results"][0]["video_id"], "kIRP3hZJuG8")
        self.assertEqual(d_vmp["siglip_results"][0]["video_id"], "kIRP3hZJuG8")
        self.assertTrue(d_vmp["benchmark"]["agreement"]["rank_1_match"])

        # 2. Temporal NMS (min_time_gap_sec=5.0) -> no two kept frames of same video within 5s
        r_nms = self.client.post(
            "/api/search",
            json={
                "query": q,
                "dataset": "newsseek",
                "k": 12,
                "auto_translate": True,
                "dedup_mode": "temporal_nms",
                "min_time_gap_sec": 5.0,
            },
        )
        self.assertEqual(r_nms.status_code, 200)
        d_nms = r_nms.json()
        self.assertEqual(d_nms["dedup_mode"], "temporal_nms")
        for key in ("clip_results", "siglip_results"):
            items = d_nms[key]
            self.assertEqual(len(items), 12)
            by_vid = {}
            for it in items:
                by_vid.setdefault(it["video_id"], []).append(float(it["timestamp_sec"]))
            for vid, ts_list in by_vid.items():
                for i in range(len(ts_list)):
                    for j in range(i + 1, len(ts_list)):
                        self.assertGreaterEqual(
                            abs(ts_list[i] - ts_list[j]),
                            5.0 - 1e-6,
                            msg=f"{key} video {vid} has frames too close: {ts_list}",
                        )

        # 3. Raw Keyframes (dedup_mode='none') -> preserves raw FAISS behavior (multiple frames from same video)
        r_raw = self.client.post(
            "/api/search",
            json={
                "query": q,
                "dataset": "newsseek",
                "k": 12,
                "auto_translate": True,
                "dedup_mode": "none",
            },
        )
        self.assertEqual(r_raw.status_code, 200)
        d_raw = r_raw.json()
        self.assertEqual(d_raw["dedup_mode"], "none")
        self.assertLess(len({x["video_id"] for x in d_raw["clip_results"]}), 12)

        # 4. Invalid dedup_mode -> HTTP 400
        r_bad = self.client.post(
            "/api/search",
            json={"query": q, "dataset": "newsseek", "k": 6, "dedup_mode": "invalid_mode"},
        )
        self.assertEqual(r_bad.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
