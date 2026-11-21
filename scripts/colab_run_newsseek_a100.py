"""Colab GPU In-Memory Streaming Pipeline for NewsSeek Dataset (CLIP + SigLIP).

Executes cleanly on Google Colab (A100 / L4 / T4) without needing pre-built manifest files:
1. Environment & safe Kaggle credential verification (no hardcoded API keys).
2. Fast datacenter download of `namnguynnnn/newsseek-core-v1` (cleans raw zip immediately).
3. Automatic keyframe discovery: uses existing `newsseek_manifest.json` if present, otherwise
   runs fast HSV color-histogram scene detection + Tiered Keyframe Policy on the fly.
4. In-memory video frame decoding (0 JPEG files written to disk -> saves 10+ GB disk space).
5. Dual-model GPU batch inference: CLIP ViT-B/32 (512D) + SigLIP ViT-B/16-224 (768D) in FP16.
6. Periodic checkpointing to `--output-dir` (e.g. Google Drive) with automatic resume on reconnect.
7. Builds exact FAISS IndexFlatIP stores (`storage/newsseek/{clip,siglip}/`) + benchmark JSON
   + packaged `newsseek_vector_stores.zip` ready to drop directly into the local FastAPI server.

Usage on Colab:
    python scripts/colab_run_newsseek_a100.py \\
        --output-dir /content/drive/MyDrive/xlav_newsseek_output \\
        --batch-size 128 \\
        --checkpoint-every 50 \\
        --max-videos 0
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force unbuffered output and UTF-8 encoding
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"


# ==============================================================================
# Helper: Tensor extraction for newer Transformers versions
# ==============================================================================
def extract_tensor_features(output: Any) -> Any:
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    if hasattr(output, "__getitem__") and not hasattr(output, "shape"):
        return output[0]
    return output


# ==============================================================================
# Step 1: Dependencies & Safe Kaggle Credential Setup
# ==============================================================================
def setup_environment(install_deps: bool = False) -> None:
    print("=== [Step 0] Verifying Environment & Kaggle Credentials ===")
    if install_deps:
        pkgs = [
            "faiss-cpu",
            "transformers",
            "sentencepiece",
            "protobuf",
            "opencv-python-headless",
            "pillow",
            "kaggle",
            "tqdm",
            "pandas",
        ]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    access_token_file = kaggle_dir / "access_token"
    kaggle_json_file = kaggle_dir / "kaggle.json"

    env_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env_token:
        access_token_file.write_text(env_token, encoding="utf-8")
        os.chmod(str(access_token_file), 0o600)
        print("Configured Kaggle credentials from KAGGLE_API_TOKEN env var.")
    elif access_token_file.exists():
        os.chmod(str(access_token_file), 0o600)
        print(f"Using existing Kaggle access token: {access_token_file}")
    elif kaggle_json_file.exists():
        os.chmod(str(kaggle_json_file), 0o600)
        print(f"Using existing Kaggle credentials: {kaggle_json_file}")
    else:
        print(
            "[Warning] No Kaggle credentials found. Set KAGGLE_API_TOKEN or create "
            "~/.kaggle/access_token (or ~/.kaggle/kaggle.json) before downloading."
        )


def download_dataset(raw_videos_dir: Path) -> None:
    raw_videos_dir.mkdir(parents=True, exist_ok=True)
    existing_mp4s = list(raw_videos_dir.rglob("*.mp4"))
    if len(existing_mp4s) >= 600:
        print(f"[Step 1] Found {len(existing_mp4s)} MP4 videos already in {raw_videos_dir}. Skipping download.")
        return

    print("--- [Step 1] Downloading namnguynnnn/newsseek-core-v1 from Kaggle ---")
    t0 = time.time()
    res = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "namnguynnnn/newsseek-core-v1", "--unzip", "-p", str(raw_videos_dir)],
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed with exit code {res.returncode}. "
            "Check your KAGGLE_API_TOKEN / ~/.kaggle/access_token."
        )

    # Clean up any leftover zip archives to save ~20GB disk space
    for z in raw_videos_dir.glob("*.zip"):
        try:
            print(f"Cleaning up {z.name} to free disk space...")
            z.unlink()
        except OSError:
            pass

    mp4s = list(raw_videos_dir.rglob("*.mp4"))
    print(f"Download & extraction finished in {time.time() - t0:.1f}s. Total MP4 videos ready: {len(mp4s)}")


# ==============================================================================
# Step 2: Scene Detection & In-Memory Keyframe Extraction
# ==============================================================================
def keyframe_count_tiered(duration_s: float) -> int:
    if duration_s <= 3.0:
        return 1
    if duration_s <= 10.0:
        return 3
    return 5


def detect_scenes_fast(cap: Any, fps: float, num_frames: int, threshold: float = 0.35) -> List[Tuple[int, int]]:
    import cv2

    if num_frames <= 0:
        return [(0, 0)]
    step = max(1, int(fps * 0.25))
    prev_hist = None
    scene_start = 0
    max_frames = int(25.0 * fps)
    scenes: List[Tuple[int, int]] = []

    for f_idx in range(0, num_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        small = cv2.resize(frame, (120, 68), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold or (f_idx - scene_start) >= max_frames:
                scenes.append((scene_start, f_idx - 1))
                scene_start = f_idx
        prev_hist = hist

    if scene_start < num_frames:
        scenes.append((scene_start, num_frames - 1))
    return scenes if scenes else [(0, num_frames - 1)]


def select_keyframes_tiered(scenes: List[Tuple[int, int]], fps: float) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for idx, (s_frame, e_frame) in enumerate(scenes):
        dur_frames = e_frame - s_frame + 1
        if dur_frames <= 0:
            continue
        count = min(dur_frames, keyframe_count_tiered(dur_frames / fps))
        pts = sorted(
            {
                min(e_frame, max(s_frame, s_frame + round((i + 0.5) * dur_frames / count)))
                for i in range(count)
            }
        )
        for p in pts:
            selected.append({"frame_number": int(p), "timestamp_sec": round(p / fps, 3), "scene_index": idx})
    return selected


def load_video_records(raw_videos_dir: Path, manifest_path: Optional[Path]) -> List[Dict[str, Any]]:
    """Loads video list from a prebuilt manifest if available, otherwise from dataset videos.csv."""
    if manifest_path is not None:
        gz_path = Path(f"{manifest_path}.gz")
        if gz_path.exists() and not manifest_path.exists():
            print(f"Decompressing {gz_path} -> {manifest_path}...")
            with gzip.open(gz_path, "rb") as f_in, open(manifest_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                records = raw.get("videos") or list(raw.values())
            else:
                records = raw
            print(f"Loaded pre-built manifest ({len(records)} videos) from {manifest_path}")
            return records

    # Fallback: read videos.csv directly from downloaded Kaggle dataset or local repo
    csv_candidates = [
        raw_videos_dir / "metadata" / "videos.csv",
        raw_videos_dir / "videos.csv",
        Path(__file__).resolve().parent.parent / "datasets" / "newsseek" / "annotations" / "videos.csv",
    ]
    csv_candidates.extend(raw_videos_dir.rglob("videos.csv"))
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"Neither manifest ({manifest_path}) nor videos.csv found in {raw_videos_dir}."
        )

    import pandas as pd

    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        vid = str(row["video_id"]).strip()
        if not vid or vid == "nan":
            continue
        records.append(
            {
                "video_id": vid,
                "title": str(row.get("title", vid)),
                "language": str(row.get("language", "en")).strip().lower() or "en",
                "source_name": str(row.get("source_name", "")),
                "local_filename": str(row.get("local_filename", "")),
                "keyframes": [],  # Will be detected on the fly via HSV scene detection
            }
        )
    print(f"Loaded {len(records)} video entries from {csv_path} (on-the-fly scene detection enabled).")
    return records


def find_all_video_files(search_dir: Path, video_records: List[Dict[str, Any]]) -> Dict[str, Path]:
    all_mp4s = list(search_dir.rglob("*.mp4"))
    vids = {v["video_id"] for v in video_records}
    video_map: Dict[str, Path] = {}
    for p in all_mp4s:
        bname = p.name
        video_map[p.stem] = p
        for vid in vids:
            if vid in bname:
                video_map[vid] = p
    return video_map


def extract_video_frames_in_memory(
    video_path: Path, keyframes: Optional[List[Dict[str, Any]]] = None
) -> Tuple[List[Tuple[Dict[str, Any], Any]], float, int]:
    """Extracts keyframes directly into PIL Images in RAM (0 JPEG writes to disk)."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or math.isnan(fps):
        fps = 25.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = num_frames / fps if fps > 0 else 0.0

    num_scenes = 0
    if not keyframes:
        scenes = detect_scenes_fast(cap, fps, num_frames)
        num_scenes = len(scenes)
        keyframes = select_keyframes_tiered(scenes, fps)
    else:
        num_scenes = len({kf.get("scene_index", 0) for kf in keyframes})

    sorted_kfs = sorted(keyframes, key=lambda x: x.get("frame_number", 0))
    extracted: List[Tuple[Dict[str, Any], Any]] = []
    current_frame = -1

    for kf in sorted_kfs:
        target_f = int(kf.get("frame_number", 0))
        if current_frame >= 0 and 0 <= target_f - current_frame < 15:
            while current_frame < target_f:
                cap.grab()
                current_frame += 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            current_frame = target_f

        ret, frame = cap.read()
        current_frame += 1
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        if max(h, w) > 480:
            scale = 480.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        extracted.append((kf, pil_img))

    cap.release()
    return extracted, duration_sec, num_scenes


# ==============================================================================
# Step 3: Checkpointing & GPU Batch Inference
# ==============================================================================
def save_checkpoint(
    ckpt_dir: Path,
    clip_embeds_list: List[Any],
    siglip_embeds_list: List[Any],
    all_metadata: List[Dict[str, Any]],
    processed_records: List[Dict[str, Any]],
) -> None:
    import numpy as np

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    clip_mat = np.concatenate(clip_embeds_list, axis=0).astype(np.float32) if clip_embeds_list else np.empty((0, 512), dtype=np.float32)
    siglip_mat = np.concatenate(siglip_embeds_list, axis=0).astype(np.float32) if siglip_embeds_list else np.empty((0, 768), dtype=np.float32)

    np.save(ckpt_dir / "clip_embeds.npy", clip_mat)
    np.save(ckpt_dir / "siglip_embeds.npy", siglip_mat)
    with open(ckpt_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False)
    with open(ckpt_dir / "processed_records.json", "w", encoding="utf-8") as f:
        json.dump(processed_records, f, ensure_ascii=False)
    print(f"    [Checkpoint] Saved {len(processed_records)} videos ({len(all_metadata)} frames) -> {ckpt_dir}")


def load_checkpoint(ckpt_dir: Path) -> Tuple[List[Any], List[Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    import numpy as np

    clip_p = ckpt_dir / "clip_embeds.npy"
    siglip_p = ckpt_dir / "siglip_embeds.npy"
    meta_p = ckpt_dir / "metadata.json"
    rec_p = ckpt_dir / "processed_records.json"

    if not (clip_p.exists() and siglip_p.exists() and meta_p.exists() and rec_p.exists()):
        return [], [], [], []

    clip_mat = np.load(clip_p)
    siglip_mat = np.load(siglip_p)
    with open(meta_p, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)
    with open(rec_p, "r", encoding="utf-8") as f:
        processed_records = json.load(f)

    if len(clip_mat) != len(all_metadata) or len(siglip_mat) != len(all_metadata):
        print("[Checkpoint] Mismatch detected, starting fresh.")
        return [], [], [], []

    print(f"[Resume] Loaded checkpoint: {len(processed_records)} videos ({len(all_metadata)} frames) from {ckpt_dir}")
    clip_list = [clip_mat] if len(clip_mat) > 0 else []
    siglip_list = [siglip_mat] if len(siglip_mat) > 0 else []
    return clip_list, siglip_list, all_metadata, processed_records


def run_streaming_encoding(
    raw_videos_dir: Path,
    output_dir: Path,
    manifest_path: Optional[Path],
    batch_size: int = 128,
    max_videos: int = 0,
    checkpoint_every: int = 50,
    resume: bool = True,
):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== [Step 2] Initializing Models on Device: {device} ===")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")

    video_records = load_video_records(raw_videos_dir, manifest_path)
    video_files_map = find_all_video_files(raw_videos_dir, video_records)
    available_records = [v for v in video_records if v["video_id"] in video_files_map]
    if max_videos > 0:
        available_records = available_records[:max_videos]
    print(f"Mapped {len(available_records)} / {len(video_records)} video files ready to encode.")

    if not available_records:
        raise RuntimeError(f"No .mp4 videos matched in {raw_videos_dir}.")

    # Load CLIP
    print("\nLoading CLIP ViT-B/32 (openai/clip-vit-base-patch32, 512D)...")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    # Load SigLIP
    print("Loading SigLIP ViT-B/16-224 (google/siglip-base-patch16-224, 768D)...")
    siglip_proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()

    ckpt_dir = output_dir / "checkpoints"
    if resume:
        clip_embeds_list, siglip_embeds_list, all_metadata, processed_records = load_checkpoint(ckpt_dir)
    else:
        clip_embeds_list, siglip_embeds_list, all_metadata, processed_records = [], [], [], []

    done_vids = {r["video_id"] for r in processed_records}
    autocast_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)) if device == "cuda" else nullcontext

    print("\n--- [Step 3] Streaming Video Keyframe Extraction & GPU Inference ---")
    t_start = time.time()
    total_videos = len(available_records)
    session_frames = 0
    session_videos = 0

    for v_idx, v in enumerate(available_records, start=1):
        vid = v["video_id"]
        if vid in done_vids:
            continue

        title = v.get("title", v.get("video_file", vid))
        lang = str(v.get("language", "en")).strip().lower() or "en"
        vpath = video_files_map[vid]
        raw_kfs = v.get("keyframes") or None

        extracted_pairs, duration_sec, num_scenes = extract_video_frames_in_memory(vpath, raw_kfs)
        if not extracted_pairs:
            continue

        images = [p[1] for p in extracted_pairs]
        v_clip_embeds = []
        v_siglip_embeds = []

        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]

            # CLIP Inference
            c_in = {k: val.to(device) for k, val in clip_proc(images=chunk, return_tensors="pt").items()}
            with torch.inference_mode(), autocast_ctx():
                c_out = extract_tensor_features(clip_model.get_image_features(**c_in))
                c_out = c_out / c_out.norm(dim=-1, keepdim=True)
            v_clip_embeds.append(c_out.cpu().float().numpy())

            # SigLIP Inference
            s_in = {k: val.to(device) for k, val in siglip_proc(images=chunk, return_tensors="pt").items()}
            with torch.inference_mode(), autocast_ctx():
                s_out = extract_tensor_features(siglip_model.get_image_features(**s_in))
                s_out = s_out / s_out.norm(dim=-1, keepdim=True)
            v_siglip_embeds.append(s_out.cpu().float().numpy())

        clip_embeds_list.append(np.concatenate(v_clip_embeds, axis=0))
        siglip_embeds_list.append(np.concatenate(v_siglip_embeds, axis=0))

        video_kfs_meta = []
        for kf_data, _ in extracted_pairs:
            f_no = int(kf_data.get("frame_number", 0))
            fname = f"frame_{f_no:06d}.jpg"
            rel_path = f"{vid}/keyframes/{fname}"
            item_meta = {
                "id": f"newsseek_{len(all_metadata)}",
                "kf_idx": len(all_metadata),
                "video_id": vid,
                "frame_number": f_no,
                "timestamp_sec": kf_data.get("timestamp_sec", 0.0),
                "scene_index": kf_data.get("scene_index", 0),
                "relative_path": rel_path,
                "video_title": title,
                "language": lang,
            }
            all_metadata.append(item_meta)
            video_kfs_meta.append(
                {
                    "frame_number": f_no,
                    "timestamp_sec": kf_data.get("timestamp_sec", 0.0),
                    "scene_index": kf_data.get("scene_index", 0),
                }
            )

        processed_records.append(
            {
                "video_id": vid,
                "title": title,
                "language": lang,
                "duration_seconds": round(duration_sec, 2),
                "num_scenes": num_scenes,
                "num_keyframes": len(extracted_pairs),
                "keyframes": video_kfs_meta,
            }
        )
        done_vids.add(vid)
        session_frames += len(extracted_pairs)
        session_videos += 1

        if session_videos % 10 == 0 or v_idx == total_videos:
            elapsed = time.time() - t_start
            fps = session_frames / max(elapsed, 0.001)
            remaining = max(0, total_videos - len(processed_records))
            eta_sec = remaining * (elapsed / max(session_videos, 1))
            print(
                f"[{len(processed_records):3d}/{total_videos}] Total frames: {len(all_metadata):,} | "
                f"Speed: {fps:.1f} fps | ETA: {eta_sec/60:.1f} min"
            )

        if checkpoint_every > 0 and session_videos % checkpoint_every == 0:
            save_checkpoint(ckpt_dir, clip_embeds_list, siglip_embeds_list, all_metadata, processed_records)

    total_time = time.time() - t_start
    avg_fps = session_frames / max(total_time, 0.001) if session_frames else 0.0
    print(f"\nEncoding Completed in {total_time:.1f}s ({total_time/60:.1f} min). Session Speed: {avg_fps:.1f} fps.")

    # Save generated manifest for future reuse
    manifest_out = output_dir / "newsseek_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(
            {"dataset": "newsseek-core-v1", "total_videos": len(processed_records), "total_keyframes": len(all_metadata), "videos": processed_records},
            f,
            ensure_ascii=False,
        )
    print(f"Saved manifest -> {manifest_out}")

    # Consolidate Embeddings
    print("Consolidating full embedding matrices...")
    all_clip_matrix = np.concatenate(clip_embeds_list, axis=0).astype(np.float32)
    all_siglip_matrix = np.concatenate(siglip_embeds_list, axis=0).astype(np.float32)

    unique_titles = [r["title"] for r in processed_records]

    # Encode Video Query Titles
    print(f"\nEncoding {len(unique_titles)} video query titles with CLIP...")
    clip_txt_list = []
    with torch.inference_mode():
        for i in range(0, len(unique_titles), 128):
            batch_txt = unique_titles[i : i + 128]
            t_in = clip_proc(text=batch_txt, padding=True, truncation=True, return_tensors="pt").to(device)
            with autocast_ctx():
                t_out = extract_tensor_features(clip_model.get_text_features(**t_in))
                t_out = t_out / t_out.norm(dim=-1, keepdim=True)
            clip_txt_list.append(t_out.cpu().float().numpy())
    all_clip_txt_matrix = np.concatenate(clip_txt_list, axis=0).astype(np.float32)

    print(f"Encoding {len(unique_titles)} video query titles with SigLIP...")
    siglip_txt_list = []
    with torch.inference_mode():
        for i in range(0, len(unique_titles), 128):
            batch_txt = unique_titles[i : i + 128]
            t_in = siglip_proc(text=batch_txt, padding="max_length", truncation=True, return_tensors="pt").to(device)
            with autocast_ctx():
                t_out = extract_tensor_features(siglip_model.get_text_features(**t_in))
                t_out = t_out / t_out.norm(dim=-1, keepdim=True)
            siglip_txt_list.append(t_out.cpu().float().numpy())
    all_siglip_txt_matrix = np.concatenate(siglip_txt_list, axis=0).astype(np.float32)

    del clip_model, siglip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        all_metadata,
        processed_records,
        all_clip_matrix,
        all_clip_txt_matrix,
        all_siglip_matrix,
        all_siglip_txt_matrix,
        total_time,
        avg_fps,
    )


# ==============================================================================
# Step 4 & 5: FAISS Vector Stores, Benchmark & ZIP Packaging
# ==============================================================================
def build_and_package_stores(
    output_dir: Path,
    all_metadata: List[Dict[str, Any]],
    processed_records: List[Dict[str, Any]],
    clip_mat: Any,
    clip_txt_mat: Any,
    siglip_mat: Any,
    siglip_txt_mat: Any,
    total_time: float,
    avg_fps: float,
) -> Path:
    import faiss
    import numpy as np
    import torch

    storage_dir = output_dir / "storage" / "newsseek"
    clip_store_dir = storage_dir / "clip"
    siglip_store_dir = storage_dir / "siglip"
    clip_store_dir.mkdir(parents=True, exist_ok=True)
    siglip_store_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- [Step 4] Building FAISS IndexFlatIP Vector Stores ---")

    # 4.1 CLIP Vector Store (512D)
    clip_mat = np.ascontiguousarray(clip_mat, dtype=np.float32)
    clip_index = faiss.IndexFlatIP(clip_mat.shape[1])
    faiss.normalize_L2(clip_mat)
    clip_index.add(clip_mat)
    faiss.write_index(clip_index, str(clip_store_dir / "index.faiss"))
    with open(clip_store_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dim": int(clip_mat.shape[1]),
                "index_type": "FlatIP",
                "model_name": "clip",
                "count": len(all_metadata),
                "items": all_metadata,
            },
            f,
            ensure_ascii=False,
        )
    print(f"Saved CLIP Vector Store: {clip_index.ntotal:,} vectors -> {clip_store_dir}")

    # 4.2 SigLIP Vector Store (768D)
    siglip_mat = np.ascontiguousarray(siglip_mat, dtype=np.float32)
    siglip_index = faiss.IndexFlatIP(siglip_mat.shape[1])
    faiss.normalize_L2(siglip_mat)
    siglip_index.add(siglip_mat)
    faiss.write_index(siglip_index, str(siglip_store_dir / "index.faiss"))
    with open(siglip_store_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dim": int(siglip_mat.shape[1]),
                "index_type": "FlatIP",
                "model_name": "siglip",
                "count": len(all_metadata),
                "items": all_metadata,
            },
            f,
            ensure_ascii=False,
        )
    print(f"Saved SigLIP Vector Store: {siglip_index.ntotal:,} vectors -> {siglip_store_dir}")

    # Step 5: Benchmark Evaluation (Video-Wise Max-Pooling)
    print("\n--- [Step 5] Evaluating Video-Wise Max-Pooling Retrieval ---")
    eval_device = "cuda" if torch.cuda.is_available() else "cpu"

    def evaluate_retrieval(kf_mat, txt_mat, records):
        kf_t = torch.from_numpy(kf_mat).to(eval_device)
        txt_t = torch.from_numpy(txt_mat).to(eval_device)
        frame_sims = torch.matmul(kf_t, txt_t.T).cpu()
        n_vid = len(records)
        video_sim_matrix = torch.zeros((n_vid, n_vid))

        kf_start = 0
        for v_idx, v in enumerate(records):
            n_kf = int(v.get("num_keyframes", len(v.get("keyframes", []))))
            if n_kf > 0 and (kf_start + n_kf) <= len(frame_sims):
                video_sim_matrix[v_idx] = frame_sims[kf_start : kf_start + n_kf].max(dim=0).values
            kf_start += n_kf

        k_vals = [1, 5, 10]
        top_max = min(max(k_vals), n_vid)
        _, topk_preds = video_sim_matrix.topk(top_max, dim=0, largest=True)
        targets = torch.arange(n_vid).unsqueeze(0).expand_as(topk_preds)
        matches = topk_preds == targets

        res: Dict[str, Any] = {}
        for k in k_vals:
            eff_k = min(k, top_max)
            res[f"R@{k}"] = round(float((matches[:eff_k].any(dim=0).float().mean().item()) * 100.0), 2)

        _, full_sorted = video_sim_matrix.sort(dim=0, descending=True)
        targets_full = torch.arange(n_vid).unsqueeze(0).expand_as(full_sorted)
        match_coords = (full_sorted == targets_full).nonzero(as_tuple=False)
        ranks = match_coords[:, 0] + 1
        res["MRR"] = round(float((1.0 / ranks.float()).mean().item()), 4)
        res["mR"] = round(float(np.mean([res[f"R@{k}"] for k in k_vals])), 2)

        gap = torch.norm(kf_t.mean(dim=0) - txt_t.mean(dim=0), p=2).item()
        res["modality_gap"] = round(float(gap), 4)

        langs = [v.get("language", "en").lower() for v in records]
        en_idx = [i for i, l in enumerate(langs) if l == "en"]
        non_en_idx = [i for i, l in enumerate(langs) if l != "en"]
        if en_idx:
            res["R@1_EN"] = round(float((matches[0, en_idx].float().mean().item()) * 100.0), 2)
        if non_en_idx:
            res["R@1_NonEN"] = round(float((matches[0, non_en_idx].float().mean().item()) * 100.0), 2)

        return res

    clip_bm = evaluate_retrieval(clip_mat, clip_txt_mat, processed_records)
    clip_bm["kf_enc_time_sec"] = round(total_time, 2)
    clip_bm["throughput_fps"] = round(avg_fps, 1)

    siglip_bm = evaluate_retrieval(siglip_mat, siglip_txt_mat, processed_records)
    siglip_bm["kf_enc_time_sec"] = round(total_time, 2)
    siglip_bm["throughput_fps"] = round(avg_fps, 1)

    bm_payload = {
        "dataset": "newsseek",
        "total_videos": len(processed_records),
        "total_keyframes": len(all_metadata),
        "clip_vit_b32": clip_bm,
        "siglip_vit_b16_224": siglip_bm,
    }

    bm_path = output_dir / "newsseek_benchmark_results.json"
    with open(bm_path, "w", encoding="utf-8") as f:
        json.dump(bm_payload, f, indent=2)

    print("\n=== NewsSeek Benchmark Summary ===")
    print(json.dumps(bm_payload, indent=2))

    # Step 6: Package ZIP archive directly in output_dir (safe on Google Drive)
    zip_output_path = output_dir / "newsseek_vector_stores.zip"
    print(f"\n--- [Step 6] Compressing Vector Stores -> {zip_output_path} ---")
    with zipfile.ZipFile(zip_output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(clip_store_dir / "index.faiss", "storage/newsseek/clip/index.faiss")
        zf.write(clip_store_dir / "metadata.json", "storage/newsseek/clip/metadata.json")
        zf.write(siglip_store_dir / "index.faiss", "storage/newsseek/siglip/index.faiss")
        zf.write(siglip_store_dir / "metadata.json", "storage/newsseek/siglip/metadata.json")
        zf.write(bm_path, "newsseek_benchmark_results.json")

    size_mb = zip_output_path.stat().st_size / (1024**2)
    print(f"Archive Created Successfully: {zip_output_path} ({size_mb:.2f} MB)")
    print("=== NEWSSEEK PIPELINE COMPLETE ===")
    return zip_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab GPU Streaming Pipeline for NewsSeek CLIP vs SigLIP")
    parser.add_argument("--raw-dir", type=Path, default=Path("/content/newsseek_raw"), help="Directory for raw MP4 videos")
    parser.add_argument("--output-dir", type=Path, default=Path("/content/output"), help="Output dir (set to Google Drive path for persistence)")
    parser.add_argument("--manifest", type=Path, default=Path("/content/newsseek_manifest.json"), help="Optional pre-built manifest path")
    parser.add_argument("--batch-size", type=int, default=128, help="GPU inference batch size (128 for T4/L4, 256 for A100)")
    parser.add_argument("--max-videos", type=int, default=0, help="Limit number of videos (0 = all 698 videos)")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save checkpoint every N videos (0 = disable)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint and start fresh")
    parser.add_argument("--install-deps", action="store_true", help="Install required pip packages automatically")
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle download step if videos already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_environment(install_deps=args.install_deps)
    if not args.skip_download:
        download_dataset(args.raw_dir)

    (
        all_metadata,
        processed_records,
        clip_mat,
        clip_txt_mat,
        siglip_mat,
        siglip_txt_mat,
        total_time,
        avg_fps,
    ) = run_streaming_encoding(
        raw_videos_dir=args.raw_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        max_videos=args.max_videos,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    build_and_package_stores(
        output_dir=args.output_dir,
        all_metadata=all_metadata,
        processed_records=processed_records,
        clip_mat=clip_mat,
        clip_txt_mat=clip_txt_mat,
        siglip_mat=siglip_mat,
        siglip_txt_mat=siglip_txt_mat,
        total_time=total_time,
        avg_fps=avg_fps,
    )


if __name__ == "__main__":
    main()
