"""Colab GPU End-to-End Execution Script for NewsSeek Dataset (JPEG Keyframes + FAISS).

Steps executed on Colab VM:
1. Setup environment and configure Kaggle credentials safely (from env or ~/.kaggle).
2. Download `namnguynnnn/newsseek-core-v1` at datacenter speed and clean up raw zip.
3. Multi-threaded video decoding & scene-aware keyframe extraction (Tiered policy -> JPEG).
4. GPU accelerated embedding extraction for CLIP & SigLIP (FP16 autocast, proper SigLIP padding).
5. Build FAISS IndexFlatIP stores (`storage/newsseek/{clip,siglip}/`) + `newsseek_vector_stores.zip`.
6. Full retrieval benchmark (Video-wise Max-Pooling, Recall@K, MRR, Modality Gap, Multilingual).

Note: For Colab Free / limited disk space, prefer `scripts/colab_run_newsseek_a100.py`
which performs in-memory frame decoding without writing JPEG files to disk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def extract_tensor_features(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    if hasattr(output, "__getitem__"):
        return output[0]
    return output


# ==============================================================================
# 1. Step 1: Configure Kaggle Credentials & Download Dataset
# ==============================================================================
def setup_kaggle_credentials() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    access_token_file = kaggle_dir / "access_token"
    kaggle_json_file = kaggle_dir / "kaggle.json"
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        access_token_file.write_text(token, encoding="utf-8")
        os.chmod(str(access_token_file), 0o600)
        print("Configured Kaggle credentials from KAGGLE_API_TOKEN.")
    elif access_token_file.exists():
        os.chmod(str(access_token_file), 0o600)
        print("Using existing Kaggle access token.")
    elif kaggle_json_file.exists():
        os.chmod(str(kaggle_json_file), 0o600)
        print("Using existing Kaggle kaggle.json.")
    else:
        print("Note: Set KAGGLE_API_TOKEN environment variable or ~/.kaggle/access_token.")


def download_dataset(dataset_dir: Path) -> Path:
    setup_kaggle_credentials()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_candidates = [dataset_dir / "metadata" / "videos.csv", dataset_dir / "videos.csv"]
    csv_path = next((p for p in csv_candidates if p.exists()), None)

    if csv_path is None:
        print("[Step 1] Downloading namnguynnnn/newsseek-core-v1 from Kaggle...")
        res = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "namnguynnnn/newsseek-core-v1", "--unzip", "-p", str(dataset_dir)],
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Kaggle download failed with exit code {res.returncode}.")
        for z in dataset_dir.glob("*.zip"):
            try:
                z.unlink()
            except OSError:
                pass
        found = list(dataset_dir.rglob("videos.csv"))
        if not found:
            raise FileNotFoundError(f"videos.csv not found in {dataset_dir} after download.")
        csv_path = found[0]
    else:
        print("[Step 1] Dataset already downloaded at", dataset_dir)
    return csv_path


# ==============================================================================
# 2. Step 2: Scene Detection & Keyframe Extraction
# ==============================================================================
def keyframe_count_tiered(duration_s: float) -> int:
    if duration_s <= 3.0:
        return 1
    if duration_s <= 10.0:
        return 3
    return 5


def detect_scenes_fast(cap, fps: float, num_frames: int, threshold: float = 0.35):
    if num_frames <= 0:
        return [(0, 0)]
    step = max(1, int(fps * 0.25))
    prev_hist = None
    scene_start = 0
    max_frames = int(25.0 * fps)
    scenes = []

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


def select_keyframes(scenes, fps):
    selected = []
    for idx, (s_frame, e_frame) in enumerate(scenes):
        dur_frames = e_frame - s_frame + 1
        if dur_frames <= 0:
            continue
        count = min(dur_frames, keyframe_count_tiered(dur_frames / fps))
        pts = sorted({
            min(e_frame, max(s_frame, s_frame + round((i + 0.5) * dur_frames / count)))
            for i in range(count)
        })
        for p in pts:
            selected.append({"frame_number": int(p), "timestamp_sec": round(p / fps, 3), "scene_index": idx})
    return selected


def process_single_video(row_data):
    row, video_path, out_dir = row_data
    vid = str(row["video_id"]).strip()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or math.isnan(fps):
        fps = 25.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = num_frames / fps if fps > 0 else 0.0

    scenes = detect_scenes_fast(cap, fps, num_frames)
    keyframes = select_keyframes(scenes, fps)

    video_folder = out_dir / vid
    kf_dir = video_folder / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    for kf in keyframes:
        f_no = kf["frame_number"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        if max(h, w) > 720:
            sc = 720 / max(h, w)
            frame = cv2.resize(frame, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)

        fname = f"frame_{f_no:06d}.jpg"
        cv2.imwrite(str(kf_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        extracted.append({
            "frame_number": f_no,
            "timestamp_sec": kf["timestamp_sec"],
            "scene_index": kf["scene_index"],
            "filename": fname,
            "relative_path": f"{vid}/keyframes/{fname}",
            "full_path": str(kf_dir / fname),
        })
    cap.release()

    if not extracted:
        return None

    meta = {
        "video_id": vid,
        "title": str(row.get("title", "")),
        "language": str(row.get("language", "en")).strip().lower() or "en",
        "source_name": str(row.get("source_name", "")),
        "duration_seconds": dur,
        "num_scenes": len(scenes),
        "num_keyframes": len(extracted),
        "keyframes": extracted,
    }

    with open(video_folder / "keyframes.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


# ==============================================================================
# 3. Step 3: GPU Batch Embeddings
# ==============================================================================
def extract_embeddings_on_gpu(model_id: str, all_keyframes: List[Dict[str, Any]], video_titles: List[str], device: str, batch_size: int = 128):
    print(f"\n--> Loading {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    from PIL import Image

    is_siglip = "siglip" in model_id.lower()
    autocast_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)) if device == "cuda" else nullcontext

    # 1. Encode Keyframes
    img_embeds = []
    t_start = time.time()
    with torch.inference_mode():
        for i in range(0, len(all_keyframes), batch_size):
            batch_slice = all_keyframes[i : i + batch_size]
            pil_imgs = [Image.open(b["full_path"]).convert("RGB") for b in batch_slice]
            inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
            with autocast_ctx():
                feats = extract_tensor_features(model.get_image_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            img_embeds.append(feats.cpu().float())

    img_tensor = torch.cat(img_embeds, dim=0)
    img_time = time.time() - t_start
    print(f"    Encoded {len(img_tensor)} keyframes in {img_time:.2f}s ({len(img_tensor)/max(img_time, 1e-3):.1f} fps)!")

    # 2. Encode Unique Video Titles (SigLIP requires padding='max_length')
    txt_embeds = []
    text_padding = "max_length" if is_siglip else True
    with torch.inference_mode():
        for i in range(0, len(video_titles), batch_size):
            batch_titles = video_titles[i : i + batch_size]
            inputs = processor(text=batch_titles, return_tensors="pt", padding=text_padding, truncation=True).to(device)
            with autocast_ctx():
                feats = extract_tensor_features(model.get_text_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            txt_embeds.append(feats.cpu().float())
    txt_tensor = torch.cat(txt_embeds, dim=0)

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return img_tensor, txt_tensor


# ==============================================================================
# 4. Step 4: Video-Wise Retrieval Benchmark Evaluation
# ==============================================================================
def evaluate_video_retrieval(img_embeds: torch.Tensor, txt_embeds: torch.Tensor, processed_videos: List[Dict[str, Any]]):
    frame_sims = torch.matmul(img_embeds, txt_embeds.T)
    n_videos = len(processed_videos)
    video_sim_matrix = torch.zeros((n_videos, n_videos))

    kf_start = 0
    for v_idx, v in enumerate(processed_videos):
        n_kf = v["num_keyframes"]
        if n_kf > 0:
            v_kf_sims = frame_sims[kf_start : kf_start + n_kf]
            video_sim_matrix[v_idx] = v_kf_sims.max(dim=0).values
        kf_start += n_kf

    k_vals = [1, 5, 10]
    top_max = min(max(k_vals), n_videos)
    _, topk_v = video_sim_matrix.topk(top_max, dim=0, largest=True)
    targets = torch.arange(n_videos).unsqueeze(0).expand_as(topk_v)
    matches = topk_v == targets

    res = {}
    for k in k_vals:
        eff_k = min(k, top_max)
        res[f"R@{k}"] = round((matches[:eff_k].any(dim=0).float().mean().item()) * 100.0, 2)

    # Exact MRR over full sorted ranking
    _, full_sorted = video_sim_matrix.sort(dim=0, descending=True)
    targets_full = torch.arange(n_videos).unsqueeze(0).expand_as(full_sorted)
    ranks = (full_sorted == targets_full).nonzero(as_tuple=False)[:, 0] + 1
    res["MRR"] = round((1.0 / ranks.float()).mean().item(), 4)

    res["mR"] = round(float(np.mean([res[f"R@{k}"] for k in k_vals])), 2)
    gap = torch.norm(img_embeds.mean(dim=0) - txt_embeds.mean(dim=0), p=2).item()
    res["modality_gap"] = round(float(gap), 4)

    return res, video_sim_matrix


# ==============================================================================
# 5. Step 5: Save FAISS Vector Stores for Local FastAPI Server
# ==============================================================================
def save_faiss_vector_stores(
    output_dir: Path,
    clip_img: torch.Tensor,
    siglip_img: torch.Tensor,
    processed_videos: List[Dict[str, Any]],
    benchmark_json_path: Path,
) -> Path:
    import faiss

    storage_dir = output_dir / "storage" / "newsseek"
    clip_dir = storage_dir / "clip"
    siglip_dir = storage_dir / "siglip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    siglip_dir.mkdir(parents=True, exist_ok=True)

    metadata_items = []
    for v in processed_videos:
        for kf in v["keyframes"]:
            metadata_items.append(
                {
                    "id": f"newsseek_{len(metadata_items)}",
                    "kf_idx": len(metadata_items),
                    "video_id": v["video_id"],
                    "frame_number": kf["frame_number"],
                    "timestamp_sec": kf["timestamp_sec"],
                    "scene_index": kf["scene_index"],
                    "relative_path": kf["relative_path"],
                    "video_title": v["title"],
                    "language": v["language"],
                }
            )

    for model_name, tensor, store_dir in [
        ("clip", clip_img, clip_dir),
        ("siglip", siglip_img, siglip_dir),
    ]:
        mat = np.ascontiguousarray(tensor.numpy(), dtype=np.float32)
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        faiss.write_index(index, str(store_dir / "index.faiss"))
        with open(store_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dim": int(mat.shape[1]),
                    "index_type": "FlatIP",
                    "model_name": model_name,
                    "count": len(metadata_items),
                    "items": metadata_items,
                },
                f,
                ensure_ascii=False,
            )
        print(f"Saved {model_name.upper()} FAISS store ({index.ntotal:,} vectors) -> {store_dir}")

    zip_path = output_dir / "newsseek_vector_stores.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(clip_dir / "index.faiss", "storage/newsseek/clip/index.faiss")
        zf.write(clip_dir / "metadata.json", "storage/newsseek/clip/metadata.json")
        zf.write(siglip_dir / "index.faiss", "storage/newsseek/siglip/index.faiss")
        zf.write(siglip_dir / "metadata.json", "storage/newsseek/siglip/metadata.json")
        if benchmark_json_path.exists():
            zf.write(benchmark_json_path, "newsseek_benchmark_results.json")

    print(f"Packaged FAISS vector stores -> {zip_path} ({zip_path.stat().st_size / (1024**2):.2f} MB)")
    return zip_path


# ==============================================================================
# Main Orchestration
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab Full NewsSeek Pipeline (JPEG Keyframes + FAISS)")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/content/newsseek_dataset"))
    parser.add_argument("--processed-dir", type=Path, default=Path("/content/newsseek_processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/output"), help="Output dir (can be Google Drive)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-videos", type=int, default=0, help="Limit number of videos (0 = all)")
    parser.add_argument("--skip-jpg-archive", action="store_true", help="Skip creating large JPEG zip archive")
    return parser.parse_args()


def main():
    args = parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  COLAB GPU FULL NEWSSEEK PIPELINE & BENCHMARK")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Step 1: Download
    csv_path = download_dataset(args.dataset_dir)
    video_root = args.dataset_dir / "videos"
    if not video_root.exists():
        video_root = args.dataset_dir

    # Step 2: Load CSV & parallel extract keyframes
    df = pd.read_csv(csv_path)
    print(f"Total videos in CSV: {len(df)}")

    tasks = []
    for _, row in df.iterrows():
        vid = str(row["video_id"]).strip()
        local_rel = str(row.get("local_filename", ""))
        candidates = [
            video_root / local_rel,
            video_root / "videos" / Path(local_rel).name,
            video_root / Path(local_rel).name,
            video_root / f"{vid}.mp4",
        ]
        vpath = next((c for c in candidates if c.exists()), None)
        if vpath is None:
            m = list(video_root.rglob(f"*{vid}*.mp4"))
            if m:
                vpath = m[0]
        if vpath is not None:
            tasks.append((row, vpath, args.processed_dir))

    if args.max_videos > 0:
        tasks = tasks[: args.max_videos]

    print(f"Found {len(tasks)} / {len(df)} video files. Starting multi-threaded extraction...")
    t0 = time.time()
    processed_videos = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in tqdm(executor.map(process_single_video, tasks), total=len(tasks), desc="Processing Videos"):
            if res is not None:
                processed_videos.append(res)
    elapsed_extract = time.time() - t0
    total_kf = sum(v["num_keyframes"] for v in processed_videos)
    print(f"Extraction done in {elapsed_extract:.1f}s. Extracted {total_kf} keyframes from {len(processed_videos)} videos.")

    # Save manifest
    manifest_data = {
        "dataset": "newsseek-core-v1",
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": len(processed_videos),
        "total_keyframes": total_kf,
        "elapsed_seconds": round(elapsed_extract, 2),
        "videos": processed_videos,
    }
    manifest_json = args.output_dir / "newsseek_manifest.json"
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    # Flatten keyframes
    all_keyframes = [kf for v in processed_videos for kf in v["keyframes"]]
    video_titles = [v["title"] for v in processed_videos]

    # Step 3: Embeddings on GPU
    clip_img, clip_txt = extract_embeddings_on_gpu(
        "openai/clip-vit-base-patch32", all_keyframes, video_titles, device, batch_size=args.batch_size
    )
    siglip_img, siglip_txt = extract_embeddings_on_gpu(
        "google/siglip-base-patch16-224", all_keyframes, video_titles, device, batch_size=args.batch_size
    )

    # Save raw numpy embeddings
    np.save(args.output_dir / "clip_keyframes.npy", clip_img.numpy())
    np.save(args.output_dir / "clip_titles.npy", clip_txt.numpy())
    np.save(args.output_dir / "siglip_keyframes.npy", siglip_img.numpy())
    np.save(args.output_dir / "siglip_titles.npy", siglip_txt.numpy())

    # Step 4: Benchmark
    clip_res, clip_mat = evaluate_video_retrieval(clip_img, clip_txt, processed_videos)
    siglip_res, siglip_mat = evaluate_video_retrieval(siglip_img, siglip_txt, processed_videos)

    benchmark_summary = {
        "dataset": "newsseek",
        "total_videos": len(processed_videos),
        "total_keyframes": total_kf,
        "clip_vit_b32": clip_res,
        "siglip_vit_b16_224": siglip_res,
    }

    benchmark_json = args.output_dir / "newsseek_benchmark_results.json"
    with open(benchmark_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<25} | {'CLIP (ViT-B/32)':<20} | {'SigLIP (ViT-B/16)':<20}")
    print("-" * 75)
    print(f"{'Text-to-Video R@1':<25} | {clip_res['R@1']:<19.2f}% | {siglip_res['R@1']:<19.2f}%")
    print(f"{'Text-to-Video R@5':<25} | {clip_res['R@5']:<19.2f}% | {siglip_res['R@5']:<19.2f}%")
    print(f"{'Text-to-Video R@10':<25} | {clip_res['R@10']:<19.2f}% | {siglip_res['R@10']:<19.2f}%")
    print(f"{'MRR':<25} | {clip_res['MRR']:<20.4f} | {siglip_res['MRR']:<20.4f}")
    print(f"{'Modality Gap':<25} | {clip_res['modality_gap']:<20.4f} | {siglip_res['modality_gap']:<20.4f}")
    print("=" * 75)

    # Step 5: Build FAISS Vector Stores & ZIP
    save_faiss_vector_stores(args.output_dir, clip_img, siglip_img, processed_videos, benchmark_json)

    # Optional: Archive JPEGs
    if not args.skip_jpg_archive:
        zip_keyframes = args.output_dir / "newsseek_keyframes_processed"
        print(f"\nArchiving keyframes to {zip_keyframes}.zip ...")
        shutil.make_archive(str(zip_keyframes), "zip", args.processed_dir)
        print(f"Archive ready: {zip_keyframes}.zip")

    print("All tasks finished successfully!")


if __name__ == "__main__":
    main()
