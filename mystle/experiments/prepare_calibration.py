#!/usr/bin/env python3
"""
从 HuggingFace datasets 构造校准集：固定条数、三种图像干预（original / visual_ablated / mismatch）。
输出目录结构见 --help。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def _answer_to_str(answer) -> str:
    if isinstance(answer, list):
        return ", ".join(str(x) for x in answer) if answer else ""
    return str(answer)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare calibration images + manifest for bridge-score experiments.")
    p.add_argument(
        "--dataset",
        type=str,
        default="Ahren09/info_vqa",
        help="HuggingFace datasets name (default: Ahren09/info_vqa).",
    )
    p.add_argument("--split", type=str, default="train", help="Dataset split.")
    p.add_argument("--num-samples", type=int, default=1000, help="Number of calibration examples.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for subsampling and mismatch pairing.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/fsas/datasets/deepseek-vl2-bridge/calibration"),
        help="Root directory for images/ and manifest.jsonl.",
    )
    p.add_argument(
        "--mismatch-shift",
        type=int,
        default=1,
        help="Circular shift index for mismatch image: sample i uses image from (i+k) mod N.",
    )
    p.add_argument(
        "--image-format",
        type=str,
        choices=("png", "jpeg"),
        default="png",
        help="Output image format. jpeg is faster and smaller; png is lossless.",
    )
    p.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality when --image-format jpeg.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    orig_dir = out / "images" / "original"
    va_dir = out / "images" / "visual_ablated"
    mm_dir = out / "images" / "mismatch"
    for d in (orig_dir, va_dir, mm_dir):
        d.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    n = len(ds)
    if n < args.num_samples:
        raise ValueError(f"Dataset has only {n} rows, need {args.num_samples}")

    rng = random.Random(args.seed)
    indices = list(range(n))
    rng.shuffle(indices)
    chosen = indices[: args.num_samples]

    # Mismatch: sample i gets image from chosen[(i + shift) % N] — same length list, simple pairing
    k = args.mismatch_shift % args.num_samples
    if k == 0:
        k = 1

    ext = ".png" if args.image_format == "png" else ".jpg"

    def _save(img: Image.Image, path: Path) -> None:
        if args.image_format == "png":
            img.save(path, format="PNG", optimize=True)
        else:
            img.save(path, format="JPEG", quality=args.jpeg_quality, optimize=True)

    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for local_i in tqdm(range(args.num_samples), desc="prepare_calibration", unit="ex"):
            row_idx = chosen[local_i]
            ex = ds[row_idx]
            img: Image.Image = ex["image"]
            if not isinstance(img, Image.Image):
                img = img.convert("RGB") if hasattr(img, "convert") else Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")

            q = ex["question"]
            a = _answer_to_str(ex.get("answer", ""))
            sid = f"{local_i:05d}"
            orig_file = orig_dir / f"{sid}{ext}"
            va_file = va_dir / f"{sid}{ext}"
            mm_file = mm_dir / f"{sid}{ext}"

            w, h = img.size
            blank = Image.new("RGB", (w, h), (0, 0, 0))

            mm_src_local = (local_i + k) % args.num_samples
            mm_row_idx = chosen[mm_src_local]
            mm_img = ds[mm_row_idx]["image"]
            if isinstance(mm_img, Image.Image):
                mm_img = mm_img.convert("RGB")
            else:
                mm_img = Image.open(mm_img).convert("RGB")

            _save(img, orig_file)
            _save(blank, va_file)
            _save(mm_img, mm_file)

            rec = {
                "sample_id": sid,
                "dataset_index": int(row_idx),
                "question": q,
                "answer": a,
                "image_size": [w, h],
                "paths": {
                    "original": str(orig_file),
                    "visual_ablated": str(va_file),
                    "mismatch": str(mm_file),
                },
                "mismatch_source_sample_id": f"{mm_src_local:05d}",
                "mismatch_source_dataset_index": int(mm_row_idx),
            }
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "mismatch_shift": k,
        "image_format": args.image_format,
        "jpeg_quality": args.jpeg_quality if args.image_format == "jpeg" else None,
        "manifest": str(manifest_path),
        "output_dir": str(out),
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.num_samples} samples under {out}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
