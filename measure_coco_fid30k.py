#!/usr/bin/env python3
"""
Measure zero-shot MS-COCO FID-30K.

Default protocol:
  generated images: results/
  real reference:   full MS-COCO val2014 images
  preprocessing:   resize both generated and real images to 256x256
  metric:          pytorch-fid, pool3/Inception 2048 features

Example:
  python measure_coco_fid30k.py \
    --gen-dir results \
    --coco-val-dir /path/to/val2014 \
    --work-dir fid_coco30k \
    --device cuda:0

If you already have the public/precomputed real stats:
  python measure_coco_fid30k.py \
    --gen-dir results \
    --real-stats ./data/mscoco_val2014_41k_full/real_im256.npz \
    --work-dir fid_coco30k \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


try:
    RESAMPLE = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE = Image.BICUBIC


def list_images(root: Path, recursive: bool = True) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")

    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = sorted([p for p in iterator if p.is_file() and p.suffix.lower() in IMG_EXTS])
    return paths


def maybe_resolve_coco_val_dir(path: Path) -> Path:
    """
    Allows either:
      --coco-val-dir /data/coco/val2014
    or:
      --coco-val-dir /data/coco
    """
    if (path / "val2014").is_dir():
        return path / "val2014"
    if (path / "val2017").is_dir():
        return path / "val2017"
    return path


def load_metadata_file_names(metadata_jsonl: Path, unique: bool) -> List[str]:
    """
    Reads the metadata.jsonl produced by the previous sampling script.
    Expected fields include: file_name, image_id, caption, output_file.
    """
    if not metadata_jsonl.exists():
        raise FileNotFoundError(f"metadata.jsonl not found: {metadata_jsonl}")

    file_names = []
    with metadata_jsonl.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "file_name" not in row or row["file_name"] is None:
                raise ValueError(f"Line {line_idx} has no file_name field.")
            file_names.append(str(row["file_name"]))

    if unique:
        file_names = list(OrderedDict((x, None) for x in file_names).keys())

    return file_names


def resolve_reference_paths(
    reference_mode: str,
    coco_val_dir: Path,
    metadata_jsonl: Optional[Path],
) -> List[Path]:
    """
    reference_mode:
      full-val:
        Use all COCO val images. Recommended for common COCO-FID-30K reporting.
      sampled-unique:
        Use unique COCO real images corresponding to sampled captions.
        Not the common full-val protocol, but sometimes useful for diagnostics.
      sampled-with-repetition:
        Use one real image per sampled caption, including duplicate real images.
        Not recommended for paper-level FID.
    """
    coco_val_dir = maybe_resolve_coco_val_dir(coco_val_dir)

    if reference_mode == "full-val":
        ref_paths = list_images(coco_val_dir, recursive=False)
        return ref_paths

    if metadata_jsonl is None:
        raise ValueError(f"--metadata-jsonl is required for reference_mode={reference_mode}")

    unique = reference_mode == "sampled-unique"
    file_names = load_metadata_file_names(metadata_jsonl, unique=unique)

    ref_paths = []
    missing = []
    for name in file_names:
        p = coco_val_dir / name
        if not p.exists():
            missing.append(str(p))
        ref_paths.append(p)

    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} referenced COCO images are missing. First examples:\n{preview}"
        )

    return ref_paths


def preprocess_images(
    src_paths: List[Path],
    out_dir: Path,
    image_size: int,
    force: bool,
    desc: str,
) -> Path:
    """
    Creates a clean directory of RGB PNG images resized to image_size x image_size.
    Direct square resize is used to match the common MS-COCO 256x256 FID pipeline.
    """
    if image_size <= 0:
        raise ValueError("--image-size must be positive, e.g. 256")

    if out_dir.exists():
        existing = list_images(out_dir, recursive=False)
        if len(existing) == len(src_paths) and not force:
            print(f"[skip] {desc}: {out_dir} already has {len(existing)} images.")
            return out_dir

        if not force:
            raise RuntimeError(
                f"{out_dir} already exists but has {len(existing)} images; "
                f"expected {len(src_paths)}. Use --force to rebuild it."
            )

        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(tqdm(src_paths, desc=desc)):
        dst = out_dir / f"{i:06d}.png"
        with Image.open(src) as im:
            im = im.convert("RGB")
            im = im.resize((image_size, image_size), resample=RESAMPLE)
            im.save(dst, format="PNG")

    return out_dir


def run_cmd(cmd: List[str], capture: bool = False) -> str:
    print("\n$ " + " ".join(cmd))
    completed = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if capture:
        assert completed.stdout is not None
        print(completed.stdout)
        return completed.stdout
    return ""


def save_fid_stats(
    image_dir: Path,
    stats_path: Path,
    device: str,
    batch_size: int,
    force: bool,
) -> Path:
    if stats_path.exists() and not force:
        print(f"[skip] stats exist: {stats_path}")
        return stats_path

    stats_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "pytorch_fid",
        "--save-stats",
        str(image_dir),
        str(stats_path),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
    ]
    run_cmd(cmd, capture=False)
    return stats_path


def compute_fid(
    real_stats: Path,
    gen_stats: Path,
    device: str,
    batch_size: int,
) -> float:
    cmd = [
        sys.executable,
        "-m",
        "pytorch_fid",
        str(real_stats),
        str(gen_stats),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
    ]
    stdout = run_cmd(cmd, capture=True)

    # pytorch-fid usually prints: "FID:  12.345..."
    m = re.search(r"FID:\s*([0-9eE+\-.]+)", stdout)
    if m:
        return float(m.group(1))

    # Fallback: parse the last floating-point number in stdout.
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", stdout)
    if not nums:
        raise RuntimeError(f"Could not parse FID from output:\n{stdout}")
    return float(nums[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gen-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing the 30K generated images.",
    )
    parser.add_argument(
        "--coco-val-dir",
        type=Path,
        default=None,
        help="Directory containing COCO validation images, e.g. /path/to/val2014.",
    )
    parser.add_argument(
        "--metadata-jsonl",
        type=Path,
        default=None,
        help="metadata.jsonl from the 30K sampling script. Only needed for sampled reference modes.",
    )
    parser.add_argument(
        "--real-stats",
        type=Path,
        default=None,
        help="Optional precomputed real COCO FID stats, e.g. real_im256.npz.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=["full-val", "sampled-unique", "sampled-with-repetition"],
        default="full-val",
        help="Use full-val for standard COCO-FID-30K reporting.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("fid_coco30k"),
        help="Working directory for resized images, cached stats, and result JSON.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resize images to this square size before FID. Default: 256.",
    )
    parser.add_argument(
        "--expected-gen-count",
        type=int,
        default=30000,
        help="Expected number of generated images. Set <=0 to disable check.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cuda:0, cuda:1, or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for Inception feature extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild resized image folders and recompute stats.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generated images
    gen_paths = list_images(args.gen_dir, recursive=True)
    print(f"[generated] found {len(gen_paths)} images in {args.gen_dir}")

    if args.expected_gen_count > 0 and len(gen_paths) != args.expected_gen_count:
        raise ValueError(
            f"Expected {args.expected_gen_count} generated images, "
            f"but found {len(gen_paths)} in {args.gen_dir}."
        )

    gen_im_dir = preprocess_images(
        src_paths=gen_paths,
        out_dir=args.work_dir / f"generated_im{args.image_size}",
        image_size=args.image_size,
        force=args.force,
        desc="resize generated",
    )

    gen_stats = save_fid_stats(
        image_dir=gen_im_dir,
        stats_path=args.work_dir / f"generated_im{args.image_size}.npz",
        device=args.device,
        batch_size=args.batch_size,
        force=args.force,
    )

    # 2. Real COCO stats
    if args.real_stats is not None:
        real_stats = args.real_stats
        if not real_stats.exists():
            raise FileNotFoundError(f"--real-stats not found: {real_stats}")
        print(f"[real] using precomputed stats: {real_stats}")
        real_ref_count = None
        real_im_dir = None
    else:
        if args.coco_val_dir is None:
            raise ValueError("Either --real-stats or --coco-val-dir must be provided.")

        ref_paths = resolve_reference_paths(
            reference_mode=args.reference_mode,
            coco_val_dir=args.coco_val_dir,
            metadata_jsonl=args.metadata_jsonl,
        )
        real_ref_count = len(ref_paths)
        print(f"[real] reference_mode={args.reference_mode}, found {real_ref_count} images")

        if args.reference_mode == "full-val" and real_ref_count != 40504:
            print(
                "[warning] full-val COCO val2014 usually has 40,504 images. "
                f"Current count: {real_ref_count}. Check --coco-val-dir."
            )

        real_im_dir = preprocess_images(
            src_paths=ref_paths,
            out_dir=args.work_dir / f"real_{args.reference_mode}_im{args.image_size}",
            image_size=args.image_size,
            force=args.force,
            desc="resize real COCO",
        )

        real_stats = save_fid_stats(
            image_dir=real_im_dir,
            stats_path=args.work_dir / f"real_{args.reference_mode}_im{args.image_size}.npz",
            device=args.device,
            batch_size=args.batch_size,
            force=args.force,
        )

    # 3. FID
    fid_value = compute_fid(
        real_stats=real_stats,
        gen_stats=gen_stats,
        device=args.device,
        batch_size=args.batch_size,
    )

    result = {
        "fid": fid_value,
        "protocol": "zero-shot MS-COCO FID-30K",
        "generated_dir": str(args.gen_dir),
        "generated_count": len(gen_paths),
        "reference_mode": args.reference_mode if args.real_stats is None else "precomputed-real-stats",
        "real_reference_count": real_ref_count,
        "image_size": args.image_size,
        "real_stats": str(real_stats),
        "generated_stats": str(gen_stats),
        "device": args.device,
        "batch_size": args.batch_size,
    }

    result_path = args.work_dir / "fid_result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n==============================")
    print(f"FID: {fid_value:.6f}")
    print(f"Saved: {result_path}")
    print("==============================")


if __name__ == "__main__":
    main()