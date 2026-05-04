#!/usr/bin/env python3
# sample_coco30k_val.py
#
# Sample 30k prompts from MS-COCO validation captions.
#
# Default behavior:
#   - reads annotations/captions_val2014.json
#   - samples 30,000 caption annotations without replacement
#   - writes:
#       prompts.txt                 # one caption per line
#       metadata.jsonl              # generation/eval manifest
#       metadata.csv
#       coco_subset_captions.json   # COCO-style subset
#       stats.json
#
# Recommended for COCO FID-30K prompt generation:
#   python sample_coco30k_val.py \
#     --ann-dir annotations \
#     --n 30000 \
#     --seed 42 \
#     --mode captions \
#     --out-dir coco_val2014_30k_seed42

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def stable_hash(seed: int, *parts: Any) -> str:
    """
    Deterministic pseudo-random key.
    This avoids dependency on Python/numpy RNG implementation details.
    """
    msg = "|".join([str(seed)] + [str(p) for p in parts])
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()


def detect_caption_ann_file(ann_dir: Path) -> Path:
    """
    Prefer COCO 2014 validation captions, because COCO FID-30K is usually
    built from MS-COCO 2014 val captions.
    """
    candidates = [
        ann_dir / "captions_val2014.json",
        ann_dir / "captions_val2017.json",
        ann_dir / "captions_val.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    found = sorted(ann_dir.glob("*caption*val*.json")) + sorted(
        ann_dir.glob("*captions*val*.json")
    )
    if found:
        return found[0]

    raise FileNotFoundError(
        f"No validation caption annotation file found in {ann_dir}. "
        "Expected e.g. annotations/captions_val2014.json"
    )


def load_coco_caption_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "annotations" not in data:
        raise ValueError(f"{path} does not contain an 'annotations' field.")
    if "images" not in data:
        raise ValueError(f"{path} does not contain an 'images' field.")

    return data


def sample_caption_annotations(
    data: Dict[str, Any],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Sample n caption annotations without replacement.

    This is the default for T2I COCO-FID-30K prompt sampling:
    one generated image per sampled caption.
    Multiple sampled captions may correspond to the same COCO image.
    """
    anns = [
        ann
        for ann in data["annotations"]
        if isinstance(ann.get("caption", None), str) and ann["caption"].strip()
    ]

    if len(anns) < n:
        raise ValueError(
            f"Only {len(anns)} caption annotations are available, but n={n}. "
            "For 30K without replacement, use captions_val2014.json rather than val2017."
        )

    ranked = sorted(
        anns,
        key=lambda ann: stable_hash(
            seed,
            "caption",
            ann.get("id"),
            ann.get("image_id"),
            ann.get("caption", ""),
        ),
    )
    return ranked[:n]


def sample_unique_images_with_one_caption(
    data: Dict[str, Any],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Sample n unique COCO images, then choose one caption for each image.

    This is not the usual caption-level FID-30K prompt protocol, but can be
    useful when you specifically want 30K unique image ids.
    """
    image_to_anns: Dict[int, List[Dict[str, Any]]] = {}
    for ann in data["annotations"]:
        caption = ann.get("caption", "")
        image_id = ann.get("image_id", None)
        if image_id is None or not isinstance(caption, str) or not caption.strip():
            continue
        image_to_anns.setdefault(int(image_id), []).append(ann)

    image_ids = list(image_to_anns.keys())
    if len(image_ids) < n:
        raise ValueError(
            f"Only {len(image_ids)} unique images with captions are available, "
            f"but n={n}. Use COCO 2014 val if you need 30K unique images."
        )

    sampled_image_ids = sorted(
        image_ids,
        key=lambda image_id: stable_hash(seed, "image", image_id),
    )[:n]

    selected: List[Dict[str, Any]] = []
    for image_id in sampled_image_ids:
        anns = image_to_anns[image_id]
        chosen = sorted(
            anns,
            key=lambda ann: stable_hash(
                seed,
                "caption_for_image",
                image_id,
                ann.get("id"),
                ann.get("caption", ""),
            ),
        )[0]
        selected.append(chosen)

    return selected


def build_rows(
    selected_anns: List[Dict[str, Any]],
    data: Dict[str, Any],
    source_file: Path,
) -> List[Dict[str, Any]]:
    image_by_id = {int(img["id"]): img for img in data["images"]}

    rows: List[Dict[str, Any]] = []
    for sample_id, ann in enumerate(selected_anns):
        image_id = int(ann["image_id"])
        img = image_by_id.get(image_id, {})

        rows.append(
            {
                "sample_id": sample_id,
                "output_file": f"{sample_id:06d}.png",
                "caption": ann["caption"].strip(),
                "ann_id": ann.get("id"),
                "image_id": image_id,
                "file_name": img.get("file_name"),
                "height": img.get("height"),
                "width": img.get("width"),
                "coco_url": img.get("coco_url"),
                "flickr_url": img.get("flickr_url"),
                "source_annotation_file": str(source_file),
            }
        )

    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_prompts_txt(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row["caption"].replace("\n", " ").strip() + "\n")


def write_coco_subset(
    path: Path,
    selected_anns: List[Dict[str, Any]],
    data: Dict[str, Any],
) -> None:
    selected_image_ids = {int(ann["image_id"]) for ann in selected_anns}
    selected_images = [
        img for img in data["images"] if int(img["id"]) in selected_image_ids
    ]

    subset: Dict[str, Any] = {}
    for key in ["info", "licenses"]:
        if key in data:
            subset[key] = data[key]

    subset["images"] = selected_images
    subset["annotations"] = selected_anns

    with path.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample 30K prompts from MS-COCO validation captions."
    )
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=Path("annotations"),
        help="Directory containing COCO annotation JSON files.",
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=None,
        help="Explicit caption annotation file. Example: annotations/captions_val2014.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("coco_val_30k"),
        help="Output directory.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=30000,
        help="Number of samples to draw.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed.",
    )
    parser.add_argument(
        "--mode",
        choices=["captions", "images"],
        default="captions",
        help=(
            "'captions': sample caption annotations; default for T2I FID-30K. "
            "'images': sample unique images and choose one caption per image."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ann_file = args.ann_file if args.ann_file is not None else detect_caption_ann_file(args.ann_dir)
    ann_file = ann_file.resolve()

    data = load_coco_caption_json(ann_file)

    if args.mode == "captions":
        selected_anns = sample_caption_annotations(data, n=args.n, seed=args.seed)
    else:
        selected_anns = sample_unique_images_with_one_caption(data, n=args.n, seed=args.seed)

    rows = build_rows(selected_anns, data, ann_file)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_prompts_txt(args.out_dir / "prompts.txt", rows)
    write_jsonl(args.out_dir / "metadata.jsonl", rows)
    write_csv(args.out_dir / "metadata.csv", rows)
    write_coco_subset(args.out_dir / "coco_subset_captions.json", selected_anns, data)

    unique_image_ids = {row["image_id"] for row in rows}
    stats = {
        "source_annotation_file": str(ann_file),
        "mode": args.mode,
        "seed": args.seed,
        "requested_n": args.n,
        "sampled_n": len(rows),
        "source_num_images": len(data["images"]),
        "source_num_annotations": len(data["annotations"]),
        "sampled_unique_images": len(unique_image_ids),
        "sampled_duplicate_image_count": len(rows) - len(unique_image_ids),
        "outputs": {
            "prompts": str(args.out_dir / "prompts.txt"),
            "metadata_jsonl": str(args.out_dir / "metadata.jsonl"),
            "metadata_csv": str(args.out_dir / "metadata.csv"),
            "coco_subset": str(args.out_dir / "coco_subset_captions.json"),
        },
    }

    with (args.out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()