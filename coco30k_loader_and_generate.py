#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            if "caption" not in row:
                raise ValueError(f"{path}:{line_no} has no 'caption' field")

            if row.get("sample_id") is None:
                row["sample_id"] = len(rows)

            row["sample_id"] = int(row["sample_id"])
            row["caption"] = str(row["caption"]).replace("\n", " ").strip()

            if not row["caption"]:
                raise ValueError(f"{path}:{line_no} has an empty caption")

            if not row.get("output_file"):
                row["output_file"] = f"{row['sample_id']:06d}.png"

            rows.append(row)

    return rows


class Coco30KPromptDataset(Dataset):
    """
    Dataset over the metadata.jsonl produced by the COCO-30K sampling script.

    Each item contains:
      - caption: text prompt for T2I generation
      - sample_id: unique index used for stable saving
      - output_file: usually 000000.png, 000001.png, ...
      - image_id / ann_id / file_name: original COCO identifiers for audit/debug

    Use sample_id/output_file for saving, not image_id, because caption-level
    sampling can include multiple captions from the same COCO image.
    """

    def __init__(
        self,
        metadata_jsonl: Path | str,
        results_dir: Optional[Path | str] = None,
        skip_existing: bool = False,
        limit: Optional[int] = None,
        sort_by_sample_id: bool = True,
    ) -> None:
        self.metadata_jsonl = Path(metadata_jsonl)
        self.results_dir = Path(results_dir) if results_dir is not None else None

        rows = read_jsonl(self.metadata_jsonl)

        if sort_by_sample_id:
            rows = sorted(rows, key=lambda r: int(r["sample_id"]))

        if limit is not None:
            rows = rows[: int(limit)]

        self._check_unique_output_files(rows)

        if skip_existing:
            if self.results_dir is None:
                raise ValueError("skip_existing=True requires results_dir")

            rows = [
                r
                for r in rows
                if not (self.results_dir / str(r["output_file"])).exists()
            ]

        self.rows = rows

    @staticmethod
    def _check_unique_output_files(rows: List[Dict[str, Any]]) -> None:
        seen = set()

        for r in rows:
            name = str(r["output_file"])

            if name in seen:
                raise ValueError(f"Duplicate output_file detected: {name}")

            seen.add(name)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]

        output_file = str(r["output_file"])
        save_path = (
            str(self.results_dir / output_file)
            if self.results_dir is not None
            else output_file
        )

        return {
            "sample_id": int(r["sample_id"]),
            "caption": str(r["caption"]),
            "output_file": output_file,
            "save_path": save_path,
            "ann_id": r.get("ann_id"),
            "image_id": r.get("image_id"),
            "file_name": r.get("file_name"),
            "source_annotation_file": r.get("source_annotation_file"),
        }


def coco_prompt_collate(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Keep strings and identifiers as Python lists.
    This is more convenient than tensorizing metadata for T2I pipelines.
    """
    keys = batch[0].keys()
    return {k: [sample.get(k) for sample in batch] for k in keys}


def build_coco30k_text_loader(
    metadata_jsonl: Path | str,
    results_dir: Path | str = "results",
    batch_size: int = 8,
    num_workers: int = 0,
    skip_existing: bool = False,
    shuffle: bool = False,
    limit: Optional[int] = None,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = Coco30KPromptDataset(
        metadata_jsonl=metadata_jsonl,
        results_dir=results_dir,
        skip_existing=skip_existing,
        limit=limit,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=coco_prompt_collate,
    )


def _pil_format(save_path: Path) -> Optional[str]:
    ext = save_path.suffix.lower()

    if ext == ".png":
        return "PNG"
    if ext in {".jpg", ".jpeg"}:
        return "JPEG"
    if ext == ".webp":
        return "WEBP"

    return None


def save_images_with_ids(
    images: List[Any],
    batch: Dict[str, List[Any]],
    manifest_path: Optional[Path | str] = None,
    overwrite: bool = False,
    common_extra: Optional[Dict[str, Any]] = None,
    per_item_extra: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Save one generated image per dataloader item, using batch['save_path'].

    images are expected to be PIL images, e.g. Diffusers output.images.
    """

    n = len(batch["caption"])

    if len(images) != n:
        raise ValueError(f"Got {len(images)} images, but batch has {n} prompts")

    if per_item_extra is not None and len(per_item_extra) != n:
        raise ValueError("per_item_extra length must match number of images")

    records: List[Dict[str, Any]] = []

    for i, image in enumerate(images):
        save_path = Path(str(batch["save_path"][i]))
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists() and not overwrite:
            status = "exists"
        else:
            tmp_path = save_path.with_name(save_path.stem + ".tmp" + save_path.suffix)
            image.save(tmp_path, format=_pil_format(save_path))
            os.replace(tmp_path, save_path)
            status = "saved"

        record = {
            "sample_id": int(batch["sample_id"][i]),
            "caption": batch["caption"][i],
            "output_file": batch["output_file"][i],
            "save_path": str(save_path),
            "ann_id": batch.get("ann_id", [None] * n)[i],
            "image_id": batch.get("image_id", [None] * n)[i],
            "file_name": batch.get("file_name", [None] * n)[i],
            "status": status,
        }

        if common_extra:
            record.update(common_extra)

        if per_item_extra:
            record.update(per_item_extra[i])

        records.append(record)

    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with manifest_path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records


def _torch_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COCO-30K prompt dataloader and optional Diffusers generation loop."
    )

    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    # Optional Diffusers generation arguments.
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    loader = build_coco30k_text_loader(
        metadata_jsonl=args.metadata_jsonl,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        skip_existing=args.skip_existing,
        shuffle=args.shuffle,
        limit=args.limit,
        pin_memory=args.device.startswith("cuda"),
    )

    print(f"Dataset items to process: {len(loader.dataset)}")

    if args.dry_run:
        batch = next(iter(loader))

        preview = []
        for i in range(min(3, len(batch["caption"]))):
            preview.append({k: batch[k][i] for k in batch.keys()})

        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return

    if args.model_id is None:
        raise ValueError(
            "No --model-id was provided. Use --dry-run for dataloader test, "
            "or provide a Diffusers model id for generation."
        )

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=_torch_dtype(args.dtype),
    ).to(args.device)

    pipe.set_progress_bar_config(disable=True)

    manifest_path = args.results_dir / "generated_manifest.jsonl"

    common_extra = {
        "model_id": args.model_id,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
    }

    for batch in tqdm(loader, total=len(loader), desc="generate"):
        prompts = batch["caption"]

        # Per-sample seeds make the result stable against batch-size changes.
        generators = [
            torch.Generator(device=args.device).manual_seed(args.seed + int(sample_id))
            for sample_id in batch["sample_id"]
        ]

        with torch.inference_mode():
            output = pipe(
                prompt=prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generators,
            )

        per_item_extra = [
            {"seed": args.seed + int(sample_id)}
            for sample_id in batch["sample_id"]
        ]

        save_images_with_ids(
            images=output.images,
            batch=batch,
            manifest_path=manifest_path,
            overwrite=args.overwrite,
            common_extra=common_extra,
            per_item_extra=per_item_extra,
        )

    print(f"Saved images to: {args.results_dir}")
    print(f"Saved generation manifest to: {manifest_path}")


if __name__ == "__main__":
    main()