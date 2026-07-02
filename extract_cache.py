#!/usr/bin/env python3
"""
Extract LLaVA reference caches for TherA.

This is the counterpart to the two cache-based inference modes:
  * A single reference cache (e.g. SUNNY.pt) used as a global condition
    for every image      -> feed to `infer_custom.py --reference-cache X.pt`
  * A directory of per-image caches matched by filename stem
                          -> feed to `infer_custom.py --cache-dir DIR/`

It loads the (frozen) LLaVA extractor, runs each image + prompt through it,
grabs the last hidden state (shape [L, 4096]) and saves it as a plain
`.pt` tensor that `load_llava_cache()` in the inference scripts can read
directly.

--------------------------------------------------------------------------
Reference-cache mode: one representative image -> one .pt
--------------------------------------------------------------------------
  python extract_cache.py \
      --image examples/ref/sunny_scene.jpg \
      --output weights/reference_caches/SUNNY.pt \
      --llava-base-path weights/llava-1.5-7b-hf \
      --llava-lora-path weights/TherA-VLM \
      --prompt "A bright sunny daytime scene in long-wave thermal infrared."

--------------------------------------------------------------------------
Per-image cache-dir mode: a folder of images -> a folder of .pt files
--------------------------------------------------------------------------
  python extract_cache.py \
      --image-dir examples/rgb \
      --output-dir weights/my_caches \
      --llava-base-path weights/llava-1.5-7b-hf \
      --llava-lora-path weights/TherA-VLM \
      --recursive
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

from thera_paths import DEFAULT_REFERENCE_CACHES, setup_project_path
from thera_llava import create_frozen_llava_extractor

setup_project_path()

# Same set the inference scripts recognise.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_PROMPT = "How would this RGB scene appear in long-wave thermal infrared spectrum"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract LLaVA hidden-state reference caches (.pt) for TherA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input: exactly one of these two ---
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Single representative RGB image -> one reference cache (global condition).",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory of RGB images -> one .pt per image (matched later by filename stem).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="With --image-dir, search subdirectories too (structure is preserved in --output-dir).",
    )

    # --- Output: pairs with the input choice ---
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pt path for --image mode (e.g. weights/reference_caches/SUNNY.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_REFERENCE_CACHES),
        help="Output directory for --image-dir mode.",
    )

    # --- LLaVA (required: extraction always needs LLaVA) ---
    parser.add_argument(
        "--llava-base-path",
        type=str,
        required=True,
        help="Base LLaVA model path (e.g. weights/llava-1.5-7b-hf).",
    )
    parser.add_argument(
        "--llava-lora-path",
        type=str,
        default=None,
        help="Optional LLaVA LoRA weights (e.g. weights/TherA-VLM).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt paired with every image during extraction.",
    )

    # --- Misc ---
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for the saved cache (float16 = smaller files).",
    )
    parser.add_argument(
        "--keep-batch-dim",
        action="store_true",
        help="Save as [1, L, 4096] instead of the default squeezed [L, 4096].",
    )
    parser.add_argument("--load-8bit", action="store_true", help="Load LLaVA in 8-bit (low VRAM).")
    parser.add_argument("--load-4bit", action="store_true", help="Load LLaVA in 4-bit (lowest VRAM).")
    parser.add_argument("--device", type=str, default="cuda", help="Default device (cuda/cpu).")
    parser.add_argument(
        "--llava-device",
        type=str,
        default=None,
        help="Device to load LLaVA on, e.g. 'cuda:1'. Defaults to --device. "
        "Handy for keeping the ~7B extractor on a separate GPU.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files instead of skipping them.",
    )

    return parser.parse_args()


def validate_args(args) -> str:
    """Return the chosen mode ('single' or 'dir') or raise on bad combos."""
    if bool(args.image) == bool(args.image_dir):
        raise ValueError("Provide exactly one of --image or --image-dir.")

    if args.image:
        if not args.output:
            raise ValueError("--output is required with --image.")
        return "single"

    # image-dir mode
    return "dir"


def find_images(image_dir: Path, recursive: bool) -> List[Path]:
    """Collect image files (mirrors infer_custom.find_images)."""
    globber = image_dir.rglob if recursive else image_dir.glob
    images: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(globber(f"*{ext}"))
        images.extend(globber(f"*{ext.upper()}"))
    return sorted(set(images))


def to_storage_tensor(hidden: torch.Tensor, dtype: torch.dtype, keep_batch: bool) -> torch.Tensor:
    """(B, L, 4096) on device -> CPU tensor ready to torch.save.

    Squeezes the batch dim for a single image so the file is the canonical
    [L, 4096] that load_llava_cache() expects (it re-adds the batch dim).
    """
    hidden = hidden.detach().to("cpu", dtype=dtype)
    if not keep_batch and hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden.squeeze(0)  # [1, L, 4096] -> [L, 4096]
    return hidden.contiguous()


def extract_one(extractor, image_path: Path, prompt: str) -> torch.Tensor:
    """Run one image+prompt through LLaVA and return its (1, L, 4096) hidden state."""
    pil = Image.open(image_path).convert("RGB")
    # extract_hidden_states takes a *list* of images and a *list* of prompts.
    return extractor.extract_hidden_states([pil], [prompt])


def main():
    args = parse_args()
    mode = validate_args(args)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    # LLaVA gets its own device; fall back to --device when not specified.
    llava_device = args.llava_device or args.device

    print("\n" + "=" * 80)
    print(f"TherA cache extraction ({'single reference' if mode == 'single' else 'per-image directory'})")
    print("=" * 80)
    print(f"LLaVA base : {args.llava_base_path}")
    if args.llava_lora_path:
        print(f"LLaVA LoRA : {args.llava_lora_path}")
    print(f"LLaVA device: {llava_device}")
    print(f"Prompt     : {args.prompt}")
    print(f"Store dtype: {args.dtype}  (batch dim {'kept' if args.keep_batch_dim else 'squeezed'})")
    print("=" * 80)

    # Load the frozen extractor once (this is the heavy step: ~7B params).
    print("\nLoading frozen LLaVA extractor...")
    extractor = create_frozen_llava_extractor(
        llava_base_path=args.llava_base_path,
        llava_lora_path=args.llava_lora_path,
        device=llava_device,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        merge_lora=True,
    )
    print("✓ Extractor ready")

    # ----------------------------------------------------------------- single
    if mode == "single":
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nExtracting from: {image_path}")
        hidden = extract_one(extractor, image_path, args.prompt)
        tensor = to_storage_tensor(hidden, dtype, args.keep_batch_dim)
        torch.save(tensor, str(output_path))
        print(f"✓ Saved reference cache: {output_path}  shape={tuple(tensor.shape)}")
        return

    # -------------------------------------------------------------- directory
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(image_dir, args.recursive)
    print(f"\nFound {len(images)} images under {image_dir}")
    if not images:
        print("No images found. Exiting.")
        return

    saved, skipped, failed = 0, 0, 0
    for img_path in tqdm(images, desc="Extracting caches"):
        # Preserve subdir structure when recursive, matching infer_custom output layout.
        if args.recursive:
            rel = img_path.relative_to(image_dir).with_suffix(".pt")
            out_path = output_dir / rel
        else:
            out_path = output_dir / f"{img_path.stem}.pt"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            hidden = extract_one(extractor, img_path, args.prompt)
            tensor = to_storage_tensor(hidden, dtype, args.keep_batch_dim)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor, str(out_path))
            saved += 1
        except Exception as e:  # noqa: BLE001 - keep going on per-image errors
            print(f"\nError on {img_path}: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Total : {len(images)}")
    print(f"Saved : {saved}")
    if skipped:
        print(f"Skipped (exists, no --overwrite): {skipped}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Caches written to: {output_dir}")
    print("Use with:  python infer_custom.py --cache-dir", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
