#!/usr/bin/env python3
"""
Batch RGB → TIR inference for TherA.

Recommended (no LLaVA weights needed): use a reference .pt cache
  python infer_custom.py --rgb-dir ./examples/rgb --output-dir ./preds \\
      --reference-cache weights/reference_caches/SUNNY.pt

On-the-fly LLaVA (optional): requires separate LLaVA checkpoints
  python infer_custom.py --rgb-dir ./examples/rgb --output-dir ./preds \\
      --llava-base-path weights/llava/llava-1.5-7b-hf \\
      --llava-lora-path weights/llava/llava-miragehd-lora

Single-image example-guided mode: see infer_example_guided.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm
from contextlib import nullcontext

from thera_paths import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MERGED_MODEL,
    DEFAULT_PRETRAINED_SD,
    setup_project_path,
)
from thera_llava import create_frozen_llava_extractor

setup_project_path()

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from torchvision import transforms
from torchvision.utils import save_image

from lavi_ip2p.llava_adapter import create_llava_adapter
from lavi_ip2p.unet_8ch import convert_unet_to_8ch

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate TIR predictions from custom RGB images")
    
    # Model paths (defaults under weights/ — see README.md)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Directory containing model.pt (TherA UNet + adapter weights)",
    )
    parser.add_argument(
        "--merged-model-path",
        type=str,
        default=str(DEFAULT_MERGED_MODEL),
        help="Directory with unet/ and adapter/ architecture configs",
    )
    parser.add_argument(
        "--pretrained-sd",
        type=str,
        default=str(DEFAULT_PRETRAINED_SD),
        help="Stable Diffusion folder with vae/ and scheduler/ subfolders",
    )
    
    # LLaVA paths (only for on-the-fly feature extraction)
    parser.add_argument(
        "--llava-base-path",
        type=str,
        default=None,
        help="Base LLaVA model. Required unless --reference-cache or --cache-dir is set.",
    )
    parser.add_argument(
        "--llava-lora-path",
        type=str,
        default=None,
        help="Optional LLaVA LoRA weights (on-the-fly mode only)",
    )
    parser.add_argument("--llava-prompt", type=str, 
                       default="How would this RGB scene appear in long-wave thermal infrared spectrum",
                       help="Prompt for LLaVA feature extraction")
    
    # Reference cache mode (alternative to on-the-fly extraction)
    parser.add_argument("--reference-cache", type=str, default=None,
                       help="Path to a reference .pt cache file. If provided, uses this fixed feature for all images instead of extracting per-image.")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Directory with per-image .pt cache files (matched by filename stem). Alternative to --reference-cache.")
    
    # Data
    parser.add_argument("--rgb-dir", type=str, required=True,
                       help="Directory containing RGB images")
    parser.add_argument("--output-dir", type=str, default="custom_predictions",
                       help="Output directory for predictions")
    parser.add_argument("--recursive", action="store_true",
                       help="Search for images recursively in subdirectories")
    
    # Sampling
    parser.add_argument("--num-steps", type=int, default=100, help="DDIM sampling steps")
    parser.add_argument("--cfg-text", type=float, default=3.5, help="CFG scale for text")
    parser.add_argument("--cfg-image", type=float, default=1.5, help="CFG scale for image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Processing
    parser.add_argument("--target-size", type=int, default=None,
                       help="Resize images to this size (default: keep original, round to 32)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size (>1 for faster processing if you have enough VRAM)")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    return parser.parse_args()


def load_models(args):
    """Load VAE, UNet, Adapter, Scheduler, and optionally LLaVA Extractor"""
    print("\n" + "="*80)
    print(f"Loading models on device {args.device}...")
    print("="*80)
    
    device = torch.device(args.device)
    
    # Determine if we need LLaVA extractor
    use_cache = args.reference_cache is not None or args.cache_dir is not None
    
    # 1. VAE
    print("\n[1/5] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_sd, subfolder="vae")
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)
    print("✓ VAE loaded")
    
    # 2. Scheduler
    print("\n[2/5] Loading scheduler...")
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_sd, subfolder="scheduler")
    print("✓ Scheduler loaded")
    
    # 3. UNet (load from checkpoint)
    checkpoint_dir = Path(args.checkpoint)
    print(f"\n[3/5] Loading UNet from checkpoint: {checkpoint_dir}")
    checkpoint_path = checkpoint_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load base UNet architecture from merged model
    merged_unet_path = Path(args.merged_model_path) / "unet"
    unet = UNet2DConditionModel.from_pretrained(str(merged_unet_path))
    
    # Convert to 8-channel for image-to-image task
    print("  Converting UNet to 8-channel input...")
    unet = convert_unet_to_8ch(unet)
    
    # Load trained weights
    unet.load_state_dict(checkpoint['unet'])
    unet.requires_grad_(False)
    unet.eval()
    unet.to(device)
    
    global_step = checkpoint.get('global_step', 'unknown')
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"✓ UNet loaded (step={global_step}, epoch={epoch})")
    
    # 4. Adapter
    print("\n[4/5] Loading LLaVA Adapter...")
    merged_adapter_path = Path(args.merged_model_path) / "adapter"
    adapter = create_llava_adapter(
        adapter_path=str(merged_adapter_path),
        use_rms_norm=False,
        learnable_scale=False,
        init_scale=1.0,
        freeze_adapter=True,
    )
    
    # Load trained adapter weights if available in checkpoint
    if 'llava_adapter' in checkpoint:
        adapter.load_state_dict(checkpoint['llava_adapter'])
        print("✓ Adapter loaded from checkpoint")
    else:
        print("✓ Adapter loaded (using base weights)")
    
    adapter.requires_grad_(False)
    adapter.eval()
    adapter.to(device)
    
    # 5. LLaVA Extractor (optional - only if not using cache)
    llava_extractor = None
    if use_cache:
        print("\n[5/5] Skipping LLaVA extractor (using cached features)")
    else:
        print("\n[5/5] Loading LLaVA Feature Extractor...")
        if not args.llava_base_path:
            raise ValueError("--llava-base-path is required when not using --reference-cache or --cache-dir")
        llava_extractor = create_frozen_llava_extractor(
            llava_base_path=args.llava_base_path,
            llava_lora_path=args.llava_lora_path,
            device=args.device,
            load_8bit=False,
            load_4bit=False,
            merge_lora=True,
        )
        print("✓ LLaVA extractor loaded")
    
    return vae, unet, adapter, scheduler, llava_extractor, device


def find_images(rgb_dir: Path, recursive: bool = False) -> List[Path]:
    """Find all image files in directory"""
    if recursive:
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(rgb_dir.rglob(f"*{ext}"))
            images.extend(rgb_dir.rglob(f"*{ext.upper()}"))
        return sorted(set(images))
    else:
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(rgb_dir.glob(f"*{ext}"))
            images.extend(rgb_dir.glob(f"*{ext.upper()}"))
        return sorted(set(images))


def load_and_prepare_image(image_path: Path, target_size: int = None, device: torch.device = None):
    """Load RGB image and prepare for processing"""
    rgb_img = Image.open(image_path).convert('RGB')
    rgb_w, rgb_h = rgb_img.size
    
    # Determine target dimensions
    if target_size is not None:
        target_w = target_h = target_size
    else:
        # Ensure dimensions divisible by 32 (required for VAE)
        target_w = ((rgb_w + 31) // 32) * 32
        target_h = ((rgb_h + 31) // 32) * 32
    
    # Resize if needed
    if target_w != rgb_w or target_h != rgb_h:
        rgb_img = rgb_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    # Convert to tensor for diffusion model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
    ])
    
    rgb_tensor = transform(rgb_img).unsqueeze(0)
    if device is not None:
        rgb_tensor = rgb_tensor.to(device)
    
    return rgb_tensor, rgb_img  # Return both tensor and PIL image


def load_llava_cache(cache_path: Path, device: torch.device):
    """Load a cached LLaVA hidden state from .pt file"""
    try:
        llava_hidden = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        # For older torch versions without weights_only
        llava_hidden = torch.load(cache_path, map_location="cpu")
    
    if not isinstance(llava_hidden, torch.Tensor):
        llava_hidden = torch.tensor(llava_hidden)
    
    # Add batch dimension if needed
    if llava_hidden.ndim == 2:
        llava_hidden = llava_hidden.unsqueeze(0)
    
    return llava_hidden.to(device)


def find_cache_for_image(image_path: Path, cache_dir: Path) -> Path:
    """Find matching cache file for an image based on filename stem"""
    stem = image_path.stem
    for ext in ['.pt', '.pth']:
        cache_path = cache_dir / f"{stem}{ext}"
        if cache_path.exists():
            return cache_path
    raise FileNotFoundError(f"No cache file found for {image_path.name} in {cache_dir}")


def create_null_text(llava_tokens):
    """Create null tokens for unconditional generation"""
    return torch.zeros_like(llava_tokens)


@torch.no_grad()
def translate_image(
    vae,
    unet,
    adapter,
    scheduler,
    rgb_tensor,
    llava_hidden,
    num_steps=100,
    cfg_text=3.5,
    cfg_image=1.5,
    device=torch.device("cuda"),
):
    """Translate RGB to TIR using LLaVA features + conditional generation with CFG"""
    
    # Encode RGB
    rgb_latents = vae.encode(rgb_tensor).latent_dist.mode() * vae.config.scaling_factor
    
    # Process LLaVA features through adapter
    use_autocast = device.type == "cuda"
    llava_input = llava_hidden.to(device)
    if use_autocast:
        llava_input = llava_input.to(dtype=torch.bfloat16)
    
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16) 
        if use_autocast else nullcontext()
    )
    with autocast_ctx:
        llava_tokens = adapter(llava_input)
    llava_tokens = llava_tokens.to(device=device, dtype=next(unet.parameters()).dtype)
    
    # Create null tokens for CFG
    null_tokens = create_null_text(llava_tokens)
    
    # Initialize random noise
    latent_shape = (1, 4, rgb_latents.shape[2], rgb_latents.shape[3])
    latents = torch.randn(latent_shape, device=device, dtype=rgb_latents.dtype)
    
    # Setup scheduler
    scheduler.set_timesteps(num_steps, device=device)
    
    zeros_rgb = torch.zeros_like(rgb_latents)
    
    for t in scheduler.timesteps:
        t_batch = torch.full((1,), int(t), device=device, dtype=torch.long)
        
        # 3 forward passes for dual-CFG
        x8_full = torch.cat([latents, rgb_latents], dim=1)
        x8_no_img = torch.cat([latents, zeros_rgb], dim=1)
        
        # Full conditioning
        eps_full = unet(x8_full, t_batch, encoder_hidden_states=llava_tokens).sample
        
        # No text (image only)
        eps_no_text = unet(x8_full, t_batch, encoder_hidden_states=null_tokens).sample
        
        # No image (text only)
        eps_no_img = unet(x8_no_img, t_batch, encoder_hidden_states=llava_tokens).sample
        
        # Combine with dual-CFG
        eps = eps_full + cfg_text * (eps_full - eps_no_text) + cfg_image * (eps_full - eps_no_img)
        eps = torch.clamp(eps, -5.0, 5.0)  # Safety guard
        
        latents = scheduler.step(eps, t, latents, eta=0.0).prev_sample
    
    # Decode
    pred_tir = vae.decode(latents / vae.config.scaling_factor).sample
    pred_tir = (pred_tir / 2 + 0.5).clamp(0, 1)
    
    return pred_tir


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Determine processing mode
    use_reference = args.reference_cache is not None
    use_cache_dir = args.cache_dir is not None
    
    if use_reference and use_cache_dir:
        raise ValueError("Cannot use both --reference-cache and --cache-dir. Choose one.")
    
    print("\n" + "="*80)
    if use_reference:
        print("Custom Dataset Inference (Reference Cache Mode)")
    elif use_cache_dir:
        print("Custom Dataset Inference (Per-Image Cache Mode)")
    else:
        print("Custom Dataset Inference (On-the-fly LLaVA Features)")
    print("="*80)
    print(f"RGB directory: {args.rgb_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    
    if use_reference:
        print(f"Reference cache: {args.reference_cache}")
    elif use_cache_dir:
        print(f"Cache directory: {args.cache_dir}")
    else:
        print(f"LLaVA base: {args.llava_base_path}")
        if args.llava_lora_path:
            print(f"LLaVA LoRA: {args.llava_lora_path}")
        print(f"Prompt: {args.llava_prompt}")
    
    print(f"Output directory: {args.output_dir}")
    print(f"DDIM steps: {args.num_steps}")
    print(f"CFG scales: text={args.cfg_text}, image={args.cfg_image}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    vae, unet, adapter, scheduler, llava_extractor, device = load_models(args)
    
    # Load reference cache if provided
    reference_hidden = None
    if use_reference:
        print(f"\nLoading reference cache from: {args.reference_cache}")
        reference_hidden = load_llava_cache(Path(args.reference_cache), device)
        print(f"✓ Reference cache loaded: {reference_hidden.shape}")
    
    # Validate cache directory if provided
    cache_dir = None
    if use_cache_dir:
        cache_dir = Path(args.cache_dir)
        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
        print(f"\nUsing cache directory: {cache_dir}")
    
    # Find all RGB images
    rgb_dir = Path(args.rgb_dir)
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    
    image_files = find_images(rgb_dir, recursive=args.recursive)
    print(f"\nFound {len(image_files)} images")
    
    if len(image_files) == 0:
        print("No images found. Exiting.")
        return
    
    # Process each image
    print("\n" + "="*80)
    print("Generating predictions...")
    print("="*80)
    
    successful = 0
    failed = 0
    skipped = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load and prepare RGB image
            rgb_tensor, rgb_pil = load_and_prepare_image(
                img_path, 
                target_size=args.target_size,
                device=device
            )
            
            # Get LLaVA hidden states based on mode
            if use_reference:
                # Use the same reference cache for all images
                llava_hidden = reference_hidden
            elif use_cache_dir:
                # Find matching cache file for this image
                try:
                    cache_path = find_cache_for_image(img_path, cache_dir)
                    llava_hidden = load_llava_cache(cache_path, device)
                except FileNotFoundError as e:
                    print(f"\nWarning: {e}")
                    skipped += 1
                    continue
            else:
                # Extract features on-the-fly using LLaVA
                llava_hidden = llava_extractor.extract_hidden_states([rgb_pil], [args.llava_prompt])
            
            # Translate RGB to TIR
            pred_tir = translate_image(
                vae,
                unet,
                adapter,
                scheduler,
                rgb_tensor,
                llava_hidden,
                num_steps=args.num_steps,
                cfg_text=args.cfg_text,
                cfg_image=args.cfg_image,
                device=device,
            )
            
            # Save prediction
            # Preserve relative path structure if recursive
            if args.recursive:
                rel_path = img_path.relative_to(rgb_dir)
                save_path = output_dir / rel_path
            else:
                save_path = output_dir / img_path.name
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(pred_tir.cpu(), str(save_path))
            
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    # Print final results
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    if skipped > 0:
        print(f"Skipped (no cache): {skipped}")
    print(f"Failed: {failed}")
    if len(image_files) > 0:
        print(f"Success rate: {successful/len(image_files)*100:.1f}%")
    print(f"Predictions saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
