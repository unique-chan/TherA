#!/usr/bin/env python3
"""
Single-image example-guided RGB → TIR translation.

Mode: cached (recommended — no LLaVA weights needed)
  python infer_example_guided.py --mode cached \\
      --reference-cache weights/reference_caches/SUNNY.pt \\
      --input-image examples/rgb/scene.jpg \\
      --output preds/scene_tir.png

Mode: two-image (extract features from a reference RGB on-the-fly)
  python infer_example_guided.py --mode two-image \\
      --reference-image examples/ref/rgb.jpg \\
      --input-image examples/rgb/scene.jpg \\
      --output preds/scene_tir.png \\
      --llava-base-path weights/llava/llava-1.5-7b-hf \\
      --llava-lora-path weights/llava/llava-miragehd-lora
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from contextlib import nullcontext

import torch
from PIL import Image

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example-guided translation with two modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True, choices=["two-image", "cached"],
                       help="Translation mode: 'two-image' (extract from reference image) or 'cached' (use .pt file)")
    
    # Input/Output
    parser.add_argument("--input-image", type=str, required=True,
                       help="Path to input RGB image (Image B) to be translated")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save output TIR image")
    
    # Reference (mode-dependent)
    parser.add_argument("--reference-image", type=str, default=None,
                       help="Path to reference RGB image (Image A). Required for 'two-image' mode.")
    parser.add_argument("--reference-cache", type=str, default=None,
                       help="Path to reference .pt cache file. Required for 'cached' mode.")
    
    # Model paths (defaults under weights/ — see README.md)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Directory containing model.pt",
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
    
    # LLaVA paths (required for two-image mode)
    parser.add_argument("--llava-base-path", type=str, default=None,
                       help="Path to base LLaVA model. Required for 'two-image' mode.")
    parser.add_argument("--llava-lora-path", type=str, default=None,
                       help="Optional: Path to LLaVA LoRA weights")
    parser.add_argument("--llava-prompt", type=str,
                       default="How would this RGB scene appear in long-wave thermal infrared spectrum",
                       help="Prompt for LLaVA feature extraction")
    
    # Yechan ->
    parser.add_argument("--llava-device", type=str, default=None,
                       help="Device for LLaVA model, e.g. cuda:1 (default: same as --device)")

    # Sampling parameters
    parser.add_argument("--num-steps", type=int, default=100, help="DDIM sampling steps")
    parser.add_argument("--cfg-text", type=float, default=3.5, help="CFG scale for text")
    parser.add_argument("--cfg-image", type=float, default=1.5, help="CFG scale for image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Processing
    parser.add_argument("--target-size", type=int, default=None,
                       help="Resize images to this size (default: keep original, round to 32)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    return parser.parse_args()


def validate_args(args):
    """Validate argument combinations based on mode"""
    if args.mode == "two-image":
        if args.reference_image is None:
            print("ERROR: --reference-image required for 'two-image' mode")
            sys.exit(1)
        if args.llava_base_path is None:
            print("ERROR: --llava-base-path required for 'two-image' mode")
            sys.exit(1)
        if not Path(args.reference_image).exists():
            print(f"ERROR: Reference image not found: {args.reference_image}")
            sys.exit(1)
    
    elif args.mode == "cached":
        if args.reference_cache is None:
            print("ERROR: --reference-cache required for 'cached' mode")
            sys.exit(1)
        if not Path(args.reference_cache).exists():
            print(f"ERROR: Reference cache not found: {args.reference_cache}")
            sys.exit(1)
    
    if not Path(args.input_image).exists():
        print(f"ERROR: Input image not found: {args.input_image}")
        sys.exit(1)
    
    if not Path(args.checkpoint, "model.pt").exists():
        print(f"ERROR: Checkpoint not found: {Path(args.checkpoint) / 'model.pt'}")
        sys.exit(1)


def load_models(args, need_llava=False):
    """Load diffusion models and optionally LLaVA extractor"""
    print("\n" + "="*80)
    print(f"Loading models on device {args.device}...")
    print("="*80)
    
    device = torch.device(args.device)
    # YECHAN ->
    llava_device = args.llava_device or args.device

    checkpoint_dir = Path(args.checkpoint)
    
    # 1. Load VAE
    print("\n[1/5] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_sd, subfolder="vae")
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)
    print("✓ VAE loaded")
    
    # 2. Load Scheduler
    print("\n[2/5] Loading scheduler...")
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_sd, subfolder="scheduler")
    print("✓ Scheduler loaded")
    
    # 3. Load UNet (load from checkpoint)
    print(f"\n[3/5] Loading UNet from checkpoint: {checkpoint_dir}")
    checkpoint_path = checkpoint_dir / "model.pt"
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
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
    
    # 4. Load Adapter
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
    
    # 5. Optionally load LLaVA
    llava_extractor = None
    if need_llava:
        print("\n[5/5] Loading LLaVA Feature Extractor...")
        if not args.llava_base_path:
            print("ERROR: --llava-base-path is required for two-image mode")
            sys.exit(1)
        llava_extractor = create_frozen_llava_extractor(
            llava_base_path=args.llava_base_path,
            llava_lora_path=args.llava_lora_path,

            # YECHAN ->
            # device=args.device,
            device = llava_device,

            load_8bit=False,
            load_4bit=False,
            merge_lora=True,
        )
        print("✓ LLaVA extractor loaded")
    else:
        print("\n[5/5] Skipping LLaVA extractor (using cached features)")
    
    return vae, unet, adapter, scheduler, llava_extractor, device


def load_and_prepare_image(image_path, target_size=None, device="cuda"):
    """Load image and prepare tensor"""
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


def load_llava_cache(cache_path, device):
    """Load pre-cached LLaVA hidden states"""
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


@torch.no_grad()
def translate_image(vae, unet, adapter, scheduler, rgb_tensor, llava_hidden, 
                   num_steps=100, cfg_text=3.5, cfg_image=1.5, device="cuda"):
    """Translate RGB to TIR using LLaVA features + conditional generation with CFG"""
    
    # Yechan ->
    diffusion_device = next(vae.parameters()).device
    diffusion_dtype = next(vae.parameters()).dtype

    device = diffusion_device  # Ensure all tensors are on the same device !!!

    rgb_tensor = rgb_tensor.to(device=diffusion_device, dtype=diffusion_dtype)

    if llava_hidden is not None:
        llava_hidden = llava_hidden.to(diffusion_device)




    # Encode RGB
    rgb_latents = vae.encode(rgb_tensor).latent_dist.mode() * vae.config.scaling_factor
    batch_size = rgb_latents.shape[0]
    
    # Process LLaVA features through adapter
    use_autocast = device.type == "cuda"
    llava_input = llava_hidden.to(device)
    # If using a single cached reference for a batch of RGB images, repeat it across batch.
    if llava_input.shape[0] == 1 and batch_size > 1:
        llava_input = llava_input.expand(batch_size, *llava_input.shape[1:])
    if use_autocast:
        llava_input = llava_input.to(dtype=torch.bfloat16)
    
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16) 
        if use_autocast else nullcontext()
    )
    with autocast_ctx:
        llava_tokens = adapter(llava_input)
    llava_tokens = llava_tokens.to(device=device, dtype=next(unet.parameters()).dtype)
    # Ensure token batch matches RGB batch (some adapters may return batch=1 even if input was expanded).
    if llava_tokens.shape[0] == 1 and batch_size > 1:
        llava_tokens = llava_tokens.expand(batch_size, *llava_tokens.shape[1:])
    
    # Create null tokens for CFG
    null_tokens = torch.zeros_like(llava_tokens)
    
    # Initialize random noise
    latent_shape = (batch_size, 4, rgb_latents.shape[2], rgb_latents.shape[3])
    latents = torch.randn(latent_shape, device=device, dtype=rgb_latents.dtype)
    
    # Setup scheduler
    scheduler.set_timesteps(num_steps, device=device)
    
    zeros_rgb = torch.zeros_like(rgb_latents)
    
    for t in scheduler.timesteps:
        t_batch = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
        
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
    validate_args(args)
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("\n" + "="*80)
    print(f"Example-Guided Translation - Mode: {args.mode.upper()}")
    print("="*80)
    print(f"Input image (B): {args.input_image}")
    if args.mode == "two-image":
        print(f"Reference image (A): {args.reference_image}")
    else:
        print(f"Reference cache: {args.reference_cache}")
    print(f"Output: {args.output}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"DDIM steps: {args.num_steps}")
    print(f"CFG scales: text={args.cfg_text}, image={args.cfg_image}")
    print("="*80)
    
    # Load models
    need_llava = (args.mode == "two-image")
    vae, unet, adapter, scheduler, llava_extractor, device = load_models(args, need_llava=need_llava)
    
    # Get reference hidden states
    print("\n" + "-"*80)
    if args.mode == "two-image":
        print(f"Extracting features from reference image: {args.reference_image}")

        # Yechan ->
        # ref_tensor, ref_pil = load_and_prepare_image(
        #     args.reference_image,
        #     target_size=args.target_size,
        #     device=device
        # )
        # llava_hidden = llava_extractor.extract_hidden_states([ref_pil], [args.llava_prompt])
        ref_tensor, ref_pil = load_and_prepare_image(
            args.reference_image,
            target_size=args.target_size,
            device=None  # Load on CPU 
        )
        llava_hidden = llava_extractor.extract_hidden_states([ref_pil], [args.llava_prompt])
        llava_hidden = llava_hidden.to(device=device)

        print(f"✓ Extracted hidden states: {llava_hidden.shape}")
    else:
        print(f"Loading cached reference: {args.reference_cache}")
        llava_hidden = load_llava_cache(Path(args.reference_cache), device)
        print(f"✓ Loaded hidden states: {llava_hidden.shape}")
    print("-"*80)
    
    # Load input image (Image B)
    print(f"\nLoading input image: {args.input_image}")
    input_tensor, input_pil = load_and_prepare_image(
        args.input_image,
        target_size=args.target_size,
        device=device
    )

    print(f"✓ Input image loaded: {input_tensor.shape}")
    
    # Translate
    print("\nTranslating image...")
    pred_tir = translate_image(
        vae, unet, adapter, scheduler,
        input_tensor, llava_hidden,
        num_steps=args.num_steps,
        cfg_text=args.cfg_text,
        cfg_image=args.cfg_image,
        device=device
    )
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(pred_tir.cpu(), str(output_path))
    
    print("="*80)
    print(f"✓ Translation complete!")
    print(f"✓ Saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()

