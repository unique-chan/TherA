"""
Validation Utilities for Frozen VLM-IP2P Training

Handles:
- Loading pre-computed vanilla IP2P predictions
- Generating predictions with EMA weights
- Creating 4-column comparison grids
- DDIM/PNDM inference with proper CFG
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid
from torchvision import transforms
from diffusers import DDIMScheduler, PNDMScheduler, DDPMScheduler

from llava.model.cfg_utils import prepare_cfg_inference_batch, apply_extended_cfg


def load_validation_cache(cache_dir: str) -> Tuple[List[Dict], int]:
    """
    Load pre-computed validation cache.
    
    Args:
        cache_dir: Path to validation cache (e.g., outputs/validation_cache/vanilla_ip2p)
    
    Returns:
        samples: List of sample metadata
        num_samples: Number of validation samples
    """
    samples_json = os.path.join(cache_dir, "validation_samples.json")
    
    if not os.path.exists(samples_json):
        raise FileNotFoundError(f"Validation cache not found: {samples_json}")
    
    with open(samples_json, 'r') as f:
        samples = json.load(f)
    
    return samples, len(samples)


def load_cached_images(cache_dir: str, sample_idx: int) -> Dict[str, Image.Image]:
    """
    Load pre-computed images for a validation sample.
    
    Args:
        cache_dir: Path to validation cache
        sample_idx: Sample index
    
    Returns:
        images: Dict with keys 'rgb', 'pred_vanilla_ip2p', 'gt'
    """
    sample_dir = os.path.join(cache_dir, f"sample_{sample_idx:03d}")
    
    images = {}
    for key in ['rgb', 'pred_vanilla_ip2p', 'gt']:
        img_path = os.path.join(sample_dir, f"{key}.png")
        if os.path.exists(img_path):
            images[key] = Image.open(img_path).convert('RGB')
        else:
            raise FileNotFoundError(f"Cached image not found: {img_path}")
    
    return images


def create_comparison_grid(
    rgb: torch.Tensor,
    pred_ours: torch.Tensor,
    gt: torch.Tensor,
    nrow: int = 3,
) -> torch.Tensor:
    """
    Create 3-column comparison grid: [RGB | Ours | GT]
    
    Args:
        rgb: Input RGB image (C, H, W) in [-1, 1]
        pred_ours: Our model prediction (C, H, W) in [-1, 1]
        gt: Ground truth thermal (C, H, W) in [-1, 1]
        nrow: Number of columns
    
    Returns:
        grid: Grid image (C, H, W*nrow) in [0, 1]
    """
    # Stack images horizontally
    images = torch.stack([rgb, pred_ours, gt], dim=0)
    
    # Create grid (make_grid handles normalization)
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    
    return grid


@torch.no_grad()
def generate_with_cfg(
    unet,
    scheduler,
    rgb_latents: torch.Tensor,
    text_embeds: torch.Tensor,
    vlm_tokens: torch.Tensor,
    vae,
    num_inference_steps: int = 100,
    image_guidance_scale: float = 1.5,
    text_guidance_scale: float = 7.5,
    vlm_guidance_scale: float = 1.5,
    scheduler_type: str = "ddim",
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate thermal image with full CFG.
    
    Args:
        unet: UNet model
        scheduler: Base scheduler (will be replaced with specified type)
        rgb_latents: RGB condition latents (B, C, H, W)
        text_embeds: Text embeddings (B, N_text, D)
        vlm_tokens: VLM tokens (B, N_vlm, D)
        vae: VAE for decoding
        num_inference_steps: Number of denoising steps
        image_guidance_scale: Image CFG scale
        text_guidance_scale: Text CFG scale
        vlm_guidance_scale: VLM CFG scale
        scheduler_type: "ddim", "pndm", or "ddpm"
        device: Device
    
    Returns:
        pred_image: Generated image (B, C, H, W) in [-1, 1]
    """
    batch_size = rgb_latents.size(0)
    
    # Initialize scheduler
    if scheduler_type == "ddim":
        inference_scheduler = DDIMScheduler.from_config(scheduler.config)
    elif scheduler_type == "pndm":
        inference_scheduler = PNDMScheduler.from_config(scheduler.config)
    else:  # ddpm
        inference_scheduler = DDPMScheduler.from_config(scheduler.config)
    
    inference_scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Initialize latents
    latent_shape = rgb_latents.shape
    latents = torch.randn(latent_shape, device=device, dtype=rgb_latents.dtype)
    
    # Prepare 4-condition batch for CFG
    batch_latents, batch_rgb, batch_text, batch_vlm = prepare_cfg_inference_batch(
        latents, rgb_latents, text_embeds, vlm_tokens
    )
    
    # Denoising loop
    for t in inference_scheduler.timesteps:
        # Concatenate RGB latent (InstructPix2Pix style)
        latents_input = torch.cat([batch_latents, batch_rgb], dim=1)  # (4B, 8, H, W)
        # Ensure dtype matches UNet
        latents_input = latents_input.to(dtype=unet.dtype)
        
        # Concatenate text + VLM for cross-attention
        encoder_hidden_states = torch.cat([batch_text, batch_vlm], dim=1)  # (4B, N, D)
        # Ensure dtype matches UNet
        encoder_hidden_states = encoder_hidden_states.to(dtype=unet.dtype)
        
        # Predict noise (batched 4-condition forward)
        t_batch = t.repeat(batch_size * 4).to(device)
        noise_pred_batch = unet(
            latents_input,
            t_batch,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # Apply extended CFG
        noise_pred = apply_extended_cfg(
            noise_pred_batch,
            image_guidance_scale=image_guidance_scale,
            text_guidance_scale=text_guidance_scale,
            vlm_guidance_scale=vlm_guidance_scale,
        )
        
        # Scheduler step
        latents = inference_scheduler.step(noise_pred, t, latents).prev_sample
        
        # Update batched latents for next iteration
        batch_latents = torch.cat([latents] * 4, dim=0)
    
    # Decode latents
    latents = latents / vae.config.scaling_factor
    pred_image = vae.decode(latents.to(vae.dtype)).sample
    
    return pred_image


def pil_to_tensor(pil_image: Image.Image, size: int = 256) -> torch.Tensor:
    """Convert PIL image to tensor in [-1, 1]"""
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transform(pil_image)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor in [-1, 1] to PIL image"""
    # Denormalize [-1, 1] -> [0, 1]
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    # To PIL
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first in batch
    
    to_pil = transforms.ToPILImage()
    return to_pil(tensor.cpu())


def save_validation_grid(
    grid: torch.Tensor,
    output_path: str,
    add_labels: bool = True,
):
    """
    Save validation grid with optional labels.
    
    Args:
        grid: Grid tensor (C, H, W) in [0, 1]
        output_path: Output path
        add_labels: Whether to add text labels (RGB, Ours, GT)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to PIL
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(grid.cpu())
    
    if add_labels:
        from PIL import ImageDraw, ImageFont
        
        # Add labels
        draw = ImageDraw.Draw(pil_image)
        
        # Try to use a nice font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Calculate positions (3 columns)
        width, height = pil_image.size
        col_width = width // 3
        labels = ["RGB", "Ours", "GT"]
        
        for i, label in enumerate(labels):
            x = col_width * i + 10
            y = 10
            
            # Draw text with black outline for visibility
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                draw.text((x+dx, y+dy), label, font=font, fill=(0, 0, 0))
            draw.text((x, y), label, font=font, fill=(255, 255, 255))
    
    pil_image.save(output_path)


class ValidationRunner:
    """
    Manages validation with 3-column grids: [RGB | Ours | GT]
    """
    def __init__(
        self,
        val_dataloader,
        val_dataset,
        num_samples: int = 4,
        scheduler_type: str = "ddim",
        num_inference_steps: int = 100,
        image_guidance_scale: float = 1.5,
        text_guidance_scale: float = 7.5,
        vlm_guidance_scale: float = 1.5,
        resolution: int = 256,
    ):
        self.val_dataloader = val_dataloader
        self.val_dataset = val_dataset  # Need dataset for diverse sampling
        self.num_samples = num_samples
        self.scheduler_type = scheduler_type
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
        self.text_guidance_scale = text_guidance_scale
        self.vlm_guidance_scale = vlm_guidance_scale
        self.resolution = resolution
    
    def run_validation(
        self,
        unet,
        vae,
        scheduler,
        llava_extractor,
        ella_connector,
        text_encoder,
        tokenizer,
        output_dir: str,
        global_step: int,
        device: str = "cuda",
    ):
        """
        Run validation and save 3-column comparison grids: [RGB | Ours | GT]
        """
        val_output_dir = os.path.join(output_dir, "validation", f"step_{global_step:07d}")
        os.makedirs(val_output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Running validation at step {global_step}")
        print(f"{'='*60}")
        
        # Use diverse random sampling from validation dataset
        # Get diverse samples from different sequences/datasets
        diverse_samples = self.val_dataset.get_diverse_val_samples(
            num_samples=self.num_samples, 
            seed=global_step  # Use global_step as seed for reproducibility
        )
        
        print(f"Selected diverse samples from sequences: {[s['sequence'] for s in diverse_samples]}")
        
        for idx, sample_data in enumerate(diverse_samples):
            print(f"  [{idx+1}/{self.num_samples}] Generating sample from sequence '{sample_data['sequence']}'...")
            
            # Get data from processed sample
            rgb_tensor = sample_data['rgb'].unsqueeze(0).to(device)  # Add batch dim
            rgb_pil = sample_data['rgb_pil']  # PIL image for LLaVA
            gt_tensor = sample_data['thermal'].unsqueeze(0).to(device)
            
            # Generate with our model
            with torch.no_grad():
                # Extract VLM features
                llava_prompt = "How would this RGB scene appear in long-wave thermal infrared spectrum"
                vlm_hidden_states = llava_extractor([rgb_pil], [llava_prompt])
                
                # Cast to UNet dtype (float16) and move to device
                vlm_hidden_states = vlm_hidden_states.to(device=device, dtype=unet.dtype)
                
                # ELLA timestep (use middle of denoising for extraction)
                # Cast timestep to long (required by ELLA time embedding)
                t_extract = torch.tensor([500], device=device, dtype=torch.long)
                
                # Use autocast to ensure ELLA runs in float16
                with torch.cuda.amp.autocast(enabled=True):
                    vlm_tokens = ella_connector(vlm_hidden_states, t_extract)
                
                # Ensure VLM tokens are in correct dtype
                vlm_tokens = vlm_tokens.to(dtype=unet.dtype)
                
                # Encode text
                clip_prompt = "turn this image into thermal infrared"
                text_inputs = tokenizer(
                    [clip_prompt],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                text_embeds = text_encoder(text_inputs.input_ids)[0].to(dtype=unet.dtype)
                
                # Encode RGB to latent
                rgb_latents = vae.encode(rgb_tensor.to(vae.dtype)).latent_dist.sample()
                rgb_latents = rgb_latents * vae.config.scaling_factor
                rgb_latents = rgb_latents.to(device)
                
                # Generate
                pred_tensor = generate_with_cfg(
                    unet=unet,
                    scheduler=scheduler,
                    rgb_latents=rgb_latents,
                    text_embeds=text_embeds,
                    vlm_tokens=vlm_tokens,
                    vae=vae,
                    num_inference_steps=self.num_inference_steps,
                    image_guidance_scale=self.image_guidance_scale,
                    text_guidance_scale=self.text_guidance_scale,
                    vlm_guidance_scale=self.vlm_guidance_scale,
                    scheduler_type=self.scheduler_type,
                    device=device,
                )
            
            # Create 3-column grid: [RGB | Ours | GT]
            grid = create_comparison_grid(
                rgb=rgb_tensor[0].cpu(),
                pred_ours=pred_tensor[0].cpu(),
                gt=gt_tensor[0].cpu(),
            )
            
            # Save grid with sequence info in filename
            sequence_name = sample_data['sequence'].replace('/', '_').replace(' ', '_')
            grid_path = os.path.join(val_output_dir, f"sample_{idx:03d}_{sequence_name}_grid.png")
            save_validation_grid(grid, grid_path, add_labels=True)
            
            print(f"    ✓ Saved: {grid_path}")
        
        print(f"\n✓ Validation complete!")
        print(f"  Output: {val_output_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test validation utils
    print("Testing validation utilities...")
    
    # Create dummy tensors
    rgb = torch.randn(3, 256, 256) * 0.5
    pred = torch.randn(3, 256, 256) * 0.3
    gt = torch.randn(3, 256, 256) * 0.5
    
    # Create grid (3-column)
    grid = create_comparison_grid(rgb, pred, gt)
    print(f"✓ Grid shape: {grid.shape}")
    
    # Test PIL conversion
    tensor = torch.randn(3, 256, 256) * 0.5
    pil_img = tensor_to_pil(tensor)
    print(f"✓ PIL conversion: {pil_img.size}")
    
    tensor_back = pil_to_tensor(pil_img)
    print(f"✓ Tensor conversion: {tensor_back.shape}")
    
    print("\n✓ All validation utility tests passed!")

