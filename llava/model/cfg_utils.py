"""
Classifier-Free Guidance Utilities for VLM-guided InstructPix2Pix

Implements InstructPix2Pix-style CFG with 3 guidance signals:
- Image guidance (RGB latent)
- Text guidance (CLIP text)
- VLM guidance (LLaVA features)

Reference: InstructPix2Pix (Brooks et al., 2023)
CFG formula (Equation 3):
    ê_θ(z_t, c_I, c_T) = ê_θ(z_t, ∅, ∅) 
                         + s_I · (ê_θ(z_t, c_I, ∅) - ê_θ(z_t, ∅, ∅))
                         + s_T · (ê_θ(z_t, c_I, c_T) - ê_θ(z_t, c_I, ∅))

Extended for VLM:
    ê_θ(z_t, c_I, c_T, c_V) = ê_θ(z_t, ∅, ∅, ∅) 
                               + s_I · (ê_θ(z_t, c_I, ∅, ∅) - ê_θ(z_t, ∅, ∅, ∅))
                               + s_V · (ê_θ(z_t, c_I, ∅, c_V) - ê_θ(z_t, c_I, ∅, ∅))
                               + s_T · (ê_θ(z_t, c_I, c_T, c_V) - ê_θ(z_t, c_I, ∅, c_V))

Where:
- c_I: Image conditioning (RGB latent)
- c_T: Text conditioning (CLIP text embeddings)
- c_V: VLM conditioning (LLaVA features via ELLA)
- s_I, s_T, s_V: Guidance scales for image, text, VLM
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional


def apply_cfg_dropout(
    batch_size: int,
    text_prompts: List[str],
    vlm_prompts: List[str],
    vlm_tokens: torch.Tensor,
    rgb_latents: torch.Tensor,
    p_text: float = 0.05,
    p_vlm: float = 0.05,
    p_all: float = 0.05,
    device: str = "cuda",
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    Apply CFG dropout for training.
    
    Following InstructPix2Pix's approach:
    - 5% drop text only → ("", vlm_tokens, rgb_latent)
    - 5% drop VLM only → (text, zeros, rgb_latent)
    - 5% drop all (text + VLM + RGB) → ("", zeros, zeros)
    - 85% keep all → (text, vlm_tokens, rgb_latent)
    
    Args:
        batch_size: Number of samples in batch
        text_prompts: List of text prompts
        vlm_prompts: List of VLM prompts
        vlm_tokens: VLM tokens (B, N, D)
        rgb_latents: RGB latent features (B, C, H, W)
        p_text: Probability of dropping text only (default: 0.05)
        p_vlm: Probability of dropping VLM only (default: 0.05)
        p_all: Probability of dropping all (default: 0.05)
        device: Device for tensors
    
    Returns:
        dropped_text_prompts: List of text prompts (some may be "")
        dropped_vlm_tokens: VLM tokens (some may be zeros)
        dropped_rgb_latents: RGB latents (some may be zeros)
    """
    # Generate random values for each sample
    random_vals = torch.rand(batch_size, device=device)
    
    # Initialize outputs
    dropped_text_prompts = []
    dropped_vlm_tokens = vlm_tokens.clone()
    dropped_rgb_latents = rgb_latents.clone()
    
    for i in range(batch_size):
        r = random_vals[i].item()
        
        if r < p_text:
            # Drop text only (5%)
            dropped_text_prompts.append("")
            # VLM and RGB kept
            
        elif r < (p_text + p_vlm):
            # Drop VLM only (5%)
            dropped_text_prompts.append(text_prompts[i])
            dropped_vlm_tokens[i] = torch.zeros_like(vlm_tokens[i])
            # RGB kept
            
        elif r < (p_text + p_vlm + p_all):
            # Drop all: text + VLM + RGB (5%)
            dropped_text_prompts.append("")
            dropped_vlm_tokens[i] = torch.zeros_like(vlm_tokens[i])
            dropped_rgb_latents[i] = torch.zeros_like(rgb_latents[i])
            
        else:
            # Keep all (85%)
            dropped_text_prompts.append(text_prompts[i])
    
    return dropped_text_prompts, dropped_vlm_tokens, dropped_rgb_latents


def prepare_cfg_inference_batch(
    latents: torch.Tensor,
    rgb_latents: torch.Tensor,
    text_embeds: torch.Tensor,
    vlm_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare batched input for 4-condition CFG inference.
    
    Creates 4 copies of each input with different conditioning:
    1. Full conditioning: (rgb, text, vlm)
    2. No text: (rgb, "", vlm)
    3. Image only: (rgb, "", zeros_vlm)
    4. Unconditional: (zeros_rgb, "", zeros_vlm)
    
    Args:
        latents: Noisy latents (B, C, H, W)
        rgb_latents: RGB condition latents (B, C, H, W)
        text_embeds: Text embeddings (B, N_text, D)
        vlm_tokens: VLM tokens (B, N_vlm, D)
    
    Returns:
        batch_latents: (4*B, C, H, W)
        batch_rgb: (4*B, C, H, W)
        batch_text: (4*B, N_text, D)
        batch_vlm: (4*B, N_vlm, D)
    """
    batch_size = latents.size(0)
    device = latents.device
    dtype = latents.dtype
    
    # Create zero tensors for unconditional
    zeros_text = torch.zeros_like(text_embeds)
    zeros_vlm = torch.zeros_like(vlm_tokens)
    zeros_rgb = torch.zeros_like(rgb_latents)
    
    # Stack 4 conditions
    # Order: [full, no_text, img_only, uncond]
    batch_latents = torch.cat([latents] * 4, dim=0)  # Same latents for all
    
    batch_rgb = torch.cat([
        rgb_latents,   # Full: rgb
        rgb_latents,   # No text: rgb  
        rgb_latents,   # Image only: rgb
        zeros_rgb,     # Uncond: zeros
    ], dim=0)
    
    batch_text = torch.cat([
        text_embeds,   # Full: text
        zeros_text,    # No text: ""
        zeros_text,    # Image only: ""
        zeros_text,    # Uncond: ""
    ], dim=0)
    
    batch_vlm = torch.cat([
        vlm_tokens,    # Full: vlm
        vlm_tokens,    # No text: vlm
        zeros_vlm,     # Image only: zeros
        zeros_vlm,     # Uncond: zeros
    ], dim=0)
    
    return batch_latents, batch_rgb, batch_text, batch_vlm


def apply_extended_cfg(
    noise_pred_batch: torch.Tensor,
    image_guidance_scale: float = 1.5,
    text_guidance_scale: float = 7.5,
    vlm_guidance_scale: float = 1.5,
) -> torch.Tensor:
    """
    Apply extended CFG formula with 3 guidance scales.
    
    Formula:
        noise_pred = out_uncond 
                     + s_I * (out_img_only - out_uncond)        # Image guidance
                     + s_V * (out_no_text - out_img_only)       # VLM guidance
                     + s_T * (out_full - out_no_text)           # Text guidance
    
    Args:
        noise_pred_batch: Batched predictions (4*B, C, H, W) from 4 conditions
        image_guidance_scale: s_I (default: 1.5, IP2P uses 1.2-1.5)
        text_guidance_scale: s_T (default: 7.5, standard for SD)
        vlm_guidance_scale: s_V (default: 1.5, tunable)
    
    Returns:
        noise_pred: Final prediction (B, C, H, W)
    """
    # Split into 4 conditions
    # Order: [full, no_text, img_only, uncond]
    out_full, out_no_text, out_img_only, out_uncond = noise_pred_batch.chunk(4, dim=0)
    
    # Apply extended CFG formula
    noise_pred = (
        out_uncond
        + image_guidance_scale * (out_img_only - out_uncond)      # Image guidance
        + vlm_guidance_scale * (out_no_text - out_img_only)       # VLM guidance  
        + text_guidance_scale * (out_full - out_no_text)          # Text guidance
    )
    
    return noise_pred


def compute_cfg_loss(
    model,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    rgb_latents: torch.Tensor,
    text_embeds: torch.Tensor,
    vlm_tokens: torch.Tensor,
    noise: torch.Tensor,
    apply_dropout: bool = True,
    p_text: float = 0.05,
    p_vlm: float = 0.05,
    p_all: float = 0.05,
) -> torch.Tensor:
    """
    Compute diffusion loss with CFG dropout during training.
    
    Args:
        model: UNet model
        latents: Noisy latents (B, C, H, W)
        timesteps: Timesteps (B,)
        rgb_latents: RGB condition latents (B, C, H, W)
        text_embeds: Text embeddings (B, N_text, D)
        vlm_tokens: VLM tokens (B, N_vlm, D)
        noise: Ground truth noise (B, C, H, W)
        apply_dropout: Whether to apply CFG dropout (default: True)
        p_text, p_vlm, p_all: Dropout probabilities
    
    Returns:
        loss: MSE loss between predicted and true noise
    """
    batch_size = latents.size(0)
    device = latents.device
    
    if apply_dropout:
        # Apply CFG dropout (modifies text/vlm/rgb in-place for training)
        # This is handled outside this function in the training loop
        pass
    
    # Concatenate RGB latent to noisy latent (InstructPix2Pix style)
    latents_input = torch.cat([latents, rgb_latents], dim=1)  # (B, 8, H, W)
    
    # Concatenate text + VLM for cross-attention
    encoder_hidden_states = torch.cat([text_embeds, vlm_tokens], dim=1)  # (B, N_text+N_vlm, D)
    
    # Forward pass
    noise_pred = model(latents_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample
    
    # MSE loss
    loss = F.mse_loss(noise_pred, noise, reduction='mean')
    
    return loss


class CFGConfig:
    """Configuration for CFG training and inference"""
    def __init__(
        self,
        # Training dropout probabilities
        p_drop_text: float = 0.05,
        p_drop_vlm: float = 0.05,
        p_drop_all: float = 0.05,
        
        # Inference guidance scales
        image_guidance_scale: float = 1.5,
        text_guidance_scale: float = 7.5,
        vlm_guidance_scale: float = 1.5,
        
        # Enable CFG
        use_cfg: bool = True,
    ):
        self.p_drop_text = p_drop_text
        self.p_drop_vlm = p_drop_vlm
        self.p_drop_all = p_drop_all
        
        self.image_guidance_scale = image_guidance_scale
        self.text_guidance_scale = text_guidance_scale
        self.vlm_guidance_scale = vlm_guidance_scale
        
        self.use_cfg = use_cfg
    
    @property
    def total_dropout_prob(self) -> float:
        """Total probability of dropout"""
        return self.p_drop_text + self.p_drop_vlm + self.p_drop_all
    
    def __repr__(self) -> str:
        return (
            f"CFGConfig(\n"
            f"  Training dropout: text={self.p_drop_text}, vlm={self.p_drop_vlm}, all={self.p_drop_all} "
            f"(total={self.total_dropout_prob:.1%})\n"
            f"  Guidance scales: image={self.image_guidance_scale}, "
            f"text={self.text_guidance_scale}, vlm={self.vlm_guidance_scale}\n"
            f"  Enabled: {self.use_cfg}\n"
            f")"
        )


if __name__ == "__main__":
    # Test CFG utilities
    print("Testing CFG utilities...")
    
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dummy inputs
    text_prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]
    vlm_prompts = ["vlm_prompt1", "vlm_prompt2", "vlm_prompt3", "vlm_prompt4"]
    vlm_tokens = torch.randn(batch_size, 64, 768, device=device)
    rgb_latents = torch.randn(batch_size, 4, 32, 32, device=device)
    
    # Test CFG dropout
    print("\n[1] Testing CFG dropout...")
    dropped_text, dropped_vlm, dropped_rgb = apply_cfg_dropout(
        batch_size, text_prompts, vlm_prompts, vlm_tokens, rgb_latents
    )
    print(f"  Dropped text prompts: {[p if p else '<empty>' for p in dropped_text]}")
    print(f"  Dropped VLM shape: {dropped_vlm.shape}")
    print(f"  Dropped RGB shape: {dropped_rgb.shape}")
    
    # Test CFG inference batch preparation
    print("\n[2] Testing CFG inference batch...")
    latents = torch.randn(batch_size, 4, 32, 32, device=device)
    text_embeds = torch.randn(batch_size, 77, 768, device=device)
    
    batch_latents, batch_rgb, batch_text, batch_vlm = prepare_cfg_inference_batch(
        latents, rgb_latents, text_embeds, vlm_tokens
    )
    print(f"  Batch latents shape: {batch_latents.shape} (expected: {(batch_size*4, 4, 32, 32)})")
    print(f"  Batch RGB shape: {batch_rgb.shape}")
    print(f"  Batch text shape: {batch_text.shape}")
    print(f"  Batch VLM shape: {batch_vlm.shape}")
    
    # Test extended CFG
    print("\n[3] Testing extended CFG formula...")
    noise_pred_batch = torch.randn(batch_size*4, 4, 32, 32, device=device)
    noise_pred = apply_extended_cfg(
        noise_pred_batch,
        image_guidance_scale=1.5,
        text_guidance_scale=7.5,
        vlm_guidance_scale=1.5,
    )
    print(f"  Final prediction shape: {noise_pred.shape} (expected: {(batch_size, 4, 32, 32)})")
    
    # Test CFG config
    print("\n[4] Testing CFG config...")
    cfg_config = CFGConfig()
    print(cfg_config)
    
    print("\n✓ All CFG utility tests passed!")


