"""
ELLA Time-Step Aware Perceiver Resampler
Adapted from: https://github.com/ELLA-Diffusion/ELLA

Original paper: ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment
This implementation uses time-step conditioning to make VLM features aware of the denoising timestep.
"""

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on timestep embeddings.
    Applies affine transformation (scale + shift) based on timestep.
    """
    def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None):
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> torch.Tensor:
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class SquaredReLU(nn.Module):
    """Squared ReLU activation: ReLU(x)^2"""
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))


class PerceiverAttentionBlock(nn.Module):
    """
    Perceiver attention block with timestep conditioning.
    Uses cross-attention from learnable latents to input features.
    """
    def __init__(
        self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, _ = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        """
        Args:
            x: Input features (B, N, D) - e.g., LLaVA hidden states
            latents: Learnable query latents (B, M, D)
            timestep_embedding: Timestep embedding (B, 1, D_time)
        Returns:
            latents: Updated latents (B, M, D)
        """
        normed_latents = self.ln_1(latents, timestep_embedding)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
        )
        latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        return latents


class ELLAPerceiverResampler(nn.Module):
    """
    Time-step aware Perceiver Resampler from ELLA.
    
    Converts variable-length LLaVA hidden states to fixed-length tokens
    conditioned on the current denoising timestep.
    
    Args:
        width: Hidden dimension of perceiver (default: 768, matches CLIP)
        layers: Number of perceiver attention blocks (default: 6, ELLA setting)
        heads: Number of attention heads (default: 8)
        num_latents: Number of output tokens (default: 64, ELLA setting)
        output_dim: Output dimension (default: None, keeps width)
        input_dim: Input dimension (default: None, keeps width; set to 4096 for LLaVA)
        time_embedding_dim: Timestep embedding dimension (default: None, keeps width)
    """
    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        output_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        time_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.num_latents = num_latents
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.time_embedding_dim = time_embedding_dim or width
        
        # Learnable latent queries
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
        
        # Time-aware linear for modulating latents with timestep
        self.time_aware_linear = nn.Linear(
            self.time_embedding_dim, width, bias=True
        )

        # Input projection (if input dim doesn't match width)
        if self.input_dim is not None and self.input_dim != width:
            self.proj_in = nn.Linear(input_dim, width)
        else:
            self.proj_in = None

        # Perceiver attention blocks
        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    width, heads, time_embedding_dim=self.time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        # Output projection (if output dim specified)
        if self.output_dim is not None and self.output_dim != width:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), 
                nn.LayerNorm(output_dim)
            )
        else:
            self.proj_out = None

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor):
        """
        Forward pass of time-aware perceiver resampler.
        
        Args:
            x: Input features (B, N, D_in) - e.g., LLaVA hidden states
            timestep_embedding: Timestep embedding (B, 1, D_time) - from UNet's timestep encoder
            
        Returns:
            latents: Resampled tokens (B, num_latents, D_out)
        """
        batch_size = x.size(0)
        
        # Project input if needed
        if self.proj_in is not None:
            x = self.proj_in(x)
        
        # Initialize learnable latents and modulate with timestep
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        latents = learnable_latents + self.time_aware_linear(
            torch.nn.functional.silu(timestep_embedding)
        )
        
        # Apply perceiver attention blocks with timestep conditioning
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)

        # Project output if needed
        if self.proj_out is not None:
            latents = self.proj_out(latents)

        return latents


class ELLAConnector(nn.Module):
    """
    Full ELLA connector: Timestep encoder + Perceiver Resampler.
    
    This is the complete module that:
    1. Encodes timestep to embedding
    2. Resamples VLM features conditioned on timestep
    
    Args:
        time_channel: Timestep encoding channels (default: 320, UNet time dim)
        time_embed_dim: Timestep embedding dimension (default: 768, CLIP dim)
        act_fn: Activation function (default: "silu")
        out_dim: Output dimension for timestep embedding (default: None)
        width: Perceiver hidden dimension (default: 768)
        layers: Number of perceiver layers (default: 6)
        heads: Number of attention heads (default: 8)
        num_latents: Number of output tokens (default: 64)
        input_dim: VLM feature dimension (default: 4096 for LLaVA)
    """
    def __init__(
        self,
        time_channel: int = 320,
        time_embed_dim: int = 768,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        input_dim: int = 4096,
    ):
        super().__init__()

        # Timestep encoding (sinusoidal positional encoding)
        self.position = Timesteps(
            time_channel, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        
        # Timestep embedding MLP
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,
        )

        # Perceiver resampler with timestep conditioning
        self.connector = ELLAPerceiverResampler(
            width=width,
            layers=layers,
            heads=heads,
            num_latents=num_latents,
            input_dim=input_dim,
            output_dim=None,  # Keep width (768) as output dimension
            time_embedding_dim=time_embed_dim,
        )

    def forward(self, vlm_hidden_states: torch.Tensor, timesteps: torch.Tensor):
        """
        Forward pass of ELLA connector.
        
        Args:
            vlm_hidden_states: VLM features (B, N, D_vlm) - e.g., LLaVA outputs
            timesteps: Timesteps (B,) - current denoising timestep
            
        Returns:
            encoder_hidden_states: Resampled tokens (B, num_latents, D_out)
        """
        device = vlm_hidden_states.device
        dtype = vlm_hidden_states.dtype

        # Encode timestep to sinusoidal features
        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        ori_time_feature = (
            ori_time_feature.unsqueeze(dim=1)
            if ori_time_feature.ndim == 2
            else ori_time_feature
        )
        ori_time_feature = ori_time_feature.expand(len(vlm_hidden_states), -1, -1)
        
        # Embed timestep
        time_embedding = self.time_embedding(ori_time_feature)

        # Resample VLM features with timestep conditioning
        encoder_hidden_states = self.connector(
            vlm_hidden_states, timestep_embedding=time_embedding
        )

        return encoder_hidden_states


if __name__ == "__main__":
    # Test the module
    print("Testing ELLA Perceiver Resampler...")
    
    batch_size = 2
    seq_len = 10
    vlm_dim = 4096  # LLaVA
    clip_dim = 768
    num_latents = 64
    
    # Create model
    ella = ELLAConnector(
        time_channel=320,
        time_embed_dim=768,
        width=768,
        layers=6,
        heads=8,
        num_latents=num_latents,
        input_dim=vlm_dim,
    )
    
    # Test input
    vlm_hidden_states = torch.randn(batch_size, seq_len, vlm_dim)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    # Forward pass
    output = ella(vlm_hidden_states, timesteps)
    
    print(f"✓ Input shape: {vlm_hidden_states.shape}")
    print(f"✓ Timesteps shape: {timesteps.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected: ({batch_size}, {num_latents}, {clip_dim})")
    
    assert output.shape == (batch_size, num_latents, clip_dim), "Shape mismatch!"
    print("\n✓ All tests passed!")

