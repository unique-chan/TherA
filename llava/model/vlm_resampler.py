"""
Perceiver Resampler for VLM-guided InstructPix2Pix

Adapted from IP-Adapter's resampler (which is from Open Flamingo/BLIP-2)
Converts variable-length LLaVA hidden states (B, L, 4096) to fixed-length tokens (B, num_queries, 768)
for injection into InstructPix2Pix UNet cross-attention.
"""

import math
import torch
import torch.nn as nn


def FeedForward(dim, mult=4):
    """Simple MLP with GELU activation"""
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    """Reshape for multi-head attention"""
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    """
    Perceiver-style cross-attention layer
    Queries attend to variable-length input features
    """
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): Input features (variable length)
                shape (b, n1, D) - e.g., LLaVA hidden states
            latents (torch.Tensor): Learnable query tokens (fixed length)
                shape (b, n2, D) - e.g., 16 query tokens
        Returns:
            Updated latents (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        # Q from latents, K/V from input + latents
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # Attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class VLMResampler(nn.Module):
    """
    Perceiver Resampler for LLaVA hidden states
    
    Converts variable-length LLaVA hidden states to fixed-length tokens suitable
    for InstructPix2Pix UNet cross-attention injection.
    
    Args:
        dim: Internal dimension (default: 1024)
        depth: Number of Perceiver layers (default: 4)
        dim_head: Dimension per attention head (default: 64)
        heads: Number of attention heads (default: 16)
        num_queries: Number of output tokens (default: 16)
        embedding_dim: LLaVA hidden dimension (default: 4096)
        output_dim: UNet cross-attention dimension (default: 768, matching CLIP)
        ff_mult: FFN expansion factor (default: 4)
    """
    def __init__(
        self,
        dim=1024,
        depth=4,
        dim_head=64,
        heads=16,
        num_queries=16,
        embedding_dim=4096,  # LLaVA hidden dim
        output_dim=768,      # CLIP text dim (for IP2P UNet)
        ff_mult=4,
    ):
        super().__init__()
        
        # Learnable query tokens
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
        # Project input from LLaVA dim to internal dim
        self.proj_in = nn.Linear(embedding_dim, dim)
        
        # Project output to UNet cross-attention dim
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        # Perceiver layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        """
        Args:
            x: LLaVA hidden states, shape (B, L, 4096) where L is variable
        
        Returns:
            Fixed-length tokens, shape (B, num_queries, 768)
        """
        # Expand learnable queries to batch size
        latents = self.latents.repeat(x.size(0), 1, 1)
        
        # Project input to internal dimension
        x = self.proj_in(x)
        
        # Apply Perceiver layers (cross-attention + FFN)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        
        # Project to output dimension
        latents = self.proj_out(latents)
        return self.norm_out(latents)


def test_resampler():
    """Test the resampler with dummy data"""
    print("Testing VLMResampler...")
    
    # Create resampler
    resampler = VLMResampler(
        embedding_dim=4096,  # LLaVA hidden dim
        dim=1024,            # Internal dim
        output_dim=768,      # CLIP text dim
        num_queries=16,      # Fixed output tokens
        depth=4,             # Number of layers
        heads=16,
        dim_head=64
    )
    
    # Test with variable-length inputs
    batch_size = 2
    
    # Simulate different sequence lengths (like LLaVA output)
    for seq_len in [50, 100, 200]:
        x = torch.randn(batch_size, seq_len, 4096)
        out = resampler(x)
        print(f"Input: {x.shape} -> Output: {out.shape}")
        assert out.shape == (batch_size, 16, 768), f"Expected (2, 16, 768), got {out.shape}"
    
    # Count parameters
    total_params = sum(p.numel() for p in resampler.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print("✓ VLMResampler test passed!")


if __name__ == "__main__":
    test_resampler()


