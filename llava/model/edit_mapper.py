"""
EditMapper: Maps LLaVA IMG token hidden states (4096) to SD text conditioning space (768).

Two implementations:
1. LightweightEditMapper: Simple MLP (recommended to start)
2. MGIEStyleEditMapper: Transformer-based with learnable queries (from MGIE paper)
"""

import torch
import torch.nn as nn
from typing import Optional


def match_dtype_to_model(mapper: nn.Module, model: nn.Module) -> nn.Module:
    """
    Helper function to match EditMapper dtype to LLaVA model dtype.
    
    Args:
        mapper: EditMapper instance
        model: LLaVA model
    
    Returns:
        mapper: EditMapper with matching dtype
    """
    model_dtype = next(model.parameters()).dtype
    mapper_dtype = next(mapper.parameters()).dtype
    
    if model_dtype != mapper_dtype:
        if model_dtype == torch.float16:
            mapper = mapper.half()
        elif model_dtype == torch.bfloat16:
            mapper = mapper.bfloat16()
        elif model_dtype == torch.float32:
            mapper = mapper.float()
        print(f"Converted EditMapper from {mapper_dtype} to {model_dtype}")
    
    return mapper


class LightweightEditMapper(nn.Module):
    """
    Lightweight EditMapper with simple MLP projection.
    
    Maps IMG token hidden states from LLaVA (4096) to SD text conditioning space (768).
    This is a stable, efficient baseline recommended for initial training.
    
    Architecture:
        LayerNorm -> Linear(4096->mid_dim) -> GELU -> Linear(mid_dim->768) -> LayerNorm
        Optional: L2 normalization + learnable scale (for CLIP stat matching)
    
    Args:
        in_dim: Input dimension (LLaVA hidden size, default: 4096)
        mid_dim: Hidden dimension (default: 1024)
        out_dim: Output dimension (SD CLIP text hidden, default: 768)
        k_tokens: Number of IMG tokens (default: 16)
        use_clip_norm: Apply L2 normalization + learnable scale to match CLIP (default: False)
    """
    
    def __init__(self, in_dim: int = 4096, mid_dim: int = 1024, 
                 out_dim: int = 768, k_tokens: int = 16,
                 use_clip_norm: bool = False):
        super().__init__()
        
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.k_tokens = k_tokens
        self.use_clip_norm = use_clip_norm
        
        # Simple MLP projection
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
        )
        
        # Output normalization
        self.out_norm = nn.LayerNorm(out_dim)
        
        # Optional: CLIP-style L2 normalization + learnable scale
        if use_clip_norm:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)  # CLIP uses ~20-30
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights with small values for stable training start.
        
        Using gain=0.02 intentionally to start with small outputs, preventing
        large initial perturbations to the frozen UNet cross-attention.
        Standard gain=1.0 can cause instability early in training.
        """
        for module in self.proj.modules():
            if isinstance(module, nn.Linear):
                # Small gain for gentle initialization
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, h_img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h_img: (B, K, in_dim) - IMG token hidden states from LLaVA
            mask: (B, K) - Optional binary mask (1=valid, 0=missing/padding)
                  Used to zero out contributions from missing IMG tokens
        
        Returns:
            z_vlm: (B, K, out_dim) - Projected features for SD cross-attention
        
        Note:
            Input dtype should match mapper dtype. If using mixed precision:
            - Convert mapper to float16: mapper.half()
            - Or convert input to float32: h_img.float()
        """
        # Project: (B, K, 4096) -> (B, K, 768)
        z = self.proj(h_img)
        z = self.out_norm(z)
        
        # Optional: CLIP-style normalization
        if self.use_clip_norm:
            # L2 normalize then scale
            z = nn.functional.normalize(z, dim=-1)  # Unit vectors
            z = z * self.scale  # Learnable scaling (CLIP uses ~20-30)
        
        # Apply mask if provided (zero out missing tokens)
        if mask is not None:
            # Expand mask: (B, K) -> (B, K, 1) for broadcasting
            mask = mask.unsqueeze(-1).to(z.dtype)
            z = z * mask
        
        return z
    
    def extra_repr(self) -> str:
        return f'in_dim={self.in_dim}, mid_dim={self.mid_dim}, out_dim={self.out_dim}, k_tokens={self.k_tokens}'


class MGIEStyleEditMapper(nn.Module):
    """
    MGIE-style EditMapper with Transformer and learnable queries.
    
    This is more expressive but also more complex. Use if the lightweight mapper
    doesn't provide enough capacity.
    
    Architecture (from MGIE paper):
        1. Project LLaVA hidden: 4096 -> 512
        2. Transformer encoder-decoder with learnable queries
        3. Project to SD space: 512 -> 768
    
    Args:
        in_dim: Input dimension (LLaVA hidden size, default: 4096)
        hid_dim: Hidden dimension for transformer (default: 512)
        out_dim: Output dimension (SD CLIP text hidden, default: 768)
        num_queries: Number of output queries (default: 77, SD max sequence length)
        num_encoder_layers: Transformer encoder layers (default: 4)
        num_decoder_layers: Transformer decoder layers (default: 4)
        nhead: Number of attention heads (default: 4)
        dim_feedforward: FFN dimension (default: 2048)
        dropout: Dropout rate (default: 0.0)
        use_positional_encoding: Add learned positional encodings (default: True)
    """
    
    def __init__(self, 
                 in_dim: int = 4096,
                 hid_dim: int = 512,
                 out_dim: int = 768,
                 num_queries: int = 77,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4,
                 nhead: int = 4,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.0,
                 use_positional_encoding: bool = True):
        super().__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_queries = num_queries
        self.use_positional_encoding = use_positional_encoding
        
        # Project LLaVA hidden to transformer dimension
        self.llm2hid = nn.Linear(in_dim, hid_dim)
        
        # Learnable queries (output sequence)
        self.query = nn.Parameter(torch.randn(1, num_queries, hid_dim))
        
        # Optional: Positional encodings for better structure
        if use_positional_encoding:
            # Positional encoding for queries (decoder input)
            self.query_pos = nn.Parameter(torch.randn(1, num_queries, hid_dim))
            # Positional encoding for encoder input (source IMG tokens)
            self.src_pos = nn.Parameter(torch.randn(1, 16, hid_dim))  # Assuming max 16 IMG tokens
        
        # Transformer encoder-decoder
        # Note: src=encoder input (IMG hiddens), tgt=decoder input (queries)
        self.mapper = nn.Transformer(
            d_model=hid_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        # Project to SD text conditioning space
        self.hid2feat = nn.Linear(hid_dim, out_dim)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable start"""
        # Initialize query and positional encodings with small values
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        
        if self.use_positional_encoding:
            nn.init.normal_(self.query_pos, mean=0.0, std=0.02)
            nn.init.normal_(self.src_pos, mean=0.0, std=0.02)
        
        # Initialize linear layers (small gain for gentle start)
        nn.init.xavier_uniform_(self.llm2hid.weight, gain=0.02)
        nn.init.zeros_(self.llm2hid.bias)
        nn.init.xavier_uniform_(self.hid2feat.weight, gain=0.02)
        nn.init.zeros_(self.hid2feat.bias)
    
    def forward(self, llm: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            llm: (B, K, in_dim) - IMG token hidden states from LLaVA
            emb: (B, K, in_dim) - Optional, IMG token embeddings (MGIE adds these)
        
        Returns:
            feat: (B, num_queries, out_dim) - Features for SD cross-attention
        
        Note:
            Transformer forward signature: transformer(src, tgt)
            - src (encoder input): IMG hiddens + positional encoding
            - tgt (decoder input): learnable queries + positional encoding
        """
        batch_size = llm.shape[0]
        k_tokens = llm.shape[1]
        
        # Add embeddings if provided (MGIE does this)
        if emb is not None:
            hid = self.llm2hid(llm + emb)  # (B, K, hid_dim)
        else:
            hid = self.llm2hid(llm)  # (B, K, hid_dim)
        
        # Add positional encoding to source (encoder input)
        if self.use_positional_encoding:
            # Trim or pad src_pos to match actual k_tokens
            src_pos = self.src_pos[:, :k_tokens, :]  # (1, K, hid_dim)
            hid = hid + src_pos  # (B, K, hid_dim)
        
        # Prepare queries (decoder input)
        queries = self.query.repeat(batch_size, 1, 1)  # (B, num_queries, hid_dim)
        
        # Add positional encoding to queries
        if self.use_positional_encoding:
            queries = queries + self.query_pos  # (B, num_queries, hid_dim)
        
        # Transformer: src=hid (encoder), tgt=queries (decoder)
        # Returns (B, num_queries, hid_dim)
        hid = self.mapper(hid, queries)
        
        # Project to SD space
        feat = self.hid2feat(hid)  # (B, num_queries, out_dim)
        
        return feat
    
    def extra_repr(self) -> str:
        return (f'in_dim={self.in_dim}, hid_dim={self.hid_dim}, out_dim={self.out_dim}, '
                f'num_queries={self.num_queries}')


# Convenience factory
def create_edit_mapper(style: str = "lightweight", **kwargs) -> nn.Module:
    """
    Factory function to create EditMapper.
    
    Args:
        style: "lightweight" or "mgie"
        **kwargs: Arguments passed to the mapper constructor
    
    Returns:
        EditMapper instance
    """
    if style == "lightweight":
        return LightweightEditMapper(**kwargs)
    elif style == "mgie":
        return MGIEStyleEditMapper(**kwargs)
    else:
        raise ValueError(f"Unknown style: {style}. Choose 'lightweight' or 'mgie'")


# For backward compatibility and convenience
EditMapper = LightweightEditMapper
