"""
Custom attention processors for UNet with decoupled dual-branch conditioning.

Implements:
1. Decoupled cross-attention: separate text and VLM conditioning branches
2. LoRA on q/k/v projections for efficient fine-tuning
3. Learnable gating to balance text vs VLM contributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALinear(nn.Module):
    """
    LoRA-augmented linear layer.
    
    y = Wx + (α/r) * (B @ A) @ x
    
    Where:
    - W: frozen base weights
    - A: (r, in_features) - low-rank down projection
    - B: (out_features, r) - low-rank up projection
    - r: rank
    - α: scaling factor (usually = r)
    """
    
    def __init__(self, base_layer: nn.Linear, r: int = 8, lora_alpha: int = 8, 
                 lora_dropout: float = 0.05):
        super().__init__()
        
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Optional dropout
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)  # Same as nn.Linear
        nn.init.zeros_(self.lora_B)  # Start with zero contribution
        
        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, in_features)
        
        Returns:
            y: (B, L, out_features)
        """
        # Base output (frozen)
        base_out = self.base_layer(x)
        
        # LoRA output (trainable)
        # x @ A^T -> (B, L, r)
        lora_out = self.lora_dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)  # (B, L, r)
        lora_out = F.linear(lora_out, self.lora_B)  # (B, L, out_features)
        lora_out = lora_out * self.scaling
        
        return base_out + lora_out
    
    def extra_repr(self) -> str:
        return f'r={self.r}, alpha={self.lora_alpha}, scaling={self.scaling:.4f}, dropout={self.lora_dropout}'


class DecoupledDualBranchAttnProcessor(nn.Module):
    """
    Decoupled dual-branch cross-attention processor for UNet.
    
    Computes cross-attention with two separate conditioning sources:
    - Branch A: Text conditioning (from CLIP text encoder, standard IP2P)
    - Branch B: VLM conditioning (from LLaVA EditMapper, our addition)
    
    Output = γ_text * Attn(q, k_text, v_text) + γ_vlm * Attn(q, k_vlm, v_vlm)
    
    With LoRA on q, k, v projections for efficient training.
    
    Args:
        hidden_size: UNet hidden size at this layer
        cross_attention_dim: Conditioning dimension (768 for SD1.5 CLIP)
        num_heads: Number of attention heads
        head_dim: Dimension per head (hidden_size // num_heads)
        lora_r: LoRA rank (default: 8)
        lora_alpha: LoRA scaling (default: 8)
        lora_dropout: LoRA dropout (default: 0.05)
        init_gamma_text: Initial weight for text branch (default: 0.3)
        init_gamma_vlm: Initial weight for VLM branch (default: 0.7)
    """
    
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int = 768,
        num_heads: int = 8,
        head_dim: int = 64,
        lora_r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        init_gamma_text: float = 0.3,
        init_gamma_vlm: float = 0.7,
        # LoRA placement toggles
        lora_on_query: bool = False,
        lora_on_text: bool = False,   # keep CLIP text branch frozen by default
        lora_on_vlm: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Query projection (from UNet features)
        # Shape: hidden_size -> (num_heads * head_dim)
        self.to_q = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        
        # Text branch: k/v projections with LoRA
        self.to_k_text = nn.Linear(cross_attention_dim, num_heads * head_dim, bias=False)
        self.to_v_text = nn.Linear(cross_attention_dim, num_heads * head_dim, bias=False)
        
        # VLM branch: k/v projections with LoRA
        self.to_k_vlm = nn.Linear(cross_attention_dim, num_heads * head_dim, bias=False)
        self.to_v_vlm = nn.Linear(cross_attention_dim, num_heads * head_dim, bias=False)
        
        # Wrap with LoRA selectively (trainable deltas)
        if lora_on_query:
            self.to_q = LoRALinear(self.to_q, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        else:
            # Freeze query to mimic IP-Adapter style (no trainable deltas on q)
            self.to_q.weight.requires_grad = False
            if getattr(self.to_q, 'bias', None) is not None:
                self.to_q.bias.requires_grad = False

        if lora_on_text:
            self.to_k_text = LoRALinear(self.to_k_text, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.to_v_text = LoRALinear(self.to_v_text, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        else:
            # Freeze text-branch projections entirely (respecting frozen CLIP guidance)
            self.to_k_text.weight.requires_grad = False
            self.to_v_text.weight.requires_grad = False
            if getattr(self.to_k_text, 'bias', None) is not None:
                self.to_k_text.bias.requires_grad = False
            if getattr(self.to_v_text, 'bias', None) is not None:
                self.to_v_text.bias.requires_grad = False

        if lora_on_vlm:
            self.to_k_vlm = LoRALinear(self.to_k_vlm, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.to_v_vlm = LoRALinear(self.to_v_vlm, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        else:
            # Optionally freeze VLM branch if desired
            self.to_k_vlm.weight.requires_grad = False
            self.to_v_vlm.weight.requires_grad = False
            if getattr(self.to_k_vlm, 'bias', None) is not None:
                self.to_k_vlm.bias.requires_grad = False
            if getattr(self.to_v_vlm, 'bias', None) is not None:
                self.to_v_vlm.bias.requires_grad = False
        
        # Output projection (shared)
        # NOTE: This should NOT be frozen - it needs to allow gradient flow!
        # We'll copy weights from original UNet but keep it trainable for gradients
        self.to_out = nn.Linear(num_heads * head_dim, hidden_size, bias=True)
        
        # Learnable gates for balancing branches
        self.gamma_text = nn.Parameter(torch.tensor(init_gamma_text, dtype=torch.float32))
        self.gamma_vlm = nn.Parameter(torch.tensor(init_gamma_vlm, dtype=torch.float32))
    
    def forward(
        self,
        attn,  # diffusers Attention module (unused; kept for API compatibility)
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,  # text encoder states
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L_spatial, hidden_size) - UNet features
            encoder_hidden_states: Dict {'text': ..., 'vlm': ...} OR tensor (for compatibility)
            encoder_hidden_states_text: (B, L_text, 768) - Text conditioning (if not in dict)
            encoder_hidden_states_vlm: (B, L_vlm, 768) - VLM conditioning (if not in dict)
            attention_mask: Optional attention mask
        
        Returns:
            output: (B, L_spatial, hidden_size) - Updated UNet features
        """
        # Encoder hidden states: text; VLM fetched via context
        encoder_hidden_states_text = encoder_hidden_states
        encoder_hidden_states_vlm = None
        try:
            from .ip_context import current_ip_tokens
            encoder_hidden_states_vlm = current_ip_tokens()
        except Exception:
            pass
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Query from UNet features
        q = self.to_q(hidden_states)  # (B, L_spatial, num_heads * head_dim)
        
        # Reshape for multi-head attention: (B, L, num_heads * head_dim) -> (B, num_heads, L, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (B, num_heads, L_spatial, head_dim)
        
        # Build text K/V
        k_text = self.to_k_text(encoder_hidden_states_text)
        v_text = self.to_v_text(encoder_hidden_states_text)
        k_text = k_text.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_text = v_text.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Optionally fetch VLM tokens from context if not provided
        if encoder_hidden_states_vlm is None:
            try:
                from .ip_context import current_ip_tokens
                encoder_hidden_states_vlm = current_ip_tokens()
            except Exception:
                encoder_hidden_states_vlm = None

        # Concatenate text and (optionally gated) VLM K/V, then do ONE attention/softmax
        if encoder_hidden_states_vlm is not None:
            k_vlm = self.to_k_vlm(encoder_hidden_states_vlm)
            v_vlm = self.to_v_vlm(encoder_hidden_states_vlm)
            k_vlm = k_vlm.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v_vlm = v_vlm.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # Scale gates to match dtype
            scale_t = self.gamma_text
            scale_v = self.gamma_vlm
            scale_t = scale_t.to(k_text.dtype)
            scale_v = scale_v.to(k_text.dtype)
            # Apply gating on VLM branch before concat (text kept at 1.0 or scale_t for symmetry)
            k_cat = torch.cat([k_text * scale_t, k_vlm * scale_v], dim=2)  # concat on sequence dim (after heads axis)
            v_cat = torch.cat([v_text * scale_t, v_vlm * scale_v], dim=2)
        else:
            scale_t = self.gamma_text.to(k_text.dtype)
            k_cat = k_text * scale_t
            v_cat = v_text * scale_t

        # Attention over concatenated keys/values
        attn_scores = torch.matmul(q, k_cat.transpose(-1, -2)) * self.scale  # (B, heads, Lq, Lk)
        if attention_mask is not None:
            # Expect mask broadcastable to attn_scores
            attn_scores = attn_scores + attention_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v_cat)  # (B, heads, L_spatial, head_dim)
        
        # Reshape back: (B, heads, L_spatial, head_dim) -> (B, L_spatial, heads * head_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Output projection (trainable for gradient flow; exclude from optimizer to keep frozen)
        output = self.to_out(output)
        
        return output
    
    def get_lora_parameters(self):
        """Get all LoRA parameters for optimizer"""
        params = []
        for module in [self.to_q, self.to_k_text, self.to_v_text, self.to_k_vlm, self.to_v_vlm]:
            if isinstance(module, LoRALinear):
                params.extend([module.lora_A, module.lora_B])
        return params
    
    def get_gate_parameters(self):
        """Get gate parameters for optimizer"""
        return [self.gamma_text, self.gamma_vlm]
    
    def freeze_output_projection(self):
        """
        Freeze output projection to prevent training (but allow gradient flow).
        Call this AFTER setting up optimizer if you don't want to train to_out.
        """
        # Note: We keep requires_grad=True for gradient flow, but exclude from optimizer
        pass
    
    def get_output_projection_parameters(self):
        """Get output projection parameters (if you want to train them)"""
        return [self.to_out.weight, self.to_out.bias]


def inject_decoupled_processors(
    unet,
    target_blocks: list = None,
    lora_r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    init_gamma_text: float = 0.3,
    init_gamma_vlm: float = 0.7,
    # LoRA placement toggles (default: keep CLIP text branch frozen)
    lora_on_query: bool = True,
    lora_on_text: bool = False,
    lora_on_vlm: bool = True,
):
    """
    Inject decoupled dual-branch attention processors into UNet.
    
    Args:
        unet: UNet2DConditionModel from diffusers
        target_blocks: List of block names to modify (default: ["down_blocks.2", "mid_block"])
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        init_gamma_text: Initial text branch weight
        init_gamma_vlm: Initial VLM branch weight
    
    Returns:
        processor_dict: Dict of {name: processor} for the modified processors
    """
    if target_blocks is None:
        #target_blocks = ["down_blocks.2", "mid_block"]
        target_blocks = [n for n in unet.attn_processors.keys() if "attn2" in n]
    print(f"\nInjecting decoupled processors into UNet:")
    print(f"  Target blocks: {target_blocks}")
    print(f"  LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"  Gates: gamma_text={init_gamma_text}, gamma_vlm={init_gamma_vlm}")
    print(f"  LoRA toggles -> query:{lora_on_query}, text:{lora_on_text}, vlm:{lora_on_vlm}")
    
    # Get current attention processors
    attn_procs = unet.attn_processors
    
    # Start with ALL existing processors (critical for diffusers!)
    processor_dict = dict(attn_procs)
    inject_count = 0
    
    for name, processor in attn_procs.items():
        # Check if this is a cross-attention processor (attn2) in a target block
        is_cross_attn = 'attn2' in name
        is_target_block = any(block in name for block in target_blocks)
        
        if is_cross_attn and is_target_block:
            # Get the actual attention module to extract parameters
            # Parse name to get module (e.g., "down_blocks.2.attentions.0.transformer_blocks.0.attn2")
            module = unet
            for part in name.split('.')[:-1]:  # Remove '.processor'
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            
            # Get attention parameters
            hidden_size = module.to_q.out_features
            cross_attention_dim = module.to_k.in_features
            num_heads = module.heads
            head_dim = hidden_size // num_heads
            
            # Create custom processor
            custom_proc = DecoupledDualBranchAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_gamma_text=init_gamma_text,
                init_gamma_vlm=init_gamma_vlm,
                lora_on_query=lora_on_query,
                lora_on_text=lora_on_text,
                lora_on_vlm=lora_on_vlm,
            )
            
            # Copy base weights from original module (handle both Linear and LoRALinear cases)
            with torch.no_grad():
                # to_q
                src_to_q = module.to_q
                dst_to_q = custom_proc.to_q.base_layer if isinstance(custom_proc.to_q, LoRALinear) else custom_proc.to_q
                dst_to_q.weight.copy_(src_to_q.weight)
                if getattr(src_to_q, 'bias', None) is not None and getattr(dst_to_q, 'bias', None) is not None:
                    dst_to_q.bias.copy_(src_to_q.bias)

                # text k/v
                src_to_k = module.to_k
                src_to_v = module.to_v
                dst_to_k_text = custom_proc.to_k_text.base_layer if isinstance(custom_proc.to_k_text, LoRALinear) else custom_proc.to_k_text
                dst_to_v_text = custom_proc.to_v_text.base_layer if isinstance(custom_proc.to_v_text, LoRALinear) else custom_proc.to_v_text
                dst_to_k_text.weight.copy_(src_to_k.weight)
                dst_to_v_text.weight.copy_(src_to_v.weight)
                if getattr(src_to_k, 'bias', None) is not None and getattr(dst_to_k_text, 'bias', None) is not None:
                    dst_to_k_text.bias.copy_(src_to_k.bias)
                if getattr(src_to_v, 'bias', None) is not None and getattr(dst_to_v_text, 'bias', None) is not None:
                    dst_to_v_text.bias.copy_(src_to_v.bias)

                # vlm k/v (init from text k/v for symmetry)
                dst_to_k_vlm = custom_proc.to_k_vlm.base_layer if isinstance(custom_proc.to_k_vlm, LoRALinear) else custom_proc.to_k_vlm
                dst_to_v_vlm = custom_proc.to_v_vlm.base_layer if isinstance(custom_proc.to_v_vlm, LoRALinear) else custom_proc.to_v_vlm
                dst_to_k_vlm.weight.copy_(src_to_k.weight)
                dst_to_v_vlm.weight.copy_(src_to_v.weight)
                if getattr(src_to_k, 'bias', None) is not None and getattr(dst_to_k_vlm, 'bias', None) is not None:
                    dst_to_k_vlm.bias.copy_(src_to_k.bias)
                if getattr(src_to_v, 'bias', None) is not None and getattr(dst_to_v_vlm, 'bias', None) is not None:
                    dst_to_v_vlm.bias.copy_(src_to_v.bias)

                # to_out (module.to_out is Sequential[Linear, ...])
                dst_to_out = custom_proc.to_out
                src_to_out = module.to_out[0]
                dst_to_out.weight.copy_(src_to_out.weight)
                if src_to_out.bias is not None and dst_to_out.bias is not None:
                    dst_to_out.bias.copy_(src_to_out.bias)

            # Ensure processor module matches UNet device/dtype
            proc_device = module.to_q.weight.device
            proc_dtype = module.to_q.weight.dtype
            custom_proc.to(device=proc_device, dtype=proc_dtype)
            
            processor_dict[name] = custom_proc
            inject_count += 1
            print(f"  ✓ Injected at {name}")
    
    print(f"\n  Total injected: {inject_count} processors")
    print(f"  Total processors in dict: {len(processor_dict)} (diffusers expects all)")
    
    return processor_dict


def get_trainable_unet_params(processor_dict):
    """
    Get all trainable parameters from decoupled processors.
    
    Args:
        processor_dict: Dict from inject_decoupled_processors()
    
    Returns:
        lora_params: List of LoRA parameters
        gate_params: List of gate parameters
    """
    lora_params = []
    gate_params = []
    
    for name, proc in processor_dict.items():
        if isinstance(proc, DecoupledDualBranchAttnProcessor):
            lora_params.extend(proc.get_lora_parameters())
            gate_params.extend(proc.get_gate_parameters())
    
    print(f"\nTrainable UNet parameters:")
    print(f"  LoRA parameters: {len(lora_params)}")
    print(f"  Gate parameters: {len(gate_params)}")
    
    total_lora = sum(p.numel() for p in lora_params)
    total_gate = sum(p.numel() for p in gate_params)
    print(f"  Total trainable: {total_lora + total_gate:,} parameters")
    
    return lora_params, gate_params

