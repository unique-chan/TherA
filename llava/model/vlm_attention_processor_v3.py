"""
VLM-Adapter Attention Processor v3 for InstructPix2Pix

Port-back from soft-prompt implementation with proven stability improvements:
1. Out-of-band VLM token passing (doesn't touch encoder_hidden_states)
2. Single-softmax concat attention (no double softmax + add)
3. Learnable gate with logit-bias (pre-softmax)
4. Token dropout (train-time regularization)
5. RMSNorm on VLM only + 1/sqrt(K) scaling
6. Xavier initialization with 0.8 scaling
7. Per-timestep ELLA support
8. Decoupled cross-attention (IP-Adapter style)
9. Proper dtype handling and logging

Key difference from soft-prompt: VLM tokens passed out-of-band, not concatenated to encoder_hidden_states.
This keeps the CLIP text branch pristine while still using proven conditioning techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class AttnProcessor(nn.Module):
    """Default processor for self-attention (attn1) - no changes"""
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        target_dtype = attn.to_q.weight.dtype
        residual = hidden_states
        hidden_states = hidden_states.to(dtype=target_dtype)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class VLMAttnProcessorV3(nn.Module):
    """
    VLM-Adapter Attention Processor v3 with proven stability techniques.
    
    Port-back from soft-prompt implementation:
    - Out-of-band VLM token passing (set_vlm_tokens)
    - Single-softmax concat attention
    - Learnable gate with logit-bias
    - Token dropout (train-time)
    - RMSNorm on VLM only + 1/sqrt(K) scaling
    
    Args:
        hidden_size: Hidden size of the attention layer
        cross_attention_dim: Dimension of cross-attention embeddings (CLIP: 768)
        num_tokens: Number of VLM tokens
        gate_init: Initial gate value (0.05)
        gate_max: Maximum gate value (0.3-0.5)
        token_dropout: Dropout rate for VLM tokens during training (0.2-0.4)
    """
    def __init__(
        self, 
        inner_dim,  # heads * head_dim (output dimension)
        cross_attention_dim=768,  # input dimension
        num_tokens=16,
        gate_init=0.05,
        gate_max=0.3,
        token_dropout=0.3,
    ):
        super().__init__()
        
        self.inner_dim = inner_dim  # heads * head_dim
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.gate_max = gate_max
        self.token_dropout = token_dropout
        
        # Separate K/V projections for VLM tokens
        # Output to inner_dim (heads * head_dim), not hidden_size (query input dim)
        self.to_k_vlm = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v_vlm = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        
        # Initialize with Xavier/Glorot + 0.8 scaling for stability
        nn.init.xavier_uniform_(self.to_k_vlm.weight)
        nn.init.xavier_uniform_(self.to_v_vlm.weight)
        with torch.no_grad():
            self.to_k_vlm.weight.mul_(0.8)
            self.to_v_vlm.weight.mul_(0.8)
        
        # RMSNorm for VLM tokens only (don't norm CLIP text)
        self.vlm_norm = RMSNorm(cross_attention_dim, eps=1e-6)
        
        # 1/sqrt(K) scaling for VLM features
        self.vlm_scale = 1.0 / math.sqrt(num_tokens)
        
        # Learnable gate (logit-bias approach)
        # Initialize to achieve gate_init after sigmoid
        gate_logit_init = math.log(gate_init / (1.0 - gate_init + 1e-6))
        self.gate_logit = nn.Parameter(torch.tensor(gate_logit_init))
        
        # Out-of-band VLM token storage
        self._vlm_tokens = None
        
        # Mark as VLM processor for logging/debugging
        self.is_vlm_proc = True
    
    def set_vlm_tokens(self, tokens):
        """Set VLM tokens for this forward pass (out-of-band)"""
        self._vlm_tokens = tokens
    
    def get_gate_value(self):
        """Get current gate value for logging"""
        return (torch.sigmoid(self.gate_logit) * self.gate_max).item()
    
    def set_gate_cap(self, new_cap: float):
        """Set new gate cap (for annealing)"""
        self.gate_max = float(new_cap)
    
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        target_dtype = attn.to_q.weight.dtype
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Query from hidden states
        hidden_states = hidden_states.to(dtype=target_dtype)
        query = attn.to_q(hidden_states)

        # CLIP text embeddings (pristine, no VLM concatenation)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states.to(dtype=target_dtype)
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # Standard K/V from CLIP text (unchanged)
        key_text = attn.to_k(encoder_hidden_states)
        value_text = attn.to_v(encoder_hidden_states)

        # VLM path (out-of-band)
        vlm_tokens = self._vlm_tokens
        self._vlm_tokens = None  # Clear to avoid stale reuse
        
        if vlm_tokens is not None:
            # Token dropout during training
            if self.training and self.token_dropout > 0:
                mask = (torch.rand(vlm_tokens.shape[:2], device=vlm_tokens.device) > self.token_dropout).unsqueeze(-1)
                vlm_tokens = vlm_tokens * mask.to(vlm_tokens.dtype)
            
            # RMSNorm + 1/sqrt(K) scaling on VLM only
            vlm_tokens = vlm_tokens.to(dtype=target_dtype)
            vlm_tokens = self.vlm_norm(vlm_tokens) * self.vlm_scale
            
            # VLM K/V projections
            key_vlm = self.to_k_vlm(vlm_tokens)
            value_vlm = self.to_v_vlm(vlm_tokens)
            
            # Concat K/V: [text | vlm]
            key = torch.cat([key_text, key_vlm], dim=1)
            value = torch.cat([value_text, value_vlm], dim=1)
        else:
            key = key_text
            value = value_text

        # Handle attention mask BEFORE head-batching
        attn_mask = None
        if attention_mask is not None:
            # Future-proof: extend mask to cover VLM tokens if present
            if vlm_tokens is not None:
                # pad along the last dimension regardless of mask rank
                if attention_mask.ndim == 2:            # (B, L)
                    b, l = attention_mask.shape
                    extra = key.shape[1] - l
                    if extra > 0:
                        pad = attention_mask.new_ones(b, extra)
                        attention_mask = torch.cat([attention_mask, pad], dim=-1)
                elif attention_mask.ndim == 3:          # (B, 1, L) or (B, H, L)
                    b, h, l = attention_mask.shape
                    extra = key.shape[1] - l
                    if extra > 0:
                        pad = attention_mask.new_ones(b, h, extra)
                        attention_mask = torch.cat([attention_mask, pad], dim=-1)
                # else: leave unknown shapes alone

            attn_mask = attn.prepare_attention_mask(attention_mask, key.shape[1], batch_size)

        # Multi-head attention
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Head-batch the mask too
        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(attn.heads, dim=0)  # (B*H, 1, Lc)

        # Compute attention logits
        scale = getattr(attn, "scale", None) or (query.shape[-1] ** -0.5)
        logits = torch.bmm(query, key.transpose(-1, -2)) * scale

        # Compute text length once for reuse
        text_len = key_text.shape[1]
        
        # Apply learnable gate as logit-bias on VLM segment
        if vlm_tokens is not None:
            # Split logits: [text | vlm]
            logits_text = logits[:, :, :text_len]
            logits_vlm = logits[:, :, text_len:]
            
            # Gate as logit bias: log(gate) is added to VLM logits
            gate = torch.sigmoid(self.gate_logit) * self.gate_max
            bias = torch.log(gate.clamp_min(1e-6))
            logits_vlm = logits_vlm + bias
            
            # Recombine
            logits = torch.cat([logits_text, logits_vlm], dim=2)

        # Apply attention mask if provided
        if attn_mask is not None:
            logits = logits + attn_mask

        # Single softmax over [text | vlm]
        attention_probs = torch.softmax(logits, dim=-1)
        
        # Log attention mass split (optional, for debugging)
        if self.training and vlm_tokens is not None and torch.rand(1).item() < 0.01:  # 1% chance
            try:
                vlm_mass = attention_probs[:, :, text_len:].sum(dim=-1).mean()
                text_mass = attention_probs[:, :, :text_len].sum(dim=-1).mean()
                # Store for logging (will be logged by trainer)
                self._last_vlm_mass = vlm_mass.item()
                self._last_text_mass = text_mass.item()
            except Exception:
                # Guard against rare shape edge cases
                pass
        
        hidden_states = torch.bmm(attention_probs, value)
        
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual.to(dtype=hidden_states.dtype)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def init_vlm_adapter_modules_v3(
    unet, 
    target_blocks=None, 
    num_tokens=16, 
    gate_init=0.05, 
    gate_max=0.3,
    token_dropout=0.3,
    inject_all_cross_attn=False,
):
    """
    Initialize VLM-Adapter v3 modules in the UNet.
    
    Uses out-of-band VLM token passing and proven stability techniques
    from the soft-prompt implementation.
    
    Args:
        unet: The InstructPix2Pix UNet model
        target_blocks: List of blocks to inject (default: mid_block + up_blocks.1,2,3)
        num_tokens: Number of VLM tokens from ELLA
        gate_init: Initial gate value (0.05)
        gate_max: Maximum gate value (0.3-0.5)
        token_dropout: Token dropout rate during training (0.2-0.4)
    
    Returns:
        adapter_modules: ModuleList of VLM adapter parameters
    """
    if target_blocks is None:
        target_blocks = ["mid_block", "up_blocks.1", "up_blocks.2", "up_blocks.3"]
    
    print(f"Initializing VLM-Adapter v3 modules in blocks: {target_blocks}")
    print(f"  - Num tokens: {num_tokens}")
    print(f"  - Gate: {gate_init} → {gate_max} (learnable)")
    print(f"  - Token dropout: {token_dropout}")
    print(f"  - VLM scale: 1/sqrt({num_tokens}) = {1.0/math.sqrt(num_tokens):.4f}")
    
    # Start with existing processors (don't overwrite self-attention!)
    attn_procs = dict(unet.attn_processors)
    
    # Handle inject everywhere option
    inject_all = inject_all_cross_attn or target_blocks in (["*"], ["ALL"], ["all"])
    
    for name in list(attn_procs.keys()):  # Use list() to avoid mutating while iterating
        # Check if this is cross-attention (attn2)
        is_cross = not name.endswith("attn1.processor")
        if not is_cross:
            continue

        # Navigate to the actual attn2 module to get dims
        mod = unet
        for part in name.replace(".processor", "").split("."):
            if part.isdigit():
                mod = mod[int(part)]
            else:
                mod = getattr(mod, part)
        # mod is the Attention module (attn2)
        inner_dim = mod.to_q.out_features               # heads * head_dim  ✅
        cross_attention_dim = mod.to_k.in_features      # context_dim       ✅

        # Figure block name from the key
        parts = name.split(".")
        if parts[0] in ["up_blocks", "down_blocks"]:
            block_name = f"{parts[0]}.{parts[1]}"
        else:
            block_name = parts[0]

        # Inject if inject_all is True OR block_name is in target_blocks
        if inject_all or block_name in target_blocks:
            print(f"  ✓ Injecting VLM-Adapter v3 to: {name} (inner_dim={inner_dim}, cross_attn_dim={cross_attention_dim})")
            
            attn_procs[name] = VLMAttnProcessorV3(
                inner_dim=inner_dim,
                cross_attention_dim=cross_attention_dim,
                num_tokens=num_tokens,
                gate_init=gate_init,
                gate_max=gate_max,
                token_dropout=token_dropout,
            )
        # else: keep existing processor for non-target cross-attention
    
    unet.set_attn_processor(attn_procs)
    
    # Collect VLM adapter modules for training
    adapter_modules = torch.nn.ModuleList([
        proc for proc in unet.attn_processors.values() 
        if isinstance(proc, VLMAttnProcessorV3)
    ])
    
    print(f"✓ Initialized {len(adapter_modules)} VLM-Adapter v3 modules")
    
    return adapter_modules


if __name__ == "__main__":
    print("VLM Attention Processor v3 module")
    print("Port-back from soft-prompt with proven stability techniques")

