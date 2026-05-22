"""
VLM-Adapter Attention Processor for InstructPix2Pix

Adapted from IP-Adapter's attention processor to inject VLM tokens from frozen LLaVA
into InstructPix2Pix UNet cross-attention layers.

Key differences from standard attention:
- Concatenates CLIP text embeddings with VLM tokens: [clip_text, vlm_tokens]
- Separate K/V projections for VLM tokens (to_k_vlm, to_v_vlm)
- Only injects to attn2 (cross-attention) in mid_block + late up_blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA (Low-Rank Adaptation) Linear layer"""
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling


class AttnProcessor(nn.Module):
    """
    Default processor for self-attention (attn1) - no changes
    """
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
        # Ensure inputs match attention weight dtype (fp16 under AMP)
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

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class VLMAttnProcessor(nn.Module):
    """
    Attention processor for VLM-Adapter injection into cross-attention (attn2).
    
    Similar to IP-Adapter, but injects VLM tokens from frozen LLaVA instead of CLIP image features.
    
    Args:
        hidden_size: Hidden size of the attention layer (e.g., 1280)
        cross_attention_dim: Dimension of cross-attention embeddings (e.g., 768 for CLIP)
        scale: Scaling factor for VLM influence (default: 1.0)
        num_tokens: Number of VLM tokens (default: 16, matching resampler output)
    """
    def __init__(
        self, 
        hidden_size, 
        cross_attention_dim=None, 
        scale=1.0, 
        num_tokens=16,
        lora_rank=8,  # kept for API compatibility; not used in base Linear path
        lora_alpha=16.0  # kept for API compatibility; not used in base Linear path
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        # Separate K/V projections for VLM tokens using BASE Linear (stable path)
        # VLM tokens from ELLA have dimension cross_attention_dim (768 for CLIP)
        in_dim = cross_attention_dim or hidden_size
        self.to_k_vlm = nn.Linear(in_dim, hidden_size, bias=False)
        self.to_v_vlm = nn.Linear(in_dim, hidden_size, bias=False)
        
        # Initialize VLM projection layers with Xavier/Glorot initialization
        # This is better for cross-attention projections than default Kaiming
        nn.init.xavier_uniform_(self.to_k_vlm.weight)
        nn.init.xavier_uniform_(self.to_v_vlm.weight)
        
        # Gentle initialization scaling for stable starts
        # Scale down initial weights by 0.8 to prevent early instability
        with torch.no_grad():
            self.to_k_vlm.weight.mul_(0.8)
            self.to_v_vlm.weight.mul_(0.8)
        
        # Per-block learnable gate for VLM influence
        # Initialize to 1.0 (equal text/VLM influence)
        self.vlm_gate = nn.Parameter(torch.ones(1))

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        # Determine target dtype from attention weights (fp16 under AMP)
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

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Ensure hidden_states matches dtype before projection
        hidden_states = hidden_states.to(dtype=target_dtype)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states.to(dtype=target_dtype)
            # Split CLIP text embeddings and VLM tokens
            # Format: [clip_text (77 tokens), vlm_tokens (num_tokens)]
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            clip_hidden_states, vlm_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],      # CLIP text
                encoder_hidden_states[:, end_pos:, :],       # VLM tokens
            )
            
            if attn.norm_cross:
                clip_hidden_states = attn.norm_encoder_hidden_states(clip_hidden_states)

        # Ensure dtypes still match attention weights (use UNet's dtype)
        clip_hidden_states = clip_hidden_states.to(dtype=target_dtype)
        # Cast VLM inputs to the dtype expected by its projection weights to avoid F.linear dtype mismatches
        vlm_dtype = self.to_k_vlm.weight.dtype
        vlm_hidden_states = vlm_hidden_states.to(dtype=vlm_dtype)

        # Standard K/V from CLIP text
        key = attn.to_k(clip_hidden_states)
        value = attn.to_v(clip_hidden_states)

        # Separate K/V from VLM tokens (ensure dtype consistency at input & outputs)
        # Safety: enforce token feature dimension is 768 (or configured cross_attention_dim)
        expected_dim = self.cross_attention_dim or self.hidden_size
        if vlm_hidden_states.shape[-1] != expected_dim:
            raise RuntimeError(
                f"VLM token dim mismatch: expected {expected_dim}, got {vlm_hidden_states.shape[-1]}"
            )
        vlm_key = self.to_k_vlm(vlm_hidden_states)
        vlm_value = self.to_v_vlm(vlm_hidden_states)

        # Unify all projections to a single dtype for subsequent attention math
        attn_dtype = query.dtype
        key = key.to(dtype=attn_dtype)
        value = value.to(dtype=attn_dtype)
        vlm_key = vlm_key.to(dtype=attn_dtype)
        vlm_value = vlm_value.to(dtype=attn_dtype)

        # Reshape for multi-head attention
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        vlm_key = attn.head_to_batch_dim(vlm_key)
        vlm_value = attn.head_to_batch_dim(vlm_value)

        # Attention with CLIP text (standard path)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        text_hidden_states = torch.bmm(attention_probs, value)

        # Attention with VLM tokens (IP-Adapter style)
        vlm_attention_probs = attn.get_attention_scores(query, vlm_key, None)
        vlm_hidden_states = torch.bmm(vlm_attention_probs, vlm_value)

        # RMSNorm both branches before mixing to prevent norm drift
        def rms_norm(x, eps=1e-6):
            """Root Mean Square Normalization"""
            norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
            return x / norm

        # Normalize both branches
        text_hidden_states = rms_norm(text_hidden_states)
        vlm_hidden_states = rms_norm(vlm_hidden_states)

        # Combine: normalized text + learnable-gated normalized VLM
        hidden_states = text_hidden_states + self.vlm_gate * vlm_hidden_states

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual.to(dtype=hidden_states.dtype)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
    def get_gate_value(self):
        """Get current VLM gate value for logging"""
        return self.vlm_gate.item()


def init_vlm_adapter_modules(unet, target_blocks=None, scale=1.0, num_tokens=16, lora_rank=8, lora_alpha=16.0):
    """
    Initialize VLM-Adapter modules in the UNet.
    
    Injects VLMAttnProcessor into cross-attention layers (attn2) of specified blocks.
    Following IP-Adapter best practices: mid_block + late up_blocks.
    
    Args:
        unet: The InstructPix2Pix UNet model
        target_blocks: List of blocks to inject (e.g., ["mid_block", "up_blocks.1", "up_blocks.2", "up_blocks.3"])
                      If None, defaults to IP-Adapter style (mid + late up)
        scale: Scaling factor for VLM influence
        num_tokens: Number of VLM tokens from resampler
    
    Returns:
        adapter_modules: ModuleList of VLM adapter parameters (to_k_vlm, to_v_vlm)
    """
    if target_blocks is None:
        target_blocks = ["mid_block", "up_blocks.1", "up_blocks.2", "up_blocks.3"]
    
    print(f"Initializing VLM-Adapter modules in blocks: {target_blocks}")
    
    attn_procs = {}
    unet_sd = unet.state_dict()
    
    for name in unet.attn_processors.keys():
        # Check if this is cross-attention (attn2)
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        
        # Determine block type
        if name.startswith("mid_block"):
            block_name = "mid_block"
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            block_name = f"up_blocks.{block_id}"
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            block_name = f"down_blocks.{block_id}"
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            block_name = "other"
            hidden_size = None
        
        # Use VLMAttnProcessor for target cross-attention blocks
        if cross_attention_dim is not None and block_name in target_blocks:
            print(f"  ✓ Injecting VLM-Adapter to: {name} (hidden_size={hidden_size})")
            
            # Create VLM processor
            attn_procs[name] = VLMAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=scale,
                num_tokens=num_tokens,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha
            )
            
            # Initialize base K/V with original text K/V weights as a sensible start
            layer_name = name.split(".processor")[0]
            if layer_name + ".to_k.weight" in unet_sd:
                orig_k = unet_sd[layer_name + ".to_k.weight"].clone()
                orig_v = unet_sd[layer_name + ".to_v.weight"].clone()
                # If dims match, copy; otherwise keep default init
                if attn_procs[name].to_k_vlm.weight.shape == orig_k.shape:
                    attn_procs[name].to_k_vlm.weight.data.copy_(orig_k)
                if attn_procs[name].to_v_vlm.weight.shape == orig_v.shape:
                    attn_procs[name].to_v_vlm.weight.data.copy_(orig_v)
        else:
            # Use default processor for self-attention and non-target blocks
            attn_procs[name] = AttnProcessor()
    
    unet.set_attn_processor(attn_procs)
    
    # Collect VLM adapter modules for training
    adapter_modules = torch.nn.ModuleList([
        proc for proc in unet.attn_processors.values() 
        if isinstance(proc, VLMAttnProcessor)
    ])
    
    print(f"✓ Initialized {len(adapter_modules)} VLM-Adapter modules")
    
    return adapter_modules


if __name__ == "__main__":
    print("VLM Attention Processor module")
    print("This module provides attention processors for VLM-guided InstructPix2Pix")

