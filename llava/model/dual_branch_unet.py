"""
Dual-branch UNet wrapper for VLM-guided InstructPix2Pix.

Simpler approach: Run UNet twice and combine outputs.
This uses the diffusers library as-is, no custom processors needed!

Forward:
    out_text = UNet(latent, t, text_cond)
    out_vlm = UNet(latent, t, vlm_cond)
    out = α_text * out_text + α_vlm * out_vlm

Where α_text and α_vlm are learnable scalars.
"""

import os
import torch
import torch.nn as nn
from typing import Optional
from diffusers import UNet2DConditionModel


class DualBranchUNet(nn.Module):
    """
    Dual-branch UNet wrapper that combines text and VLM conditioning.
    
    This is a simpler alternative to modifying UNet attention processors.
    Just runs UNet twice and combines outputs with learnable weights.
    
    Args:
        unet: Pretrained UNet2DConditionModel
        init_alpha_text: Initial weight for text branch (default: 0.3)
        init_alpha_vlm: Initial weight for VLM branch (default: 0.7)
        freeze_unet: If True, freeze UNet weights (only train alphas)
                    If False, fine-tune UNet with LoRA or full weights
    """
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        init_alpha_text: float = 0.3,
        init_alpha_vlm: float = 0.7,
        freeze_unet: bool = True,
    ):
        super().__init__()
        
        self.unet = unet
        
        # Learnable mixing weights
        self.alpha_text = nn.Parameter(torch.tensor(init_alpha_text))
        self.alpha_vlm = nn.Parameter(torch.tensor(init_alpha_vlm))
        
        # Optionally freeze UNet
        if freeze_unet:
            self.unet.requires_grad_(False)
            print(f"  UNet frozen (only alphas trainable)")
        else:
            print(f"  UNet trainable (full fine-tuning)")
    
    def forward(
        self,
        latent_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states_text: torch.Tensor,
        encoder_hidden_states_vlm: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            latent_input: (B, C, H, W) - Noisy latents (+ RGB latents for IP2P)
            timestep: (B,) - Timestep
            encoder_hidden_states_text: (B, 77, 768) - Text conditioning from CLIP
            encoder_hidden_states_vlm: (B, 16, 768) - VLM conditioning from EditMapper
                                       If None, use text-only mode
        
        Returns:
            noise_pred: (B, 4, H, W) - Predicted noise
        """
        # Branch A: Text conditioning (standard IP2P)
        noise_pred_text = self.unet(
            latent_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states_text,
            **kwargs
        ).sample
        
        # Branch B: VLM conditioning (our addition)
        if encoder_hidden_states_vlm is not None:
            noise_pred_vlm = self.unet(
                latent_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states_vlm,
                **kwargs
            ).sample
            
            # Combine with learnable weights
            noise_pred = self.alpha_text * noise_pred_text + self.alpha_vlm * noise_pred_vlm
        else:
            # Text-only mode (fallback to standard IP2P)
            noise_pred = noise_pred_text
        
        return noise_pred
    
    def get_trainable_params(self):
        """Get trainable parameters for optimizer"""
        params = {
            'alphas': [self.alpha_text, self.alpha_vlm],
            'unet': list(self.unet.parameters()) if any(p.requires_grad for p in self.unet.parameters()) else []
        }
        return params
    
    def print_alpha_values(self):
        """Print current alpha values"""
        return f"α_text={self.alpha_text.item():.4f}, α_vlm={self.alpha_vlm.item():.4f}"
    
    def save_pretrained(self, save_path):
        """Save UNet and alphas"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save UNet
        self.unet.save_pretrained(os.path.join(save_path, "unet"))
        
        # Save alphas
        torch.save({
            'alpha_text': self.alpha_text,
            'alpha_vlm': self.alpha_vlm,
        }, os.path.join(save_path, "dual_branch_alphas.pt"))
        
        print(f"Saved dual-branch UNet to: {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path, unet=None):
        """Load dual-branch UNet"""
        if unet is None:
            unet = UNet2DConditionModel.from_pretrained(
                os.path.join(load_path, "unet")
            )
        
        # Create instance
        instance = cls(unet)
        
        # Load alphas
        alpha_path = os.path.join(load_path, "dual_branch_alphas.pt")
        if os.path.exists(alpha_path):
            alphas = torch.load(alpha_path)
            instance.alpha_text.data.copy_(alphas['alpha_text'])
            instance.alpha_vlm.data.copy_(alphas['alpha_vlm'])
            print(f"Loaded alphas: {instance.print_alpha_values()}")
        
        return instance


class DualBranchUNetWithLoRA(DualBranchUNet):
    """
    Dual-branch UNet with LoRA fine-tuning on cross-attention.
    
    This combines the simplicity of dual-branch forward with efficient LoRA training.
    
    Args:
        unet: Pretrained UNet
        init_alpha_text: Initial text weight
        init_alpha_vlm: Initial VLM weight
        lora_r: LoRA rank (default: 8)
        lora_alpha: LoRA scaling (default: 8)
        target_modules: Which modules to apply LoRA (default: ["to_k", "to_v", "to_q"])
    """
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        init_alpha_text: float = 0.3,
        init_alpha_vlm: float = 0.7,
        lora_r: int = 8,
        lora_alpha: int = 8,
        target_modules: list = None,
    ):
        # Don't freeze UNet yet (will add LoRA)
        super().__init__(unet, init_alpha_text, init_alpha_vlm, freeze_unet=False)
        
        if target_modules is None:
            target_modules = ["to_k", "to_v", "to_q"]
        
        # Apply LoRA to UNet
        print(f"\n  Applying LoRA to UNet cross-attention...")
        print(f"    Rank: {lora_r}, Alpha: {lora_alpha}")
        print(f"    Target modules: {target_modules}")
        
        try:
            from peft import LoraConfig, get_peft_model
            
            # Find cross-attention layers (attn2 in transformer blocks)
            target_names = []
            for name, module in self.unet.named_modules():
                if any(target in name for target in target_modules) and 'attn2' in name:
                    target_names.append(name)
            
            if target_names:
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=0.05,
                    bias="none",
                )
                
                self.unet = get_peft_model(self.unet, lora_config)
                self.unet.print_trainable_parameters()
            else:
                print(f"    ! No matching modules found, UNet will be fully trainable")
        
        except ImportError:
            print(f"    ! peft not available, UNet will be fully trainable")


def create_dual_branch_unet(
    unet: UNet2DConditionModel,
    use_lora: bool = False,
    freeze_unet: bool = True,
    **kwargs
):
    """
    Factory function to create dual-branch UNet.
    
    Args:
        unet: Pretrained UNet2DConditionModel
        use_lora: If True, use LoRA version (efficient fine-tuning)
                 If False, use frozen UNet (only train alphas)
        freeze_unet: Only used if use_lora=False
        **kwargs: Additional arguments for DualBranchUNet or DualBranchUNetWithLoRA
    
    Returns:
        DualBranchUNet instance
    """
    if use_lora:
        return DualBranchUNetWithLoRA(unet, **kwargs)
    else:
        return DualBranchUNet(unet, freeze_unet=freeze_unet, **kwargs)

