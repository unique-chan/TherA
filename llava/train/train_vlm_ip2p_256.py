#!/usr/bin/env python3
"""
Training script for VLM-guided InstructPix2Pix (256x256 resolution).

Optimized for faster training with 256x256 images.

Key Features:
1. 256x256 diffusion resolution (4x faster than 512x512)
2. Choice of EditMapper: Simple MLP or MGIE-style Transformer
3. All cross-attention blocks injected by default
4. Compatible with base IP2P model (no need for fine-tuned checkpoint)

Usage:
    deepspeed --num_gpus=4 llava/train/train_vlm_ip2p_256.py \\
        --llava-base-path checkpoints/llava-1.5-7b-hf \\
        --llava-lora-path checkpoints/llava-miragehd-prior-bbox \\
        --ip2p-pretrained ../diffusers/ip2p_ema_safetensor \\
        --data-path data/llava_no_ranking_nobbox.json \\
        --image-folder ../ \\
        --output-dir ./outputs/ip_adapter_256_v1 \\
        --batch-size 4 \\
        --num-epochs 100 \\
        --use-mgie-mapper \\
        --deepspeed ./scripts/deepspeed_zero2_vlm_ip2p.json
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

# Import the base trainer
sys.path.append(str(Path(__file__).parent.parent.parent))
from llava.train.train_vlm_guided_ip2p import VLMGuidedIP2PTrainer, parse_args


class VLMGuidedIP2P256Trainer(VLMGuidedIP2PTrainer):
    """
    Trainer optimized for 256x256 resolution.
    
    Inherits from base trainer and overrides data loading
    to use 256x256 images for diffusion.
    """
    
    def setup_dataloaders(self):
        """Setup dataloaders with 256x256 resolution for diffusion"""
        from llava.data import create_dataloaders
        
        print("\n" + "="*80)
        print("SETTING UP DATALOADERS (256x256 RESOLUTION)")
        print("="*80)
        
        # Override diffusion_size to 256
        original_diffusion_size = self.args.diffusion_size if hasattr(self.args, 'diffusion_size') else 512
        self.args.diffusion_size = 256  # Force 256x256 for diffusion
        
        print(f"  Diffusion resolution: {self.args.diffusion_size}x{self.args.diffusion_size}")
        print(f"  (Original was: {original_diffusion_size}x{original_diffusion_size})")
        
        # Call parent setup with modified args
        super().setup_dataloaders()
        
        print("="*80 + "\n")


def main():
    # Parse arguments
    args = parse_args()
    
    # Override defaults for 256x256 training
    print("\n" + "="*80)
    print("VLM-GUIDED IP2P TRAINING (256x256 OPTIMIZED)")
    print("="*80)
    print(f"Resolution: 256x256 (4x faster than 512x512)")
    print(f"EditMapper: {'MGIE-style' if args.use_mgie_mapper else 'Simple MLP'}")
    print(f"Target blocks: {'All cross-attention' if args.target_blocks is None else args.target_blocks}")
    print(f"IP2P checkpoint: {args.ip2p_pretrained}")
    
    # Warn if using fine-tuned checkpoint
    if args.ip2p_checkpoint:
        print("\n⚠ WARNING: You specified --ip2p-checkpoint")
        print("  For 256x256 training, you should use the base model (--ip2p-pretrained only)")
        print("  Fine-tuned checkpoints may be trained on 512x512 resolution")
        response = input("  Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    print("="*80 + "\n")
    
    # Create trainer
    trainer = VLMGuidedIP2P256Trainer(args)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()

