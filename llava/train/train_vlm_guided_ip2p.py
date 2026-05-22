#!/usr/bin/env python3
"""
Training script for VLM-guided InstructPix2Pix.

Combines:
1. LLaVA language model (for structured thermal understanding)
2. EditMapper (projects IMG tokens to diffusion space)
3. InstructPix2Pix UNet (with decoupled cross-attention)

Loss = L_CE + λ_diffusion * L_diffusion
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

# Diffusion components
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, AutoTokenizer

# LLaVA components
sys.path.append(str(Path(__file__).parent.parent.parent))
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.model import (
    EditMapper,
    MGIEStyleEditMapper,
    match_dtype_to_model,
    add_img_tokens_to_tokenizer,
    resize_token_embeddings_and_init,
    setup_llava_trainable_params_mgie_style,
    save_tokenizer_with_img_tokens,
    verify_img_token_setup,
    get_img_token_ids,
    inject_decoupled_processors,
    get_trainable_unet_params,
)
from llava.model.dual_branch_unet import create_dual_branch_unet
from llava.data import create_dataloaders


class VLMGuidedIP2PTrainer:
    """
    Trainer for VLM-guided InstructPix2Pix.
    
    Trains three components jointly:
    1. LLaVA: embed_tokens + lm_head (emit IMG tokens)
    2. EditMapper: projects IMG hiddens to diffusion space
    3. UNet: LoRA on cross-attention + learnable gates
    """
    
    def __init__(self, args):
        self.args = args
        
        # Setup device (DeepSpeed will handle actual placement)
        if args.deepspeed:
            import os
            deepspeed.init_distributed()
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
            self.is_main_process = (int(os.environ.get("RANK", "0")) == 0)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.is_main_process = True
        
        # Setup
        self.setup_models()
        self.setup_dataloaders()
        
        # Setup optimizer (only if not using DeepSpeed - DeepSpeed creates from config)
        if not args.deepspeed:
            self.setup_optimizer()
        
        # Save reference to processors BEFORE DeepSpeed wrapping (DeepSpeed changes attribute access)
        if args.use_custom_processors and hasattr(self.unet, 'attn_processors'):
            self.saved_processors = self.unet.attn_processors
            print(f"\n✓ Saved {len(self.saved_processors)} processors before DeepSpeed wrapping")
            # Check if they have gamma parameters
            sample_proc = list(self.saved_processors.values())[0] if self.saved_processors else None
            if sample_proc:
                print(f"  Sample processor type: {type(sample_proc).__name__}")
                print(f"  Has gamma_text: {hasattr(sample_proc, 'gamma_text')}")
                print(f"  Has gamma_vlm: {hasattr(sample_proc, 'gamma_vlm')}")
                if hasattr(sample_proc, 'gamma_text'):
                    print(f"  gamma_text init value: {sample_proc.gamma_text.item():.3f}, requires_grad: {sample_proc.gamma_text.requires_grad}")
                if hasattr(sample_proc, 'gamma_vlm'):
                    print(f"  gamma_vlm init value: {sample_proc.gamma_vlm.item():.3f}, requires_grad: {sample_proc.gamma_vlm.requires_grad}")
        else:
            self.saved_processors = None
            print(f"\n⚠ No processors to save (use_custom_processors={args.use_custom_processors}, has_attn_processors={hasattr(self.unet, 'attn_processors')})")
        
        # Initialize DeepSpeed engine (this handles device placement and creates optimizer)
        if args.deepspeed:
            self.init_deepspeed()
        
        # Tracking
        self.global_step = 0  # Tracks optimizer updates (not dataloader iterations)
        self.dataloader_step = 0  # Tracks dataloader iterations
        self.epoch = 0
        
        # Mixed precision
        self.scaler = GradScaler() if args.use_amp and not args.deepspeed else None
        
        # Load checkpoint if resuming
        if args.resume_from_checkpoint:
            self.load_checkpoint(args.resume_from_checkpoint)
    
    def setup_models(self):
        """Initialize all models"""
        print("\n" + "="*80)
        print("SETTING UP MODELS")
        print("="*80)
        
        # 1. Load LLaVA
        print("\n1. Loading LLaVA...")
        
        # When using DeepSpeed, avoid Accelerate's device_map to prevent hooks
        # Pass device_map explicitly in kwargs to override the default
        if self.args.llava_lora_path:
            # Load LoRA model
            from peft import PeftModel
            
            # For DeepSpeed: use device="cuda" and override device_map to None in kwargs
            # For non-DeepSpeed: use normal device loading
            if self.args.deepspeed:
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    self.args.llava_base_path, None, 
                    get_model_name_from_path(self.args.llava_base_path),
                    load_8bit=False, load_4bit=False, device="cuda",
                    device_map=None  # Override in kwargs to avoid Accelerate
                )
            else:
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    self.args.llava_base_path, None, 
                    get_model_name_from_path(self.args.llava_base_path),
                    load_8bit=False, load_4bit=False, device=self.device
                )
            
            model = PeftModel.from_pretrained(model, self.args.llava_lora_path)
            
            # Load non-LoRA trainables
            non_lora_path = os.path.join(self.args.llava_lora_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_path):
                non_lora_weights = torch.load(non_lora_path, map_location='cpu')
                non_lora_weights = {k: v.to(torch.float16) for k, v in non_lora_weights.items()}
                model.load_state_dict(non_lora_weights, strict=False)
        else:
            # Load merged model
            model_name = get_model_name_from_path(self.args.llava_path)
            if self.args.deepspeed:
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    self.args.llava_path, None, model_name,
                    load_8bit=False, load_4bit=False, device="cuda",
                    device_map=None  # Override in kwargs to avoid Accelerate
                )
            else:
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    self.args.llava_path, None, model_name,
                    load_8bit=False, load_4bit=False, device=self.device
                )
        
        self.llava_model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        print(f"  ✓ LLaVA loaded: {model.config.hidden_size}D, {len(tokenizer)} vocab")
        
        # Print device info for debugging
        if self.args.deepspeed:
            print(f"  Loaded without device_map (no Accelerate hooks) for DeepSpeed compatibility")
        
        # Move model to device only if NOT using DeepSpeed
        # (DeepSpeed will handle device placement in init_deepspeed())
        if not self.args.deepspeed:
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            self.llava_model.to(self.device)
            
            base = self.llava_model.get_model()
            
            # mm_projector to device (if present)
            mp = getattr(base, "mm_projector", None)
            if mp is not None and hasattr(mp, "to"):
                mp.to(self.device)
            
            # vision tower to device (if present)
            vt = getattr(base, "vision_tower", None)
            if vt is not None and hasattr(vt, "to"):
                vt.to(self.device)

                
        
        # 2. Add IMG tokens
        print("\n2. Adding IMG tokens to LLaVA...")
        add_img_tokens_to_tokenizer(self.tokenizer)
        resize_token_embeddings_and_init(self.llava_model, self.tokenizer, init_method="mean")
        self.img_token_ids = get_img_token_ids(self.tokenizer)
        
        # 3. Setup LLaVA trainable params
        print("\n3. Setting up LLaVA trainable parameters (MGIE-style)...")
        setup_llava_trainable_params_mgie_style(self.llava_model)
        
        # 4. Verify setup
        print("\n4. Verifying LLaVA setup...")
        success = verify_img_token_setup(self.llava_model, self.tokenizer)
        if not success:
            raise ValueError("LLaVA setup verification failed!")
        
        # 5. Save tokenizer
        print("\n5. Saving tokenizer with IMG tokens...")
        save_tokenizer_with_img_tokens(self.tokenizer, self.args.output_dir)
        
        # 6. Create EditMapper
        print("\n6. Creating EditMapper...")
        if self.args.use_mgie_mapper:
            print(f"  Using MGIE-style Transformer mapper")
            self.edit_mapper = MGIEStyleEditMapper(
                in_dim=4096,
                hid_dim=512,
                out_dim=768,
                num_queries=77,  # SD max sequence length
                num_encoder_layers=4,
                num_decoder_layers=4,
                nhead=4,
                dim_feedforward=2048,
                dropout=0.0,
                use_positional_encoding=True
            )
        else:
            print(f"  Using lightweight MLP mapper")
            self.edit_mapper = EditMapper(
                in_dim=4096,
                mid_dim=self.args.mapper_mid_dim,
                out_dim=768,
                k_tokens=16
            )
        # Move to device only if not using DeepSpeed
        if not self.args.deepspeed:
            self.edit_mapper = self.edit_mapper.to(self.device)
        self.edit_mapper = match_dtype_to_model(self.edit_mapper, self.llava_model)
        
        print(f"  ✓ EditMapper created: {sum(p.numel() for p in self.edit_mapper.parameters()):,} params")
        
        # 7. Load diffusion components
        print("\n7. Loading InstructPix2Pix components...")
        
        # VAE (frozen) - load with proper dtype
        self.vae = AutoencoderKL.from_pretrained(
            self.args.ip2p_pretrained, subfolder="vae",
            torch_dtype=torch.float16
        )
        if not self.args.deepspeed:
            self.vae = self.vae.to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        # UNet - load with proper dtype
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.ip2p_pretrained, subfolder="unet",
            torch_dtype=torch.float16
        )
        if not self.args.deepspeed:
            self.unet = self.unet.to(self.device)
        
        # Load fine-tuned weights if provided
        if self.args.ip2p_checkpoint:
            print(f"  Loading fine-tuned weights from: {self.args.ip2p_checkpoint}")
            checkpoint = torch.load(self.args.ip2p_checkpoint, map_location=self.device)
            
            if 'ema' in checkpoint and checkpoint['ema'] is not None:
                print(f"  Using EMA weights")
                ema_state = checkpoint['ema']
                if 'shadow_params' in ema_state:
                    unet_state = {}
                    for i, (name, param) in enumerate(self.unet.named_parameters()):
                        if i < len(ema_state['shadow_params']):
                            unet_state[name] = ema_state['shadow_params'][i]
                    self.unet.load_state_dict(unet_state)
                else:
                    self.unet.load_state_dict(checkpoint['unet'])
            else:
                self.unet.load_state_dict(checkpoint['unet'])
            
            print(f"  ⚠ WARNING: UNet state loaded from checkpoint - processors will be set to defaults")
            print(f"  Will re-inject custom processors after loading...")
        
        # Choose UNet dual-branch approach
        # IMPORTANT: Must happen AFTER checkpoint loading to avoid processors being reset
        print(f"\n8. Setting up dual-branch UNet...")
        print(f"  Mode: {'custom_processors' if self.args.use_custom_processors else 'dual_forward'}")
        
        if self.args.use_custom_processors:
            # APPROACH 1: Custom attention processors (decoupled cross-attention)
            print(f"  Using custom attention processors (DecoupledDualBranchAttnProcessor)")
            
            # Determine LoRA placement based on args
            lora_on_text = self.args.lora_on_text_backbone
            backbone_lora_r = self.args.backbone_lora_r if lora_on_text else self.args.lora_r
            
            if lora_on_text:
                print(f"  ⚠ Adding small backbone LoRA (rank {backbone_lora_r}) on text K/V for better adaptation")
            
            self.processor_dict = inject_decoupled_processors(
                self.unet,
                target_blocks=self.args.target_blocks,
                lora_r=backbone_lora_r if lora_on_text else self.args.lora_r,
                lora_alpha=backbone_lora_r if lora_on_text else self.args.lora_alpha,  # Match alpha to rank for backbone
                lora_dropout=self.args.lora_dropout,
                init_gamma_text=self.args.init_alpha_text,
                init_gamma_vlm=self.args.init_alpha_vlm,
                lora_on_query=False,
                lora_on_text=lora_on_text,   # Enable if backbone LoRA requested
                lora_on_vlm=True,
            )
            self.unet.set_attn_processor(self.processor_dict)
            unet_lora_params, unet_gate_params = [], []
            for proc in self.unet.attn_processors.values():
                if hasattr(proc, "get_lora_parameters"):
                    unet_lora_params += list(proc.get_lora_parameters())
                if hasattr(proc, "get_gate_parameters"):
                    unet_gate_params += list(proc.get_gate_parameters())

            assert unet_lora_params or unet_gate_params, "No trainable UNet params found — check injection."

            print(f"  ✓ Custom processors injected")
            
            # Optionally train full UNet (in addition to LoRA and gates)
            if self.args.train_full_unet:
                print(f"\n  ⚠ TRAINING FULL UNET (not just LoRA/gates)")
                print(f"    This requires significantly more VRAM!")
                self.unet.requires_grad_(True)
                unet_full_params = list(self.unet.parameters())
                print(f"    Full UNet parameters: {sum(p.numel() for p in unet_full_params):,}")
        else:
            # APPROACH 2: Simplified dual forward (run UNet twice)
            print(f"  Using simplified dual-forward wrapper")
            self.unet = create_dual_branch_unet(
                self.unet,
                use_lora=self.args.use_unet_lora,
                freeze_unet=not self.args.use_unet_lora,
                init_alpha_text=self.args.init_alpha_text,
                init_alpha_vlm=self.args.init_alpha_vlm,
                lora_r=self.args.lora_r if self.args.use_unet_lora else None,
                lora_alpha=self.args.lora_alpha if self.args.use_unet_lora else None,
            )
            self.processor_dict = None
            print(f"  ✓ Dual-branch wrapper created")
            print(f"    α_text={self.args.init_alpha_text}, α_vlm={self.args.init_alpha_vlm}")
        
        # Text encoder (frozen) - load with proper dtype
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args.ip2p_pretrained, subfolder="text_encoder",
            torch_dtype=torch.float16
        )
        if not self.args.deepspeed:
            self.text_encoder = self.text_encoder.to(self.device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
        # Text tokenizer
        self.text_tokenizer = CLIPTokenizer.from_pretrained(
            self.args.ip2p_pretrained, subfolder="tokenizer"
        )
        
        # Diffusion noise scheduler (keep separate from LR scheduler)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.args.ip2p_pretrained, subfolder="scheduler"
        )
        
        print(f"  ✓ Diffusion components loaded")
        
        # Print trainable parameter summary
        self.print_trainable_summary()
    
    def setup_dataloaders(self):
        """Create train/val dataloaders"""
        print("\n" + "="*80)
        print("SETTING UP DATALOADERS")
        print("="*80)
        
        self.train_loader, self.val_loader = create_dataloaders(
            data_path=self.args.data_path,
            image_folder=self.args.image_folder,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            img_token_ids=self.img_token_ids,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            image_aspect_ratio=self.args.image_aspect_ratio,
            diffusion_size=self.args.diffusion_size,
            filter_datasets=self.args.filter_datasets,
            validation_datasets=self.args.validation_datasets,
        )
        
        print(f"\nDataloaders created:")
        print(f"  Training batches: {len(self.train_loader)}")
        print(f"  Validation batches: {len(self.val_loader) if self.val_loader else 0}")
    
    def setup_optimizer(self):
        """Setup optimizer with separate parameter groups"""
        print("\n" + "="*80)
        print("SETTING UP OPTIMIZER")
        print("="*80)
        
        # Create parameter groups
        param_groups = []
        
        if self.args.use_custom_processors:
            # APPROACH 1: Custom processors (LoRA + gates)
            unet_lora_params, unet_gate_params = [], []
            for proc in self.unet.attn_processors.values():
                if hasattr(proc, "get_lora_parameters"):
                    unet_lora_params += list(proc.get_lora_parameters())
                if hasattr(proc, "get_gate_parameters"):
                    unet_gate_params += list(proc.get_gate_parameters())

            assert unet_lora_params or unet_gate_params, "No trainable UNet params found — check injection."

            param_groups.extend([
                # UNet LoRA
                {
                    "params": unet_lora_params,
                    "lr": self.args.lr_unet,
                    "weight_decay": self.args.weight_decay
                },
                # UNet gates
                {
                    "params": unet_gate_params,
                    "lr": max(self.args.lr_unet * 0.25, 1e-5),
                    "weight_decay": 0.0  # No decay on gates
                },
            ])
        else:
            # APPROACH 2: Dual-forward wrapper (alphas + optional LoRA)
            unet_trainable = self.unet.get_trainable_params()
            
            param_groups.append({
                "params": unet_trainable['alphas'],
                "lr": self.args.lr_unet,
                "weight_decay": 0.0  # No decay on scalars
            })
            
            # Add UNet params if training with LoRA
            if unet_trainable['unet']:
                param_groups.insert(1, {
                    "params": unet_trainable['unet'],
                    "lr": self.args.lr_unet,
                    "weight_decay": self.args.weight_decay
                })
        
        # Common params (same for both approaches)
        param_groups.extend([
            # EditMapper
            {
                "params": self.edit_mapper.parameters(),
                "lr": self.args.lr_mapper,
                "weight_decay": self.args.weight_decay
            },
            
            # LLaVA embed_tokens (full matrix, low LR, no decay)
            {
                "params": self.llava_model.get_model().embed_tokens.parameters(),
                "lr": self.args.lr_llava,
                "weight_decay": 0.0
            },
            
            # LLaVA lm_head (full matrix, low LR, no decay)
            {
                "params": self.llava_model.lm_head.parameters(),
                "lr": self.args.lr_llava,
                "weight_decay": 0.0
            },
        ])
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Scheduler with warmup
        from transformers import get_cosine_schedule_with_warmup
        
        total_steps = len(self.train_loader) * self.args.num_epochs
        warmup_steps = self.args.warmup_steps
        
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"\nOptimizer setup:")
        print(f"  Parameter groups: {len(param_groups)}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps} ({100*warmup_steps/total_steps:.1f}%)")
        print(f"  LR schedule: Cosine (min={self.args.lr_min})")
    
    def init_deepspeed(self):
        """Initialize DeepSpeed engine - this handles device placement"""
        print("\n" + "="*80)
        print("INITIALIZING DEEPSPEED")
        print("="*80)
        
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed is not available. Install with: pip install deepspeed")
        
        # Load and potentially modify DeepSpeed config
        import json
        with open(self.args.deepspeed, 'r') as f:
            ds_config = json.load(f)
        
        # Override gradient accumulation steps if provided via argument
        if self.args.gradient_accumulation_steps is not None:
            ds_config['gradient_accumulation_steps'] = self.args.gradient_accumulation_steps
            print(f"  ✓ Overriding gradient_accumulation_steps: {self.args.gradient_accumulation_steps}")
        
        # Create a simple wrapper module that contains all trainable models
        # This allows DeepSpeed to properly handle device placement
        class TrainableWrapper(nn.Module):
            def __init__(self, llava, mapper, unet, vae, text_encoder):
                super().__init__()
                self.llava_model = llava
                self.edit_mapper = mapper
                self.unet = unet
                self.vae = vae
                self.text_encoder = text_encoder
        
        wrapper = TrainableWrapper(
            self.llava_model, 
            self.edit_mapper, 
            self.unet,
            self.vae,
            self.text_encoder
        )
        
        # Collect all trainable parameters (including custom processors if present)
        # Custom processors are in a dict and might not be included by wrapper.parameters()
        trainable_params = []
        
        # Add all wrapper parameters
        trainable_params.extend(wrapper.parameters())
        
        # Explicitly add custom processor parameters if using them
        if self.args.use_custom_processors and hasattr(self.unet, 'attn_processors'):
            for proc in self.unet.attn_processors.values():
                if hasattr(proc, 'parameters'):
                    trainable_params.extend(proc.parameters())
        
        # Add full UNet parameters if training full UNet
        if self.args.train_full_unet:
            print("  Including full UNet parameters...")
            trainable_params.extend(self.unet.parameters())
        
        # Remove duplicates (some params might be counted twice)
        seen_params = set()
        unique_params = []
        for p in trainable_params:
            if id(p) not in seen_params:
                seen_params.add(id(p))
                unique_params.append(p)
        
        print(f"  Total parameters for DeepSpeed optimizer: {len(unique_params)}")
        import sys
        sys.stdout.flush()  # Force flush to ensure print appears
        
        # Initialize DeepSpeed without passing optimizer/scheduler
        # DeepSpeed will create them from the config
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=wrapper,
            model_parameters=unique_params,  # Use explicitly collected params
            config=ds_config,  # Pass modified config dict
        )
        
        print(f"  ✓ DeepSpeed initialization complete")
        sys.stdout.flush()
        
        # Update references
        self.model_engine = model_engine
        self.llava_model = model_engine.module.llava_model
        self.edit_mapper = model_engine.module.edit_mapper
        self.unet = model_engine.module.unet
        self.vae = model_engine.module.vae
        self.text_encoder = model_engine.module.text_encoder
        self.optimizer = optimizer  # Use DeepSpeed's optimizer
        self.lr_scheduler = lr_scheduler  # Use DeepSpeed's scheduler
        
        print(f"  ✓ DeepSpeed engine initialized")
        print(f"    Local rank: {self.device}")
        print(f"    Zero optimization stage: {ds_config.get('zero_optimization', {}).get('stage', 'N/A')}")
        print(f"    Micro batch size per GPU: {ds_config.get('train_micro_batch_size_per_gpu', 'N/A')}")
        print(f"    Gradient accumulation steps: {ds_config.get('gradient_accumulation_steps', 1)}")
        print(f"    Optimizer: {ds_config.get('optimizer', {}).get('type', 'N/A')}")
    
    def print_trainable_summary(self):
        """Print summary of trainable parameters"""
        print("\n" + "="*80)
        print("TRAINABLE PARAMETERS SUMMARY")
        print("="*80)
        
        # LLaVA
        llava_trainable = sum(
            p.numel() for p in self.llava_model.parameters() if p.requires_grad
        )
        llava_total = sum(p.numel() for p in self.llava_model.parameters())
        
        print(f"\nLLaVA:")
        print(f"  Trainable: {llava_trainable:,} / {llava_total:,} ({100*llava_trainable/llava_total:.2f}%)")
        
        # EditMapper
        mapper_params = sum(p.numel() for p in self.edit_mapper.parameters())
        print(f"\nEditMapper:")
        print(f"  Total: {mapper_params:,} (all trainable)")
        
        # UNet
        if self.args.use_custom_processors:
            # Custom processors approach (collect from live processors)
            unet_lora_params, unet_gate_params = [], []
            for proc in self.unet.attn_processors.values():
                if hasattr(proc, "get_lora_parameters"):
                    unet_lora_params += list(proc.get_lora_parameters())
                if hasattr(proc, "get_gate_parameters"):
                    unet_gate_params += list(proc.get_gate_parameters())
            unet_trainable = sum(p.numel() for p in unet_lora_params + unet_gate_params)
            unet_total = sum(p.numel() for p in self.unet.parameters())
            
            print(f"\nUNet (Custom Processors):")
            print(f"  Trainable: {unet_trainable:,} / {unet_total:,} ({100*unet_trainable/unet_total:.2f}%)")
            print(f"    LoRA: {sum(p.numel() for p in unet_lora_params):,}")
            print(f"    Gates: {sum(p.numel() for p in unet_gate_params):,}")
        else:
            # Dual-forward wrapper approach
            unet_trainable_dict = self.unet.get_trainable_params()
            unet_alpha_params = sum(p.numel() for p in unet_trainable_dict['alphas'])
            unet_other_params = sum(p.numel() for p in unet_trainable_dict['unet'])
            unet_trainable = unet_alpha_params + unet_other_params
            
            # Get total UNet params (from wrapped unet.unet)
            unet_total = sum(p.numel() for p in self.unet.unet.parameters())
            
            print(f"\nUNet (Dual-Forward Wrapper):")
            print(f"  Trainable: {unet_trainable:,} / {unet_total:,} ({100*unet_trainable/unet_total:.2f}%)")
            print(f"    Alphas: {unet_alpha_params} (2 scalars)")
            if unet_other_params > 0:
                print(f"    LoRA/Full: {unet_other_params:,}")
            else:
                print(f"    UNet frozen (only alphas train)")
        
        # Total
        total_trainable = llava_trainable + mapper_params + unet_trainable
        total_params = llava_total + mapper_params + unet_total
        
        print(f"\nGrand Total:")
        print(f"  Trainable: {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.2f}%)")
    
    def encode_text_prompt(self, prompts):
        """Encode text prompts with CLIP"""
        text_inputs = self.text_tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_embeds = self.text_encoder(**text_inputs).last_hidden_state
        
        return text_embeds
    
    def train_step(self, batch):
        """Single training step"""
        # Zero gradients (only if not using DeepSpeed - DeepSpeed handles this)
        if not self.args.deepspeed:
            self.optimizer.zero_grad()
        
        # Move batch to device (use non_blocking for efficiency)
        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Context for mixed precision (only if not using DeepSpeed - DeepSpeed handles FP16)
        amp_context = autocast() if (self.args.use_amp and not self.args.deepspeed) else torch.enable_grad()
        
        with amp_context:
            # 1. LLaVA forward (with IMG hidden extraction)
            llava_outputs = self.llava_model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                images=batch["images"],
                return_dict=True,
                return_img_hiddens=True,
                img_token_ids=self.img_token_ids
            )
            
            loss_ce = llava_outputs['loss']
            img_hiddens = llava_outputs['img_hiddens']  # (B, 16, 4096)
            
            # 2. EditMapper: project to diffusion space
            z_vlm = self.edit_mapper(img_hiddens)  # (B, 16, 768)
            
            # 3. Encode text prompt
            text_prompts = ["Turn this image into thermal infrared"] * batch["input_ids"].size(0)
            z_text = self.encode_text_prompt(text_prompts)  # (B, 77, 768)
            
            # 4. Encode images to latent space
            with torch.no_grad():
                # RGB input - ensure proper dtype for VAE (float16)
                rgb_latents = self.vae.encode(batch["rgb_diffusion"].to(dtype=torch.float16)).latent_dist.sample()
                rgb_latents = rgb_latents * 0.18215
                
                # TIR target - ensure proper dtype for VAE (float16)
                tir_latents = self.vae.encode(batch["tir_diffusion"].to(dtype=torch.float16)).latent_dist.sample()
                tir_latents = tir_latents * 0.18215
            
            # 5. Add noise (diffusion forward process)
            noise = torch.randn_like(tir_latents)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (tir_latents.shape[0],),
                device=self.device
            ).long()
            
            noisy_latents = self.noise_scheduler.add_noise(tir_latents, noise, timesteps)
            
            # 6. UNet forward (dual-branch: text + VLM)
            unet_input = torch.cat([noisy_latents, rgb_latents], dim=1)  # (B, 8, H, W)
            
            if self.args.use_custom_processors:
                # APPROACH 1: Custom processors (IP-Adapter style via context)
                from llava.model.ip_context import ip_tokens_context
                with ip_tokens_context(z_vlm.to(z_text.dtype).to(z_text.device)):
                    noise_pred = self.unet(
                        unet_input,
                        timesteps,
                        encoder_hidden_states=z_text,
                    ).sample
            else:
                # APPROACH 2: Dual-forward wrapper (runs UNet twice)
                noise_pred = self.unet(
                    latent_input=unet_input,
                    timestep=timesteps,
                    encoder_hidden_states_text=z_text,      # (B, 77, 768)
                    encoder_hidden_states_vlm=z_vlm,        # (B, 16, 768)
                )
            
            # 7. Compute diffusion loss
            loss_diffusion = F.mse_loss(noise_pred, noise)
            
            # 8. Staged lambda_diffusion (lower during Stage A for stability)
            if self.global_step < self.args.stage_a_steps:
                lambda_diff = self.args.lambda_diffusion_stage_a
            else:
                # Optionally ramp up gradually over 2k steps
                ramp_steps = 2000
                if self.global_step < self.args.stage_a_steps + ramp_steps:
                    progress = (self.global_step - self.args.stage_a_steps) / ramp_steps
                    lambda_diff = self.args.lambda_diffusion_stage_a + progress * (self.args.lambda_diffusion - self.args.lambda_diffusion_stage_a)
                else:
                    lambda_diff = self.args.lambda_diffusion
            
            # 9. Gate regularization (only for custom processors)
            loss_gate_reg = 0.0
            if self.args.use_custom_processors and self.args.gate_reg_weight > 0:
                for proc in self.unet.attn_processors.values():
                    if hasattr(proc, "gamma_text") and hasattr(proc, "gamma_vlm"):
                        loss_gate_reg += (proc.gamma_text - self.args.init_alpha_text) ** 2
                        loss_gate_reg += (proc.gamma_vlm - self.args.init_alpha_vlm) ** 2
                loss_gate_reg = self.args.gate_reg_weight * loss_gate_reg
            
            # 10. Combined loss
            total_loss = self.args.lambda_ce*loss_ce + lambda_diff * loss_diffusion + loss_gate_reg
        
        # Backward
        if self.args.deepspeed:
            # DeepSpeed handles backward, grad clipping, and optimizer step
            self.model_engine.backward(total_loss)
            self.model_engine.step()
        elif self.scaler:
            # AMP with grad scaler
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.llava_model.parameters()) + 
                list(self.edit_mapper.parameters()) + 
                list(self.unet.parameters()),
                self.args.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
        else:
            # Standard backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.llava_model.parameters()) + 
                list(self.edit_mapper.parameters()) + 
                list(self.unet.parameters()),
                self.args.grad_clip
            )
            self.optimizer.step()
            self.lr_scheduler.step()

        
        return {
            'loss_total': total_loss.item(),
            'loss_ce': loss_ce.item(),
            'loss_diffusion': loss_diffusion.item(),
            'loss_gate_reg': loss_gate_reg.item() if isinstance(loss_gate_reg, torch.Tensor) else loss_gate_reg,
            'lambda_diff': lambda_diff,
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.llava_model.train()
        self.edit_mapper.train()
        self.unet.train()
        
        # Set epoch for distributed sampler (important for proper shuffling)
        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)
        
        # Calculate expected optimizer steps for this epoch
        # In distributed training, each GPU sees a subset of the data
        dataloader_iters = len(self.train_loader)
        grad_accum = self.args.gradient_accumulation_steps if (self.args.deepspeed and self.args.gradient_accumulation_steps) else 1
        expected_opt_steps = dataloader_iters // grad_accum
        
        # Create progress bar that tracks optimizer steps (not dataloader iterations)
        pbar = tqdm(total=expected_opt_steps, desc=f"Epoch {self.epoch}")
        
        epoch_start_opt_step = self.global_step
        
        for batch in self.train_loader:
            losses = self.train_step(batch)
            
            # Track dataloader iterations
            self.dataloader_step += 1
            
            # Check if optimizer updated (happens every gradient_accumulation_steps)
            # With DeepSpeed, we need to check the micro_steps counter
            if self.args.deepspeed:
                # DeepSpeed handles gradient accumulation internally
                # We track when the optimizer actually updates
                if self.dataloader_step % grad_accum == 0:
                    self.global_step += 1
                    is_optimizer_step = True
                    # Update progress bar only on optimizer steps
                    pbar.update(1)
                else:
                    is_optimizer_step = False
            else:
                # Non-DeepSpeed: increment every time (no gradient accumulation tracking)
                self.global_step += 1
                is_optimizer_step = True
                pbar.update(1)
            
            # Update progress bar postfix every iteration (to show latest loss)
            pbar.set_postfix({
                'loss': f"{losses['loss_total']:.4f}",
                'ce': f"{losses['loss_ce']:.4f}",
                'diff': f"{losses['loss_diffusion']:.4f}",
            })
            
            # Only log/eval/save on optimizer steps
            if is_optimizer_step:
                # Log
                if self.global_step % self.args.log_interval == 0:
                    self.log_metrics(losses)
                
                # Evaluate
                if self.global_step % self.args.eval_interval == 0 and self.global_step > 0:
                    self.evaluate(num_samples=3)
                
                # Save checkpoint
                if self.global_step % self.args.save_interval == 0 and self.global_step > 0:
                    self.save_checkpoint()
        
        pbar.close()
        self.epoch += 1
    
    @torch.no_grad()
    def evaluate(self, num_samples=5):
        """
        Evaluation: Generate images with text-only, VLM-only, and both.
        
        Saves 3 comparison grids to visualize the contribution of each branch.
        """
        if not self.is_main_process:
            return
        
        if self.val_loader is None or len(self.val_loader) == 0:
            print("No validation data, skipping evaluation")
            return
        
        print(f"\n" + "="*60)
        print(f"EVALUATION at step {self.global_step}")
        print("="*60)
        
        self.llava_model.eval()
        self.edit_mapper.eval()
        self.unet.eval()
        
        # Collect validation samples from different datasets
        # Goal: Get diverse samples (1 from M3FD, 1 from FLIR/thermal, etc.)
        val_samples = []
        seen_datasets = set()
        
        for batch in self.val_loader:
            # Get dataset info if available
            dataset_names = batch.get('dataset', None)
            
            if dataset_names is not None:
                # Iterate through samples in batch
                for idx in range(batch["input_ids"].size(0)):
                    dataset_name = dataset_names[idx] if isinstance(dataset_names, list) else dataset_names
                    
                    # Only take one sample per dataset for diversity
                    if dataset_name not in seen_datasets:
                        sample = {k: v[idx:idx+1] if isinstance(v, torch.Tensor) else v 
                                 for k, v in batch.items()}
                        val_samples.append(sample)
                        seen_datasets.add(dataset_name)
                        print(f"  Selected sample from: {dataset_name}")
                        
                        if len(val_samples) >= num_samples:
                            break
            else:
                # No dataset info, just take first num_samples
                for idx in range(min(num_samples, batch["input_ids"].size(0))):
                    sample = {k: v[idx:idx+1] if isinstance(v, torch.Tensor) else v 
                             for k, v in batch.items()}
                    val_samples.append(sample)
                break
            
            if len(val_samples) >= num_samples:
                break
        
        if len(val_samples) == 0:
            print("No validation samples available")
            return
        
        # Combine samples into a single batch
        val_batch = {}
        for key in val_samples[0].keys():
            if isinstance(val_samples[0][key], torch.Tensor):
                val_batch[key] = torch.cat([s[key] for s in val_samples], dim=0)
            else:
                val_batch[key] = [s[key] for s in val_samples]
        
        # Move to device
        val_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in val_batch.items()}
        
        # Update num_samples
        num_samples = val_batch["input_ids"].size(0)
        print(f"  Using {num_samples} samples from {len(seen_datasets)} different datasets")
        
        # Extract VLM and text conditioning
        # NOTE: img_hiddens extraction requires labels to be present (see llava_llama.py line 112)
        # Create pseudo-labels from input_ids for evaluation (we don't use the loss, just need labels present)
        pseudo_labels = val_batch["input_ids"].clone()
        
        llava_outputs = self.llava_model(
            input_ids=val_batch["input_ids"],
            images=val_batch["images"],
            return_dict=True,
            return_img_hiddens=True,
            img_token_ids=self.img_token_ids,
            labels=pseudo_labels  # Need labels present for img_hiddens extraction
        )
        
        # Extract img_hiddens
        if 'img_hiddens' in llava_outputs:
            img_hiddens = llava_outputs['img_hiddens']
        else:
            print("Warning: 'img_hiddens' not in outputs, skipping evaluation")
            print(f"  Available keys: {llava_outputs.keys() if isinstance(llava_outputs, dict) else 'not a dict'}")
            return
        
        z_vlm = self.edit_mapper(img_hiddens)
        
        text_prompts = ["Turn this image into thermal infrared"] * num_samples
        z_text = self.encode_text_prompt(text_prompts)
        
        # Encode RGB to latents - ensure proper dtype
        rgb_latents = self.vae.encode(val_batch["rgb_diffusion"].to(dtype=torch.float16)).latent_dist.sample() * 0.18215
        
        # Initialize noise
        latents = torch.randn_like(rgb_latents)
        
        # Denoise with different configurations
        results = {}
        
        for mode in ['text_only', 'vlm_only', 'both']:
            print(f"\nGenerating with {mode}...")

            latents_denoised = latents.clone()
            self.noise_scheduler.set_timesteps(50, device=self.device)

            if self.args.use_custom_processors:
                # ---- CUSTOM PROCESSORS PATH ----
                # Save and override gate values per-processor
                saved = []
                for proc in self.unet.attn_processors.values():
                    if hasattr(proc, "gamma_text") and hasattr(proc, "gamma_vlm"):
                        g_t = proc.gamma_text.detach().clone()
                        g_v = proc.gamma_vlm.detach().clone()
                        saved.append((proc, g_t, g_v))
                        if mode == 'text_only':
                            proc.gamma_text.data.fill_(1.0)
                            proc.gamma_vlm.data.fill_(0.0)
                        elif mode == 'vlm_only':
                            proc.gamma_text.data.fill_(0.0)
                            proc.gamma_vlm.data.fill_(1.0)
                        # 'both' -> leave as is

                from llava.model.ip_context import ip_tokens_context

                for t in self.noise_scheduler.timesteps:
                    unet_input = torch.cat([latents_denoised, rgb_latents], dim=1)
                    with ip_tokens_context(z_vlm.to(z_text.dtype).to(z_text.device)):
                        noise_pred = self.unet(
                            unet_input,
                            t.unsqueeze(0).repeat(num_samples),
                            encoder_hidden_states=z_text,
                        ).sample
                    latents_denoised = self.noise_scheduler.step(noise_pred, t, latents_denoised).prev_sample

                # Restore gates
                for proc, g_t, g_v in saved:
                    proc.gamma_text.data.copy_(g_t)
                    proc.gamma_vlm.data.copy_(g_v)

            else:
                # ---- DUAL-FORWARD WRAPPER PATH (your current code) ----
                # Save original alphas
                orig_alpha_text = self.unet.alpha_text.item()
                orig_alpha_vlm = self.unet.alpha_vlm.item()

                # Temporarily override alphas
                if mode == 'text_only':
                    self.unet.alpha_text.data.fill_(1.0)
                    self.unet.alpha_vlm.data.fill_(0.0)
                elif mode == 'vlm_only':
                    self.unet.alpha_text.data.fill_(0.0)
                    self.unet.alpha_vlm.data.fill_(1.0)
                # 'both' uses current alpha values

                # Denoise
                for t in self.noise_scheduler.timesteps:
                    unet_input = torch.cat([latents_denoised, rgb_latents], dim=1)
                    noise_pred = self.unet(
                        latent_input=unet_input,
                        timestep=t.unsqueeze(0).repeat(num_samples),
                        encoder_hidden_states_text=z_text,
                        encoder_hidden_states_vlm=z_vlm,
                    )
                    latents_denoised = self.noise_scheduler.step(noise_pred, t, latents_denoised).prev_sample

                # Restore alphas
                self.unet.alpha_text.data.fill_(orig_alpha_text)
                self.unet.alpha_vlm.data.fill_(orig_alpha_vlm)

            # Decode & collect
            images = self.vae.decode(latents_denoised / 0.18215).sample
            images = (images + 1) / 2
            images = images.clamp(0, 1)
            results[mode] = images
        
        # Save comparison grid
        self.save_eval_grid(val_batch, results)
        
        # Back to training mode
        self.llava_model.train()
        self.edit_mapper.train()
        self.unet.train()
    
    def save_eval_grid(self, batch, results):
        """Save evaluation comparison grid"""
        import torchvision.utils as vutils
        from PIL import Image
        
        eval_dir = os.path.join(self.args.output_dir, f"eval_step_{self.global_step}")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Save individual modes
        for mode, images in results.items():
            grid = vutils.make_grid(images, nrow=images.size(0))
            vutils.save_image(grid, os.path.join(eval_dir, f"{mode}.png"))
        
        # Save RGB and TIR ground truth
        rgb_grid = vutils.make_grid((batch["rgb_diffusion"] + 1) / 2, nrow=batch["rgb_diffusion"].size(0))
        tir_grid = vutils.make_grid((batch["tir_diffusion"] + 1) / 2, nrow=batch["tir_diffusion"].size(0))
        
        vutils.save_image(rgb_grid, os.path.join(eval_dir, "rgb_input.png"))
        vutils.save_image(tir_grid, os.path.join(eval_dir, "tir_gt.png"))
        
        print(f"  ✓ Saved eval grids to: {eval_dir}")
    
    def log_metrics(self, losses):
        """Log metrics to wandb and console (only on main process)"""
        if not self.is_main_process:
            return
        
        # Log alpha/gate values
        if self.args.use_custom_processors:
            # Get average gate values from custom processors
            # Use saved_processors reference (captured before DeepSpeed wrapping)
            alpha_text_vals = []
            alpha_vlm_vals = []
            
            # Use saved processors (captured before DeepSpeed wrapping changes attribute access)
            processors_to_check = self.saved_processors if self.saved_processors is not None else {}
            
            for proc in processors_to_check.values():
                # Check if processor has gamma attributes (trainable gates)
                if hasattr(proc, 'gamma_text') and hasattr(proc, 'gamma_vlm'):
                    alpha_text_vals.append(proc.gamma_text.item())
                    alpha_vlm_vals.append(proc.gamma_vlm.item())
            
            # Debug: Check if gammas are actually training (only once at step 10)
            if self.global_step == 10:
                print(f"\n{'='*80}")
                print(f"GAMMA TRAINING DEBUG (Step 10)")
                print(f"{'='*80}")
                print(f"processors_to_check has {len(processors_to_check)} processors")
                print(f"self.saved_processors is None: {self.saved_processors is None}")
                if processors_to_check:
                    # Count processor types
                    type_counts = {}
                    custom_procs = []
                    for name, proc in processors_to_check.items():
                        ptype = type(proc).__name__
                        type_counts[ptype] = type_counts.get(ptype, 0) + 1
                        if hasattr(proc, 'gamma_text'):
                            custom_procs.append((name, proc))
                    
                    print(f"Processor type distribution:")
                    for ptype, count in type_counts.items():
                        print(f"  {ptype}: {count}")
                    
                    if custom_procs:
                        name, sample_proc = custom_procs[0]
                        print(f"\n✓ Found {len(custom_procs)} custom processors with gamma!")
                        print(f"Sample custom processor: {name}")
                        print(f"  Type: {type(sample_proc).__name__}")
                        print(f"  gamma_text: value={sample_proc.gamma_text.item():.4f}, requires_grad={sample_proc.gamma_text.requires_grad}, grad={sample_proc.gamma_text.grad is not None}")
                        print(f"  gamma_vlm: value={sample_proc.gamma_vlm.item():.4f}, requires_grad={sample_proc.gamma_vlm.requires_grad}, grad={sample_proc.gamma_vlm.grad is not None}")
                        if sample_proc.gamma_text.grad is not None:
                            print(f"  gamma_text grad norm: {sample_proc.gamma_text.grad.norm().item():.6f}")
                        if sample_proc.gamma_vlm.grad is not None:
                            print(f"  gamma_vlm grad norm: {sample_proc.gamma_vlm.grad.norm().item():.6f}")
                    else:
                        print(f"\n✗ NO custom processors found with gamma parameters!")
                        print(f"All processors are defaults (likely --use-custom-processors not set)")
                else:
                    print(f"✗ No processors to check!")
                print(f"{'='*80}\n")
                import sys
                sys.stdout.flush()
            
            # Debug: Print warning if no gamma values found (only first time)
            if not alpha_text_vals and self.global_step <= self.args.log_interval:
                print(f"  ⚠ Warning: No gamma values found in {len(processors_to_check)} processors")
                if processors_to_check:
                    sample_proc = list(processors_to_check.values())[0]
                    print(f"    Processor types: {[type(p).__name__ for p in list(processors_to_check.values())[:3]]}")
                    print(f"    Sample processor type: {type(sample_proc)}")
                    print(f"    Has gamma_text: {hasattr(sample_proc, 'gamma_text')}")
                    print(f"    Has gamma_vlm: {hasattr(sample_proc, 'gamma_vlm')}")
                else:
                    print(f"    No processors found! saved_processors is None: {self.saved_processors is None}")
            
            alpha_values = {
                'alpha_text': sum(alpha_text_vals) / len(alpha_text_vals) if alpha_text_vals else 0.0,
                'alpha_vlm': sum(alpha_vlm_vals) / len(alpha_vlm_vals) if alpha_vlm_vals else 0.0,
            }
        else:
            alpha_values = {
                'alpha_text': self.unet.alpha_text.item(),
                'alpha_vlm': self.unet.alpha_vlm.item(),
            }
        
        if self.args.use_wandb:
            logs = {**losses, **alpha_values, 'epoch': self.epoch, 'step': self.global_step}
            for i, g in enumerate(self.optimizer.param_groups):
                logs[f'lr_group_{i}'] = g['lr']
            wandb.log(logs, step=self.global_step)
        
        # Improved logging with stage and gate regularization info
        stage_info = f"Stage A (λ={losses.get('lambda_diff', 0):.2f})" if self.global_step < self.args.stage_a_steps else f"Stage B (λ={losses.get('lambda_diff', 1.0):.2f})"
        print(f"\n[Step {self.global_step}] {stage_info}")
        print(f"  Loss: {losses['loss_total']:.4f} (CE: {losses['loss_ce']:.4f}, Diff: {losses['loss_diffusion']:.4f}, Gate: {losses.get('loss_gate_reg', 0):.6f})")
        print(f"  Gammas: γ_text={alpha_values['alpha_text']:.3f}, γ_vlm={alpha_values['alpha_vlm']:.3f}")
    
    def save_checkpoint(self):
        """Save checkpoint"""
        ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        
        if self.args.deepspeed:
            # DeepSpeed checkpoint saving is a COLLECTIVE operation - ALL ranks must participate
            # Only main process creates the directory
            if self.is_main_process:
                os.makedirs(ckpt_dir, exist_ok=True)
            
            # Wait for directory creation
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
            # ALL ranks must call save_checkpoint (collective operation)
            # Pass client_state to save training progress
            client_state = {
                'global_step': self.global_step,
                'dataloader_step': self.dataloader_step,
                'epoch': self.epoch,
            }
            self.model_engine.save_checkpoint(ckpt_dir, client_state=client_state)
            
            # Only main process saves tokenizer, processors, and individual model states
            if self.is_main_process:
                # Save tokenizer
                save_tokenizer_with_img_tokens(self.tokenizer, ckpt_dir)
                
                # Save individual model state dicts for easier loading
                torch.save(self.llava_model.state_dict(), f"{ckpt_dir}/llava_model.pt")
                torch.save(self.edit_mapper.state_dict(), f"{ckpt_dir}/edit_mapper.pt")
                
                # Save UNet (use save_pretrained if available, otherwise state_dict)
                if hasattr(self.unet, 'save_pretrained'):
                    self.unet.save_pretrained(os.path.join(ckpt_dir, "unet"))
                else:
                    torch.save(self.unet.state_dict(), f"{ckpt_dir}/unet.pt")
                
                # Save optimizer state dict
                torch.save(self.optimizer.state_dict(), f"{ckpt_dir}/optimizer.pt")
                
                # Save custom processors if present
                if self.args.use_custom_processors:
                    # Only save processors that are nn.Module instances (have state_dict)
                    processors_to_save = {
                        k: v.state_dict() 
                        for k, v in self.unet.attn_processors.items() 
                        if hasattr(v, 'state_dict')
                    }
                    if processors_to_save:
                        torch.save(processors_to_save, os.path.join(ckpt_dir, "attn_processors.pt"))
                
                print(f"\n✓ Checkpoint saved: {ckpt_dir}")
                print(f"  - DeepSpeed checkpoint (for resuming)")
                print(f"  - Individual model states (for inference/loading)")
        else:
            # Non-DeepSpeed: only main process saves
            if not self.is_main_process:
                return
            
            os.makedirs(ckpt_dir, exist_ok=True)
            
            # Save models manually
            torch.save(self.llava_model.state_dict(), f"{ckpt_dir}/llava_model.pt")
            torch.save(self.edit_mapper.state_dict(), f"{ckpt_dir}/edit_mapper.pt")
            self.unet.save_pretrained(os.path.join(ckpt_dir, "dual_branch_unet"))
            
            torch.save(self.optimizer.state_dict(), f"{ckpt_dir}/optimizer.pt")
            
            if self.args.use_custom_processors:
                # Only save processors that are nn.Module instances (have state_dict)
                processors_to_save = {
                    k: v.state_dict() 
                    for k, v in self.unet.attn_processors.items() 
                    if hasattr(v, 'state_dict')
                }
                if processors_to_save:
                    torch.save(processors_to_save, os.path.join(ckpt_dir, "attn_processors.pt"))
            
            # Save tokenizer
            save_tokenizer_with_img_tokens(self.tokenizer, ckpt_dir)
            
            # Save training state metadata
            import json
            training_state = {
                'global_step': self.global_step,
                'dataloader_step': self.dataloader_step,
                'epoch': self.epoch,
            }
            with open(os.path.join(ckpt_dir, 'training_state.json'), 'w') as f:
                json.dump(training_state, f, indent=2)
            
            print(f"\n✓ Checkpoint saved: {ckpt_dir}")
    
    def load_checkpoint(self, ckpt_dir):
        """Load checkpoint to resume training"""
        if self.is_main_process:
            print("\n" + "="*80)
            print(f"LOADING CHECKPOINT FROM: {ckpt_dir}")
            print("="*80)
        
        if not os.path.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        
        if self.args.deepspeed:
            # DeepSpeed checkpoint loading
            if self.is_main_process:
                print("\nLoading DeepSpeed checkpoint...")
            
            # Load DeepSpeed checkpoint (this is a collective operation)
            _, client_state = self.model_engine.load_checkpoint(ckpt_dir)
            
            # Restore training state from client_state (if saved)
            if client_state:
                self.global_step = client_state.get('global_step', 0)
                self.dataloader_step = client_state.get('dataloader_step', 0)
                self.epoch = client_state.get('epoch', 0)
                if self.is_main_process:
                    print(f"  ✓ Restored training state from client_state: epoch={self.epoch}, opt_step={self.global_step}, dl_step={self.dataloader_step}")
            else:
                if self.is_main_process:
                    print("  ⚠ No client state found in DeepSpeed checkpoint")
                
                # Try to load from training_state.json (if saved as individual state)
                import json
                meta_path = os.path.join(ckpt_dir, "training_state.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    self.global_step = meta.get('global_step', 0)
                    self.dataloader_step = meta.get('dataloader_step', 0)
                    self.epoch = meta.get('epoch', 0)
                    if self.is_main_process:
                        print(f"  ✓ Restored from training_state.json: epoch={self.epoch}, opt_step={self.global_step}, dl_step={self.dataloader_step}")
                else:
                    # Fallback: Try to infer from checkpoint name or subdirectory
                    import re
                    # First try to find global_step directory (DeepSpeed format)
                    subdirs = [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d)) and d.startswith('global_step')]
                    if subdirs:
                        # Extract step from directory name like "global_step6"
                        match = re.search(r'global_step(\d+)', subdirs[0])
                        if match:
                            self.global_step = int(match.group(1))
                            if self.is_main_process:
                                print(f"  ✓ Inferred global_step={self.global_step} from DeepSpeed subdirectory: {subdirs[0]}")
                    else:
                        # Try checkpoint directory name (e.g., checkpoint-6)
                        match = re.search(r'checkpoint-(\d+)', ckpt_dir)
                        if match:
                            self.global_step = int(match.group(1))
                            if self.is_main_process:
                                print(f"  ✓ Inferred global_step={self.global_step} from checkpoint name")
                        else:
                            if self.is_main_process:
                                print(f"  ⚠ Could not determine training state, starting from step 0")
                            self.global_step = 0
                    self.epoch = 0
            
            # Optionally load individual model states if DeepSpeed checkpoint incomplete
            # (Usually DeepSpeed handles everything, but individual states provide a fallback)
            llava_path = os.path.join(ckpt_dir, "llava_model.pt")
            mapper_path = os.path.join(ckpt_dir, "edit_mapper.pt")
            
            if os.path.exists(llava_path) and self.is_main_process:
                print(f"  ℹ Individual model states available for inspection/debugging")
        else:
            # Non-DeepSpeed checkpoint loading
            print("\nLoading individual model states...")
            
            # Load model states
            llava_path = os.path.join(ckpt_dir, "llava_model.pt")
            mapper_path = os.path.join(ckpt_dir, "edit_mapper.pt")
            optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")
            
            if os.path.exists(llava_path):
                self.llava_model.load_state_dict(torch.load(llava_path, map_location=self.device))
                print(f"  ✓ Loaded LLaVA model")
            else:
                print(f"  ⚠ LLaVA checkpoint not found at {llava_path}")
            
            if os.path.exists(mapper_path):
                self.edit_mapper.load_state_dict(torch.load(mapper_path, map_location=self.device))
                print(f"  ✓ Loaded EditMapper")
            else:
                print(f"  ⚠ EditMapper checkpoint not found at {mapper_path}")
            
            # Load UNet
            unet_dir = os.path.join(ckpt_dir, "unet")
            unet_pt = os.path.join(ckpt_dir, "unet.pt")
            
            if os.path.exists(unet_dir) and hasattr(self.unet, 'from_pretrained'):
                # Load from saved directory
                print(f"  ✓ Loading UNet from {unet_dir}")
                # Note: This may require manual loading depending on wrapper type
            elif os.path.exists(unet_pt):
                self.unet.load_state_dict(torch.load(unet_pt, map_location=self.device))
                print(f"  ✓ Loaded UNet from state dict")
            else:
                print(f"  ⚠ UNet checkpoint not found")
            
            # Load custom processors if present
            proc_path = os.path.join(ckpt_dir, "attn_processors.pt")
            if self.args.use_custom_processors and os.path.exists(proc_path):
                proc_state = torch.load(proc_path, map_location=self.device)
                for name, state_dict in proc_state.items():
                    if hasattr(self.unet.attn_processors[name], 'load_state_dict'):
                        self.unet.attn_processors[name].load_state_dict(state_dict)
                print(f"  ✓ Loaded {len(proc_state)} custom processors")
            
            # Load optimizer
            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
                print(f"  ✓ Loaded optimizer state")
            else:
                print(f"  ⚠ Optimizer checkpoint not found at {optimizer_path}")
            
            # Try to load training state from a metadata file
            meta_path = os.path.join(ckpt_dir, "training_state.json")
            if os.path.exists(meta_path):
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                self.global_step = meta.get('global_step', 0)
                self.dataloader_step = meta.get('dataloader_step', 0)
                self.epoch = meta.get('epoch', 0)
                print(f"  ✓ Restored training state: epoch={self.epoch}, opt_step={self.global_step}, dl_step={self.dataloader_step}")
            else:
                # Try to infer from checkpoint name (e.g., checkpoint-1000 -> step 1000)
                import re
                match = re.search(r'checkpoint-(\d+)', ckpt_dir)
                if match:
                    self.global_step = int(match.group(1))
                    print(f"  ✓ Inferred global_step={self.global_step} from checkpoint name")
                else:
                    print(f"  ⚠ Could not determine training state, starting from step 0")
                    self.global_step = 0
                self.epoch = 0
        
        if self.is_main_process:
            print(f"\n✓ Checkpoint loaded successfully!")
            print(f"  Resuming from: epoch={self.epoch}, global_step={self.global_step}")
    
    def train(self):
        """Main training loop"""
        if self.is_main_process:
            print("\n" + "="*80)
            print("STARTING TRAINING")
            print("="*80)
        
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.run_name,
                config=vars(self.args)
            )
        
        for epoch in range(self.args.num_epochs):
            self.train_epoch()
        
        # Final save
        self.save_checkpoint()
        
        if self.args.use_wandb:
            wandb.finish()

    def debug_one_step(self):
        """Run a single debug step: forward+backward and print key diagnostics."""
        print("\n" + "="*80)
        print("DEBUG: One step forward/backward + diagnostics")
        print("="*80)

        self.llava_model.train()
        # UNet remains in eval (frozen base), processors carry grads
        self.unet.eval()
        self.edit_mapper.train()

        batch = next(iter(self.train_loader))
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Shapes/dtypes
        print(f"  input_ids: {tuple(batch['input_ids'].shape)} dtype={batch['input_ids'].dtype}")
        print(f"  images: {tuple(batch['images'].shape)} dtype={batch['images'].dtype}")

        # LLaVA forward for VLM tokens
        out = self.llava_model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            images=batch['images'],
            return_dict=True,
            return_img_hiddens=True,
            img_token_ids=self.img_token_ids,
        )
        z_vlm = self.edit_mapper(out['img_hiddens'])  # (B, L_vlm, 768)
        print(f"  z_vlm: {tuple(z_vlm.shape)} dtype={z_vlm.dtype} device={z_vlm.device}")

        # Text
        text_prompts = ["Turn this image into thermal infrared"] * batch['input_ids'].size(0)
        z_text = self.encode_text_prompt(text_prompts)
        print(f"  z_text: {tuple(z_text.shape)} dtype={z_text.dtype} device={z_text.device}")

        # Latents - ensure proper dtype
        with torch.no_grad():
            rgb_latents = self.vae.encode(batch["rgb_diffusion"].to(dtype=torch.float16)).latent_dist.sample() * 0.18215
            tir_latents = self.vae.encode(batch["tir_diffusion"].to(dtype=torch.float16)).latent_dist.sample() * 0.18215
        noise = torch.randn_like(tir_latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (tir_latents.shape[0],), device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(tir_latents, noise, timesteps)
        unet_input = torch.cat([noisy_latents, rgb_latents], dim=1)
        print(f"  unet_input: {tuple(unet_input.shape)} dtype={unet_input.dtype} device={unet_input.device}")

        # Forward
        if self.args.use_custom_processors:
            from llava.model.ip_context import ip_tokens_context
            with ip_tokens_context(z_vlm.to(z_text.dtype).to(z_text.device)):
                noise_pred = self.unet(unet_input, timesteps, encoder_hidden_states=z_text).sample
        else:
            noise_pred = self.unet(
                latent_input=unet_input,
                timestep=timesteps,
                encoder_hidden_states_text=z_text,
                encoder_hidden_states_vlm=z_vlm,
            )
        print(f"  noise_pred: {tuple(noise_pred.shape)} dtype={noise_pred.dtype} device={noise_pred.device}")

        loss_diff = F.mse_loss(noise_pred, noise)
        loss_ce = out['loss']
        total_loss = self.args.lambda_ce*loss_ce + self.args.lambda_diffusion * loss_diff
        print(f"  losses: ce={loss_ce.item():.4f} diff={loss_diff.item():.4f} total={total_loss.item():.4f}")

        # Backward (no optimizer step)
        (self.scaler.scale(total_loss) if self.scaler else total_loss).backward()

        # Print gates and LoRA grad presence
        if self.args.use_custom_processors:
            proc_dict = self.unet.attn_processors
            gate_vals = []
            lora_grads = []
            from llava.model.unet_processors import DecoupledDualBranchAttnProcessor, LoRALinear
            for name, proc in proc_dict.items():
                if isinstance(proc, DecoupledDualBranchAttnProcessor):
                    gate_vals.append((proc.gamma_text.item(), proc.gamma_vlm.item()))
                    for layer in (proc.to_k_vlm, proc.to_v_vlm):
                        if isinstance(layer, LoRALinear):
                            lora_grads.append((layer.lora_A.grad is not None, layer.lora_B.grad is not None))
            if gate_vals:
                g_text_mean = sum(g[0] for g in gate_vals) / len(gate_vals)
                g_vlm_mean = sum(g[1] for g in gate_vals) / len(gate_vals)
                print(f"  gates mean: gamma_text={g_text_mean:.4f}, gamma_vlm={g_vlm_mean:.4f}")
            if lora_grads:
                ok = all(a and b for (a, b) in lora_grads)
                print(f"  VLM LoRA grads present on all processors: {ok}")

        # Clear grads
        self.llava_model.zero_grad(set_to_none=True)
        self.edit_mapper.zero_grad(set_to_none=True)
        if not self.args.use_custom_processors:
            self.unet.zero_grad(set_to_none=True)
        print("  ✓ Debug step completed")


def parse_args():
    parser = argparse.ArgumentParser(description="Train VLM-guided InstructPix2Pix")
    
    # Model paths
    parser.add_argument("--llava-path", type=str, default=None,
                       help="Path to merged LLaVA model")
    parser.add_argument("--llava-base-path", type=str, default=None,
                       help="Path to base LLaVA (if using LoRA)")
    parser.add_argument("--llava-lora-path", type=str, default=None,
                       help="Path to LLaVA LoRA weights")
    parser.add_argument("--ip2p-pretrained", type=str, required=True,
                       help="Path to pretrained InstructPix2Pix")
    parser.add_argument("--ip2p-checkpoint", type=str, default=None,
                       help="Path to fine-tuned IP2P checkpoint (train_state.pt)")
    
    # Data
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to training JSON")
    parser.add_argument("--image-folder", type=str, required=True,
                       help="Root folder for images")
    parser.add_argument("--filter-datasets", type=str, nargs='+', 
                       default=["DTUAV", "visdrone"],
                       help="Datasets to exclude")
    parser.add_argument("--validation-datasets", type=str, nargs='+',
                       default=["M3FD", "FLIR"],
                       help="Datasets to use for validation")
    
    # Training
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for checkpoints")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                       help="Path to checkpoint directory to resume from")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lr-unet", type=float, default=1e-4,
                       help="Learning rate for UNet LoRA")
    parser.add_argument("--lr-mapper", type=float, default=1e-4,
                       help="Learning rate for EditMapper")
    parser.add_argument("--lr-llava", type=float, default=5e-6,
                       help="Learning rate for LLaVA language matrices")
    parser.add_argument("--lr-min", type=float, default=1e-7,
                       help="Minimum LR for cosine schedule")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-diffusion", type=float, default=1.0,
                       help="Weight for diffusion loss")
    parser.add_argument("--lambda-ce", type=float, default=0.5,
                    help="Weight for CE loss")
    parser.add_argument("--warmup-steps", type=int, default=2000,
                    help="Number of warmup steps for learning rate")
    parser.add_argument("--stage-a-steps", type=int, default=3000,
                    help="Number of steps for Stage A (CE+Mapper only, low diffusion loss)")
    parser.add_argument("--lambda-diffusion-stage-a", type=float, default=0.1,
                    help="Diffusion loss weight during Stage A")
    parser.add_argument("--gate-reg-weight", type=float, default=1e-3,
                    help="L2 regularization weight for gate parameters (keeps them near init)")
    parser.add_argument("--lora-on-text-backbone", action="store_true",
                    help="Add small LoRA (rank 4) on backbone K/V for better adaptation")
    parser.add_argument("--backbone-lora-r", type=int, default=4,
                    help="LoRA rank for backbone K/V (if --lora-on-text-backbone is set)")

    # Model config
    parser.add_argument("--mapper-mid-dim", type=int, default=1024,
                       help="EditMapper hidden dimension")
    parser.add_argument("--use-mgie-mapper", action="store_true",
                       help="Use MGIE-style Transformer mapper instead of lightweight MLP")
    parser.add_argument("--train-full-unet", action="store_true",
                       help="Train full UNet (not just LoRA). Requires more VRAM but better quality.")
    
    # Dual-branch approach selection
    parser.add_argument("--use-custom-processors", action="store_true",
                       help="Use custom attention processors (APPROACH 1) instead of dual-forward (APPROACH 2)")
    parser.add_argument("--target-blocks", type=str, nargs='+',
                       default=["down_blocks.2", "mid_block"],
                       help="UNet blocks for custom processors (only if --use-custom-processors)")
    
    # UNet fine-tuning
    parser.add_argument("--use-unet-lora", action="store_true",
                       help="Use LoRA for UNet (only for dual-forward approach)")
    parser.add_argument("--lora-r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=8,
                       help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                       help="LoRA dropout")
    
    # Mixing weights
    parser.add_argument("--init-alpha-text", type=float, default=0.3,
                       help="Initial weight for text branch")
    parser.add_argument("--init-alpha-vlm", type=float, default=0.7,
                       help="Initial weight for VLM branch")
    parser.add_argument("--image-aspect-ratio", type=str, default="square",
                       choices=["square", "pad"])
    parser.add_argument("--diffusion-size", type=int, default=512,
                       help="Image size for diffusion")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=500,
                       help="Evaluation interval (generates 3-way comparison)")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="vlm-guided-ip2p")
    parser.add_argument("--run-name", type=str, default=None)
    
    # Mixed precision
    parser.add_argument("--use-amp", action="store_true",
                       help="Use automatic mixed precision")
    
    # Gradient accumulation
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                       help="Number of gradient accumulation steps (overrides DeepSpeed config if provided)")
    
    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None,
                       help="Path to DeepSpeed config JSON")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    
    # Debug
    parser.add_argument("--debug-one-step", action="store_true",
                       help="Run a single debug forward+backward step and exit")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create trainer
    trainer = VLMGuidedIP2PTrainer(args)
    
    # Optional one-shot debug step
    if args.debug_one_step:
        trainer.debug_one_step()
        print("\nDEBUG COMPLETE - exiting")
        return
    
    # Train
    trainer.train()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

