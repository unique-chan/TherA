# llava/train/train_rgb_tir.py (New file)
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import torch.nn as nn
import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMG_TOKENS
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer
from llava.data.rgb_tir_dataset import RGBTIRDataset

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
import torch.nn.functional as F

# Import diffusion components
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    
    # New arguments for RGB-TIR
    instructpix2pix_path: Optional[str] = field(default=None)
    enable_diffusion_loss: bool = field(default=True)
    diffusion_loss_weight: float = field(default=1.0)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_size: int = field(default=512, metadata={"help": "Image size for diffusion model"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=2048)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

class IMGTokenProjector(nn.Module):
    """Project IMG tokens to diffusion features"""
    
    def __init__(self, hidden_size: int, diffusion_hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        self.diffusion_hidden_size = diffusion_hidden_size
        
        # Project IMG tokens to diffusion features
        self.img_projection = nn.Linear(hidden_size, diffusion_hidden_size)
        
    def forward(self, img_token_embeddings):
        """
        Args:
            img_token_embeddings: [batch_size, num_img_tokens, hidden_size]
        Returns:
            diffusion_features: [batch_size, num_img_tokens, diffusion_hidden_size]
        """
        return self.img_projection(img_token_embeddings)

class RGBTIRTrainer(LLaVATrainer):
    """Custom trainer for RGB-TIR translation with diffusion loss"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize diffusion components
        self._init_diffusion_components()
        
        # Initialize IMG token projector
        self.img_projector = IMGTokenProjector(
            hidden_size=self.model.config.hidden_size,
            diffusion_hidden_size=768  # CLIP hidden size
        ).to(self.model.device)
        
    def _init_diffusion_components(self):
        """Initialize InstructPix2Pix components"""
        if self.args.instructpix2pix_path:
            # Load InstructPix2Pix components
            self.unet = UNet2DConditionModel.from_pretrained(
                self.args.instructpix2pix_path, subfolder="unet"
            ).to(self.model.device)
            
            self.vae = AutoencoderKL.from_pretrained(
                self.args.instructpix2pix_path, subfolder="vae"
            ).to(self.model.device)
            
            # Freeze diffusion components
            self.unet.requires_grad_(False)
            self.vae.requires_grad_(False)
            
            print("Loaded InstructPix2Pix components")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute combined LLaVA + diffusion loss"""
        
        # Standard LLaVA loss
        outputs = model(**inputs)
        llava_loss = outputs.loss
        
        if not self.args.enable_diffusion_loss:
            return (llava_loss, outputs) if return_outputs else llava_loss
        
        # Extract IMG token embeddings
        img_token_embeddings = self._extract_img_token_embeddings(model, inputs)
        
        # Project to diffusion features
        diffusion_features = self.img_projector(img_token_embeddings)
        
        # Compute diffusion loss
        diffusion_loss = self._compute_diffusion_loss(diffusion_features, inputs)
        
        # Combined loss
        total_loss = llava_loss + self.args.diffusion_loss_weight * diffusion_loss
        
        if return_outputs:
            outputs.loss = total_loss
            return total_loss, outputs
        return total_loss
    
    def _extract_img_token_embeddings(self, model, inputs):
        """Extract embeddings for IMG tokens"""
        # Get model outputs
        outputs = model(**inputs, output_hidden_states=True)
        
        # Find IMG token positions in the sequence
        img_token_ids = [model.tokenizer.convert_tokens_to_ids(token) for token in IMG_TOKENS]
        
        # Extract embeddings for IMG tokens
        hidden_states = outputs.hidden_states[-1]  # Last layer
        input_ids = inputs['input_ids']
        
        batch_size = input_ids.size(0)
        img_embeddings = []
        
        for i in range(batch_size):
            # Find IMG token positions
            img_positions = []
            for token_id in img_token_ids:
                positions = (input_ids[i] == token_id).nonzero(as_tuple=True)[0]
                img_positions.extend(positions.tolist())
            
            if img_positions:
                # Extract embeddings
                img_emb = hidden_states[i, img_positions, :]  # [num_img_tokens, hidden_size]
                img_embeddings.append(img_emb)
            else:
                # Create zero embeddings if no IMG tokens found
                zero_emb = torch.zeros(len(IMG_TOKENS), hidden_states.size(-1), device=hidden_states.device)
                img_embeddings.append(zero_emb)
        
        return torch.stack(img_embeddings)  # [batch_size, num_img_tokens, hidden_size]
    
    def _compute_diffusion_loss(self, diffusion_features, inputs):
        """Compute loss between projected features and target thermal image"""
        if not hasattr(self, 'unet'):
            return torch.tensor(0.0, device=diffusion_features.device)
        
        # Get target thermal images
        tir_images = inputs['tir_diffusion']  # [batch_size, 3, 512, 512]
        
        # Encode thermal images to latent space
        with torch.no_grad():
            tir_latents = self.vae.encode(tir_images).latent_dist.sample()
            tir_latents = tir_latents * self.vae.config.scaling_factor
        
        # Simple MSE loss between projected features and thermal latents
        # This is a simplified approach - you might want to use more sophisticated loss
        target_features = tir_latents.mean(dim=(2, 3))  # [batch_size, 4]
        target_features = target_features.unsqueeze(1).expand(-1, len(IMG_TOKENS), -1)  # [batch_size, num_img_tokens, 4]
        
        # Project target to same dimension
        target_projected = torch.zeros_like(diffusion_features)
        target_projected[:, :, :4] = target_features
        
        loss = F.mse_loss(diffusion_features, target_projected)
        return loss

def train():
    global local_rank
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    
    # Load model and tokenizer
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2"
    )
    
    model.config.use_cache = False
    
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # Load image processor
    image_processor = transformers.CLIPImageProcessor.from_pretrained(
        model_args.vision_tower,
        cache_dir=training_args.cache_dir,
    )
    
    # Create dataset
    dataset = RGBTIRDataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_max_length=training_args.model_max_length,
        image_size=data_args.image_size,
    )
    
    # Create trainer
    trainer = RGBTIRTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=None,  # Will use default
    )
    
    # Train
    trainer.train()
    trainer.save_state()
    
    # Save model
    model.config.use_cache = True
    trainer.save_model()

if __name__ == "__main__":
    train()