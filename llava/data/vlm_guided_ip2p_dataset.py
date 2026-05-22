"""
Dataset for VLM-guided InstructPix2Pix training.

Combines LLaVA's structured text understanding with RGB-TIR pairs for diffusion training.
"""

import os
import json
import copy
from typing import Dict, List, Optional
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMG_TOKENS
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token


def _safe_replace_rgb_with_tir(path: str) -> str:
    """Replace /RGB/ with /TIR/ in path to get thermal pair"""
    return path.replace("/RGB/", "/TIR/")


def _expand2square(pil_img: Image.Image, background_color=(0, 0, 0)):
    """Expand image to square by padding"""
    w, h = pil_img.size
    if w == h:
        return pil_img
    if w > h:
        result = Image.new(pil_img.mode, (w, w), background_color)
        result.paste(pil_img, (0, (w - h) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (h, h), background_color)
        result.paste(pil_img, ((h - w) // 2, 0))
        return result


class VLMGuidedIP2PDataset(Dataset):
    """
    Dataset for VLM-guided InstructPix2Pix training.
    
    Loads RGB-TIR pairs from MIRAGE dataset and prepares:
    1. LLaVA inputs (text + image) with IMG tokens appended to response
    2. Diffusion pairs (RGB and TIR images for InstructPix2Pix)
    
    Args:
        data_path: Path to JSON file (e.g., data/llava_no_ranking_nobbox.json)
        image_folder: Root folder for images (e.g., ../MIRAGE_HD)
        tokenizer: LLaVA tokenizer (with IMG tokens added)
        image_processor: CLIP image processor for LLaVA
        img_token_ids: List of IMG token IDs
        image_aspect_ratio: "square" or "pad" (for LLaVA)
        diffusion_size: Image size for diffusion model (default: 512)
        model_max_length: Max sequence length (default: 2048)
        filter_datasets: List of dataset names to exclude (e.g., ["DTUAV", "visdrone"])
        validation_datasets: List of dataset names to use as validation (e.g., ["M3FD", "FLIR"])
        is_validation: If True, use validation datasets; if False, use training datasets
    """
    
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer,
        image_processor,
        img_token_ids: List[int],
        image_aspect_ratio: str = "square",
        diffusion_size: int = 512,
        model_max_length: int = 2048,
        filter_datasets: Optional[List[str]] = None,
        validation_datasets: Optional[List[str]] = None,
        is_validation: bool = False,
    ):
        super().__init__()
        
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.img_token_ids = img_token_ids
        self.image_aspect_ratio = image_aspect_ratio
        self.diffusion_size = diffusion_size
        self.model_max_length = model_max_length
        
        # Load raw data
        print(f"Loading dataset from: {data_path}")
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        print(f"  Raw dataset size: {len(raw_data)}")
        
        # Filter datasets
        if filter_datasets is None:
            filter_datasets = ["DTUAV", "visdrone"]  # Default exclusions
        
        if validation_datasets is None:
            validation_datasets = ["M3FD", "FLIR"]  # Default validation sets
        
        # Split into train/val
        self.data = []
        train_count = 0
        val_count = 0
        filtered_count = 0
        
        for item in raw_data:
            image_path = item.get("image", "")
            
            # Check if should be filtered out
            if any(excluded in image_path for excluded in filter_datasets):
                filtered_count += 1
                continue
            
            # Check if validation
            is_val_item = any(val_dataset in image_path for val_dataset in validation_datasets)
            
            if is_validation and is_val_item:
                self.data.append(item)
                val_count += 1
            elif not is_validation and not is_val_item:
                self.data.append(item)
                train_count += 1
        
        split_name = "VALIDATION" if is_validation else "TRAINING"
        print(f"  {split_name} dataset size: {len(self.data)}")
        print(f"  Filtered out: {filtered_count} (from {filter_datasets})")
        if is_validation:
            print(f"  Validation sources: {validation_datasets}")
        else:
            print(f"  Excluded from training: {validation_datasets}")
        
        # IMG token string (append to GPT response)
        self.img_token_str = " " + " ".join([self.tokenizer.decode([tid], skip_special_tokens=False) 
                                              for tid in self.img_token_ids])
        
        print(f"  IMG token string: '{self.img_token_str}'")
        
        # Counter for truncated samples
        self.truncation_count = 0
        
        # Diffusion transforms
        self.diffusion_transform = transforms.Compose([
            transforms.Lambda(lambda img: _expand2square(img, (0, 0, 0))),
            transforms.Resize(diffusion_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
        ])
    
    def __len__(self):
        return len(self.data)
    
    def get_truncation_stats(self):
        """Get statistics about truncated samples"""
        total = len(self.data)
        truncated = self.truncation_count
        pct = 100.0 * truncated / total if total > 0 else 0.0
        return {
            'total_samples': total,
            'truncated_samples': truncated,
            'truncation_rate': pct
        }
    
    def print_truncation_stats(self):
        """Print truncation statistics"""
        stats = self.get_truncation_stats()
        print(f"\nDataset truncation stats:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Truncated: {stats['truncated_samples']} ({stats['truncation_rate']:.2f}%)")
        if stats['truncation_rate'] > 5.0:
            print(f"  ⚠️  High truncation rate! Consider increasing model_max_length")
        elif stats['truncation_rate'] > 0:
            print(f"  ✓ Low truncation rate (acceptable)")
    
    def _preprocess_multimodal(self, sources: List[List[Dict]]) -> List[List[Dict]]:
        """Preprocess multimodal conversations (handle <image> token)"""
        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                    sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                    sentence["value"] = sentence["value"].strip()
                    
                    # Handle mm_use_im_start_end if needed
                    if hasattr(self.image_processor, 'image_mean'):
                        # Standard CLIP processor
                        replace_token = DEFAULT_IMAGE_TOKEN
                    else:
                        replace_token = DEFAULT_IMAGE_TOKEN
                    
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
        
        return sources
    
    def _preprocess_text(self, sources: List[List[Dict]], has_image: bool = True):
        """Preprocess text conversations into input_ids and labels"""
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        
        conversations = []
        for source in sources:
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip non-human first message
                source = source[1:]
            
            conv.messages = []
            for sentence in source:
                role = roles[sentence["from"]]
                conv.append_message(role, sentence["value"])
            
            conversations.append(conv.get_prompt())
        
        # Tokenize
        if has_image:
            input_ids = torch.stack([
                tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
                for prompt in conversations
            ], dim=0)
        else:
            input_ids = self.tokenizer(
                conversations,
                return_tensors="pt",
                padding="longest",
                max_length=self.model_max_length,
                truncation=True
            ).input_ids
        
        targets = input_ids.clone()
        
        # Check for truncation (guard against long samples)
        for seq in input_ids:
            if len(seq) >= self.model_max_length:
                self.truncation_count += 1
        
        # Mask human turns (keep only GPT responses in labels)
        if conv.sep_style == conversation_lib.SeparatorStyle.TWO:
            sep = conv.sep + conv.roles[1] + ": "
            for conversation, target in zip(conversations, targets):
                total_len = int(target.ne(self.tokenizer.pad_token_id).sum())
                rounds = conversation.split(conv.sep2)
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                
                for i, rou in enumerate(rounds):
                    if rou == "":
                        break
                    
                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    
                    parts[0] += sep
                    
                    if has_image:
                        round_len = len(tokenizer_image_token(rou, self.tokenizer))
                        instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) - 2
                    else:
                        round_len = len(self.tokenizer(rou).input_ids)
                        instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2
                    
                    # Mask instruction (human part)
                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
                    
                    cur_len += round_len
                
                target[cur_len:] = IGNORE_INDEX
                
                # Safety check
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
        
        return dict(input_ids=input_ids[0], labels=targets[0])
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns:
            input_ids: Tokenized input
            labels: Labels with human turns masked (IMG tokens NOT masked!)
            image: Preprocessed RGB image for LLaVA
            rgb_diffusion: RGB image for diffusion model (512x512, [-1,1])
            tir_diffusion: TIR image for diffusion model (512x512, [-1,1])
        """
        item = self.data[idx]
        
        # Get image paths
        rgb_rel_path = item["image"]
        rgb_path = os.path.join(self.image_folder, rgb_rel_path)
        tir_path = _safe_replace_rgb_with_tir(rgb_path)
        
        # Load images
        try:
            rgb_image = Image.open(rgb_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load RGB image: {rgb_path}") from e
        
        try:
            tir_image = Image.open(tir_path).convert("RGB")
        except Exception as e:
            # If TIR doesn't exist, use RGB as fallback (will print warning)
            print(f"Warning: TIR not found, using RGB: {tir_path}")
            tir_image = rgb_image.copy()
        
        # Prepare LLaVA image (using CLIP processor)
        if self.image_aspect_ratio == "pad":
            bg = tuple(int(x * 255) for x in self.image_processor.image_mean)
            rgb_image_llava = _expand2square(rgb_image, bg)
        else:
            rgb_image_llava = rgb_image
        
        image_tensor = self.image_processor.preprocess(rgb_image_llava, return_tensors="pt")["pixel_values"][0]
        
        # Prepare diffusion images (512x512, [-1, 1])
        rgb_diffusion = self.diffusion_transform(rgb_image)
        tir_diffusion = self.diffusion_transform(tir_image)
        
        # Prepare conversation with IMG tokens appended
        conversations = copy.deepcopy(item["conversations"])
        
        # Append IMG tokens to the GPT response
        if len(conversations) >= 2 and conversations[1]["from"] == "gpt":
            conversations[1]["value"] = conversations[1]["value"] + self.img_token_str
        else:
            raise ValueError(f"Expected gpt response in conversations[1], got: {conversations}")
        
        # Build sources for tokenization
        sources = [conversations]
        
        # Preprocess multimodal
        sources = self._preprocess_multimodal(copy.deepcopy(sources))
        
        # Tokenize and create labels
        text_dict = self._preprocess_text(sources, has_image=True)
        
        return {
            "input_ids": text_dict["input_ids"],
            "labels": text_dict["labels"],
            "image": image_tensor,
            "rgb_diffusion": rgb_diffusion,
            "tir_diffusion": tir_diffusion,
        }


def create_dataloaders(
    data_path: str,
    image_folder: str,
    tokenizer,
    image_processor,
    img_token_ids: List[int],
    batch_size: int = 4,
    num_workers: int = 4,
    image_aspect_ratio: str = "square",
    diffusion_size: int = 512,
    filter_datasets: Optional[List[str]] = None,
    validation_datasets: Optional[List[str]] = None,
):
    """
    Create training and validation dataloaders.
    
    Args:
        data_path: Path to JSON file
        image_folder: Root folder for images
        tokenizer: LLaVA tokenizer (with IMG tokens)
        image_processor: CLIP image processor
        img_token_ids: List of IMG token IDs
        batch_size: Batch size
        num_workers: Number of workers
        image_aspect_ratio: "square" or "pad"
        diffusion_size: Image size for diffusion
        filter_datasets: Datasets to exclude entirely (default: ["DTUAV", "visdrone"])
        validation_datasets: Datasets to use as validation (default: ["M3FD", "FLIR"])
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    # Create train dataset
    train_dataset = VLMGuidedIP2PDataset(
        data_path=data_path,
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        img_token_ids=img_token_ids,
        image_aspect_ratio=image_aspect_ratio,
        diffusion_size=diffusion_size,
        filter_datasets=filter_datasets,
        validation_datasets=validation_datasets,
        is_validation=False  # Training set
    )
    
    # Create validation dataset
    val_dataset = VLMGuidedIP2PDataset(
        data_path=data_path,
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        img_token_ids=img_token_ids,
        image_aspect_ratio=image_aspect_ratio,
        diffusion_size=diffusion_size,
        filter_datasets=filter_datasets,
        validation_datasets=validation_datasets,
        is_validation=True  # Validation set
    )
    
    # Custom collate function
    def collate_fn(batch):
        """Collate batch items"""
        # Find max length for padding
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # Stack images
        images = torch.stack([item["image"] for item in batch])
        rgb_diffusion = torch.stack([item["rgb_diffusion"] for item in batch])
        tir_diffusion = torch.stack([item["tir_diffusion"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "images": images,
            "rgb_diffusion": rgb_diffusion,
            "tir_diffusion": tir_diffusion,
        }
    
    # Check if we're in distributed training
    import torch.distributed as dist
    is_distributed = dist.is_available() and dist.is_initialized()
    
    # Create samplers for distributed training
    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=False
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False
        ) if len(val_dataset) > 0 else None
    else:
        train_sampler = None
        val_sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    ) if len(val_dataset) > 0 else None
    
    print(f"\nDataloaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader) if val_loader else 0}")
    
    return train_loader, val_loader

