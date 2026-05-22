"""
Dataset for Frozen VLM-guided InstructPix2Pix Training

Key features:
1. Per-sequence train/val split (98:2) to avoid temporal leakage
2. Excludes FLIR/M3FD/DTUAV/visdrone (for later domain adaptation)
3. Uses VTMOT_test/VIVID/NSAVP sequences
4. Separate prompts for CLIP (simple) and LLaVA (detailed)
5. Resizes to 256×256 (matching IP2P checkpoint)
"""

import json
import os
import random
from typing import List, Dict, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision import transforms


class FrozenVLMIP2PDataset(Dataset):
    """
    Dataset for training Frozen VLM-guided InstructPix2Pix.
    
    Returns:
    - rgb_image: RGB input (256×256)
    - thermal_image: Thermal ground truth (256×256)
    - clip_prompt: Simple prompt for CLIP text encoder
    - llava_prompt: Detailed prompt for frozen LLaVA
    - metadata: Additional info (sequence, dataset, etc.)
    """
    def __init__(
        self,
        json_path: str,
        image_folder: str,  # Can be a single folder or list of folders
        split: str = "train",
        thermal_templates_path: str = "llava/data/thermal_templates.txt",
        size: int = 256,
        flip_p: float = 0.0,  # No flip for now (thermal may not be symmetric)
        train_split_ratio: float = 0.98,  # Train/val split ratio (default 98:2)
    ):
        super().__init__()
        
        self.json_path = json_path
        # Support multiple image folder roots
        if isinstance(image_folder, str):
            self.image_folders = [image_folder]
        else:
            self.image_folders = image_folder
        self.split = split
        self.train_split_ratio = train_split_ratio
        self.size = size
        self.flip_p = flip_p
        
        # Load thermal templates for CLIP
        self.thermal_templates = self._load_thermal_templates(thermal_templates_path)
        
        # Load and filter data
        self.data = self._load_and_filter_data()
        
        # Split by sequence
        self.data = self._split_by_sequence()
        
        print(f"[{split.upper()}] Loaded {len(self.data)} samples from {len(set(s['sequence'] for s in self.data))} sequences")
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1]
        ])
    
    def _load_thermal_templates(self, path: str) -> List[str]:
        """Load CLIP prompt templates"""
        if not os.path.exists(path):
            # Default templates
            return [
                "turn this into thermal infrared image",
                "convert to thermal infrared",
                "transform to thermal image",
            ]
        
        with open(path, 'r') as f:
            templates = [line.strip() for line in f if line.strip()]
        return templates
    
    def _load_and_filter_data(self) -> List[Dict]:
        """
        Load JSON and filter out excluded datasets.
        Exclude: FLIR, M3FD, DTUAV, visdrone (even if they're in IR500k)
        Include: IR500k, VTMOT_test, VIVID, NSAVP, CAMEL, MetuVis, STheReO, TRI2I, others
        
        Supports two JSON formats:
        1. Old format: {"image": "path/to/RGB/file.jpg", ...}
        2. New format: {"rgb": "path/to/rgb.jpg", "thermal": "path/to/thermal.jpg", ...}
        """
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        # Always exclude these datasets (including typo variant DUTUAV)
        # Note: FLIR and M3FD are now included in training
        excluded_keywords = ['DTUAV', 'DUTUAV', 'visdrone']
        
        filtered_data = []
        for item in data:
            # Handle both old and new JSON formats
            if 'rgb' in item and 'thermal' in item:
                # New format with explicit rgb/thermal paths
                rgb_path = item['rgb']
                thermal_path = item['thermal']
            elif 'image' in item:
                # Old format with single image path
                rgb_path = item['image']
                thermal_path = rgb_path.replace('/RGB/', '/TIR/')
            else:
                print(f"Warning: Item has neither 'rgb'/'thermal' nor 'image' key: {item}")
                continue
            
            # Check if any excluded keyword is in the paths
            if any(keyword.lower() in rgb_path.lower() for keyword in excluded_keywords):
                continue
            if any(keyword.lower() in thermal_path.lower() for keyword in excluded_keywords):
                continue
            
            # Extract sequence name from path
            # Format: "dataset/sequence/RGB/file.jpg" or "MIRAGE_HD/VIVID/seq_001/RGB/frame_0001.jpg"
            path_parts = rgb_path.split('/')
            
            # Find sequence name (parent of RGB)
            sequence = item.get('sequence')  # Use existing if available
            if sequence is None:
                for i, part in enumerate(path_parts):
                    if part == 'RGB' and i > 0:
                        sequence = path_parts[i-1]
                        break
            
            if sequence is None:
                print(f"Warning: Could not extract sequence from {rgb_path}")
                continue
            
            # Store normalized paths
            item['rgb'] = rgb_path
            item['thermal'] = thermal_path
            item['sequence'] = sequence
            if 'dataset' not in item:
                item['dataset'] = path_parts[0] if len(path_parts) > 0 else 'unknown'
            
            filtered_data.append(item)
        
        print(f"Filtered {len(filtered_data)} samples (excluded DTUAV/visdrone)")
        print(f"  Includes: All MIRAGE_HD datasets (FLIR/M3FD/AVIID1/MSRS/VIVID/NSAVP/VTMOT/IR500K)")
        
        return filtered_data
    
    def _split_by_sequence(self) -> List[Dict]:
        """
        Split data by sequence to avoid temporal leakage.
        Uses configurable train_split_ratio (default 95:5).
        """
        # Group by sequence
        sequences = {}
        for item in self.data:
            seq = item['sequence']
            if seq not in sequences:
                sequences[seq] = []
            sequences[seq].append(item)
        
        # Sort sequences for reproducibility
        sorted_sequences = sorted(sequences.keys())
        
        # Split sequences using configurable ratio
        random.seed(42)  # Reproducible split
        
        # Ensure 'unknown' sequence (largest) goes to training
        if 'unknown' in sorted_sequences:
            sorted_sequences.remove('unknown')
            random.shuffle(sorted_sequences)
            # Put 'unknown' at the beginning so it goes to training
            sorted_sequences = ['unknown'] + sorted_sequences
        else:
            random.shuffle(sorted_sequences)
        
        split_idx = int(len(sorted_sequences) * self.train_split_ratio)
        train_sequences = set(sorted_sequences[:split_idx])
        val_sequences = set(sorted_sequences[split_idx:])
        
        print(f"Split: {len(train_sequences)} train sequences, {len(val_sequences)} val sequences")
        
        # Filter data based on split
        if self.split == "train":
            split_data = [item for item in self.data if item['sequence'] in train_sequences]
        else:  # val
            split_data = [item for item in self.data if item['sequence'] in val_sequences]
        
        if len(split_data) == 0:
            raise ValueError(f"No data found for split '{self.split}'! Check paths and filters.")
        
        return split_data
    
    def __len__(self):
        return len(self.data)
    
    def _find_image_path(self, relative_path: str) -> str:
        """Try to find image in any of the configured root folders"""
        for folder in self.image_folders:
            full_path = os.path.join(folder, relative_path)
            if os.path.exists(full_path):
                return full_path
        # If not found, return path with first folder (will error on load)
        return os.path.join(self.image_folders[0], relative_path)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load RGB image (try all root folders)
        rgb_path = self._find_image_path(item['rgb'])
        rgb_image = Image.open(rgb_path).convert('RGB')
        
        # Load thermal image (try all root folders)
        thermal_path = self._find_image_path(item['thermal'])
        thermal_image = Image.open(thermal_path).convert('RGB')  # Thermal saved as RGB
        
        # Random flip (if enabled)
        if random.random() < self.flip_p:
            rgb_image = rgb_image.transpose(Image.FLIP_LEFT_RIGHT)
            thermal_image = thermal_image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Keep original PIL image for LLaVA (it will apply its own preprocessing)
        # LLaVA's CLIP processor expects high-res and will resize to 336x336 or 224x224
        rgb_pil = rgb_image.copy()
        
        # Apply transforms for diffusion (resize to 256x256 + normalize to [-1, 1])
        rgb_tensor = self.transform(rgb_image)
        thermal_tensor = self.transform(thermal_image)
        
        # CLIP prompt (simple, random from templates)
        clip_prompt = random.choice(self.thermal_templates)
        
        # LLaVA prompt (detailed, from JSON)
        llava_prompt = None
        
        # Handle new JSON format with direct llava_prompt field
        if 'llava_prompt' in item:
            llava_prompt = item['llava_prompt']
        # Handle old JSON format with conversations
        elif 'conversations' in item and len(item['conversations']) > 0:
            for conv in item['conversations']:
                if conv.get('from') == 'human':
                    llava_prompt = conv.get('value', '')
                    # Remove <image> token (LLaVA extractor handles this)
                    llava_prompt = llava_prompt.replace('<image>', '').strip()
                    break
        
        # Fallback if no prompt
        if llava_prompt is None or llava_prompt == '':
            llava_prompt = "How would this RGB scene appear in long-wave thermal infrared spectrum"
        
        return {
            'rgb': rgb_tensor,              # Normalized tensor for diffusion [-1, 1]
            'rgb_pil': rgb_pil,              # PIL image for LLaVA [0, 255]
            'thermal': thermal_tensor,       # Normalized tensor for diffusion [-1, 1]
            'clip_prompt': clip_prompt,
            'llava_prompt': llava_prompt,
            'sequence': item['sequence'],
            'dataset': item['dataset'],
            'id': item.get('id', f"{item['sequence']}_{idx}"),
        }
    
    def get_diverse_val_samples(self, num_samples=4, seed=None):
        """
        Get diverse validation samples from different sequences/datasets.
        
        Args:
            num_samples: Number of samples to return
            seed: Random seed for reproducibility (None for random)
        
        Returns:
            List of diverse validation samples
        """
        if self.split != "val":
            raise ValueError("This method only works for validation split")
        
        # Group by dataset and sequence for diversity
        by_dataset = {}
        for item in self.data:
            dataset = item['dataset']
            sequence = item['sequence']
            key = f"{dataset}_{sequence}"
            
            if dataset not in by_dataset:
                by_dataset[dataset] = {}
            if sequence not in by_dataset[dataset]:
                by_dataset[dataset][sequence] = []
            
            by_dataset[dataset][sequence].append(item)
        
        # Get diverse samples
        if seed is not None:
            random.seed(seed)
        
        diverse_samples = []
        datasets = list(by_dataset.keys())
        sequences_per_dataset = {d: list(s.keys()) for d, s in by_dataset.items()}
        
        # Try to get samples from different datasets/sequences
        for i in range(num_samples):
            # Cycle through datasets
            dataset_idx = i % len(datasets)
            dataset = datasets[dataset_idx]
            
            # Get sequences for this dataset
            sequences = sequences_per_dataset[dataset]
            if not sequences:
                continue
                
            # Pick a random sequence from this dataset
            sequence = random.choice(sequences)
            
            # Pick a random sample from this sequence
            sample = random.choice(by_dataset[dataset][sequence])
            diverse_samples.append(sample)
        
        # Process samples through __getitem__ to get proper tensors
        processed_samples = []
        for sample_data in diverse_samples:
            # Find the index of this sample in the dataset
            sample_idx = None
            for i, item in enumerate(self.data):
                if (item['rgb'] == sample_data['rgb'] and 
                    item['thermal'] == sample_data['thermal'] and
                    item['sequence'] == sample_data['sequence']):
                    sample_idx = i
                    break
            
            if sample_idx is not None:
                processed_sample = self[sample_idx]  # Process through __getitem__
                processed_samples.append(processed_sample)
        
        return processed_samples


def custom_collate_fn(batch):
    """
    Custom collate function to handle PIL images.
    
    - Tensors (rgb, thermal): stack into batch tensor
    - PIL images (rgb_pil): keep as list (LLaVA will process each individually)
    - Strings (prompts, ids): keep as list
    """
    collated = {}
    
    # Stack tensors
    collated['rgb'] = torch.stack([item['rgb'] for item in batch])
    collated['thermal'] = torch.stack([item['thermal'] for item in batch])
    
    # Keep PIL images as list (don't stack)
    collated['rgb_pil'] = [item['rgb_pil'] for item in batch]
    
    # Keep strings as list
    collated['clip_prompt'] = [item['clip_prompt'] for item in batch]
    collated['llava_prompt'] = [item['llava_prompt'] for item in batch]
    
    # Keep metadata as list
    collated['sequence'] = [item['sequence'] for item in batch]
    collated['dataset'] = [item['dataset'] for item in batch]
    collated['id'] = [item['id'] for item in batch]
    
    return collated


def create_dataloaders(
    json_path: str,
    image_folder,  # str or List[str] - single folder or multiple root folders
    batch_size: int = 8,
    num_workers: int = 4,
    thermal_templates_path: str = "llava/data/thermal_templates.txt",
    size: int = 256,
    train_split_ratio: float = 0.98,  # Default 98:2 train:val split
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        json_path: Path to combined training JSON
        image_folder: Root folder(s) for images - str or List[str]
                     Examples: "MIRAGE_HD" or ["MIRAGE_HD", "../MIRAGE_cvpr"]
        batch_size: Batch size
        num_workers: Number of dataloader workers
        thermal_templates_path: Path to thermal prompt templates
        size: Image size (default: 256)
        train_split_ratio: Train/val split ratio (default: 0.98 for 98:2)
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = FrozenVLMIP2PDataset(
        json_path=json_path,
        image_folder=image_folder,
        split='train',
        thermal_templates_path=thermal_templates_path,
        size=size,
        flip_p=0.0,  # No flip for now
        train_split_ratio=train_split_ratio,
    )
    
    val_dataset = FrozenVLMIP2PDataset(
        json_path=json_path,
        image_folder=image_folder,
        split='val',
        thermal_templates_path=thermal_templates_path,
        size=size,
        flip_p=0.0,
        train_split_ratio=train_split_ratio,
    )
    
    # Use DistributedSampler if in distributed context
    use_dist = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_dist else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_dist else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False if use_dist else True,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # For stable batch size in DeepSpeed
        collate_fn=custom_collate_fn,  # Handle PIL images
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn,  # Handle PIL images
    )
    
    return train_loader, val_loader


def test_dataset():
    """Test dataset loading"""
    print("Testing FrozenVLMIP2PDataset...")
    
    json_path = "data/llava_miragehd_train_all.json"
    image_folder = "MIRAGE_HD"
    
    # Create dataset
    dataset = FrozenVLMIP2PDataset(
        json_path=json_path,
        image_folder=image_folder,
        split='train',
        size=256,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"RGB shape: {sample['rgb'].shape}")
    print(f"Thermal shape: {sample['thermal'].shape}")
    print(f"CLIP prompt: {sample['clip_prompt']}")
    print(f"LLaVA prompt: {sample['llava_prompt'][:100]}...")
    print(f"Sequence: {sample['sequence']}")
    print(f"Dataset: {sample['dataset']}")
    
    # Test dataloader
    train_loader, val_loader = create_dataloaders(
        json_path=json_path,
        image_folder=image_folder,
        batch_size=4,
        num_workers=0,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"Batch RGB shape: {batch['rgb'].shape}")
    print(f"Batch thermal shape: {batch['thermal'].shape}")
    print(f"Batch CLIP prompts: {batch['clip_prompt']}")
    
    print("✓ Dataset test passed!")


if __name__ == "__main__":
    test_dataset()


