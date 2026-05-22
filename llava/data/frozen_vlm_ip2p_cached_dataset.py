import os
import json
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image


class FrozenVLMIP2PCachedDataset(Dataset):
    """
    Dataset that loads RGB/TIR pairs and cached LLaVA hidden states.

    Expected layout per sample (VIVID-only for now):
      - rgb path:   .../MIRAGE_HD/train/VIVID/<seq>/RGB/xxxx.png
      - tir path:   .../MIRAGE_HD/train/VIVID/<seq>/TIR/xxxx.png
      - cache path: .../MIRAGE_HD/train/VIVID/<seq>/LLaVa_cache/xxxx.pt  (tensor [L, 4096])

    Notes:
      - We assume fixed image preprocessing for diffusion (256x256, normalized to [-1, 1]).
      - Cached hidden states can be variable-length L; we return per-item tensors and let collate_fn pad.
      - Horizontal flips must be disabled to keep caches aligned with RGBs (flip_p=0.0).
    """

    def __init__(
        self,
        pairs_json: str,
        image_roots: List[str],
        split: str = "train",
        size: int = 256,
        flip_p: float = 0.0,
        dataset_filter: Optional[str] = "VIVID",
    ) -> None:
        super().__init__()

        self.image_roots = image_roots
        self.split = split
        self.flip_p = flip_p

        # transforms for diffusion tensors
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        with open(pairs_json, 'r') as f:
            data = json.load(f)

        # Filter to split by folder heuristic and dataset name (VIVID only for now)
        def in_split(path: str) -> bool:
            return ("/train/" in path) if split == "train" else ("/val/" in path)

        items: List[Dict] = []
        for it in data:
            # Exclude FLIR entirely (train and val)
            if it.get('dataset') == 'FLIR':
                continue
            if dataset_filter and it.get('dataset') != dataset_filter:
                continue
            if not in_split(it.get('rgb', '')):
                continue
            items.append(it)

        if len(items) == 0:
            raise ValueError(f"No items found for split={split}, dataset={dataset_filter}")

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def _resolve(self, rel_path: str) -> str:
        # Try each root; return first that exists
        for root in self.image_roots:
            full = os.path.join(root, rel_path)
            if os.path.exists(full):
                return full
        # fallback to first root even if missing (better error message later)
        return os.path.join(self.image_roots[0], rel_path)

    @staticmethod
    def _cache_path_from_rgb(rgb_path: str) -> str:
        # .../<SEQ>/RGB/xxxx.png -> .../<SEQ>/LLaVa_cache/xxxx.pt
        seq_dir, last = os.path.split(os.path.dirname(rgb_path))  # seq_dir/.../RGB
        cache_dir = os.path.join(seq_dir, 'LLaVa_cache')
        base = os.path.splitext(os.path.basename(rgb_path))[0] + '.pt'
        return os.path.join(cache_dir, base)

    def __getitem__(self, idx: int) -> Dict:
        it = self.items[idx]

        rgb_path = self._resolve(it['rgb'])
        tir_path = self._resolve(it['thermal'])
        cache_rel = self._cache_path_from_rgb(it['rgb'])
        cache_path = self._resolve(cache_rel)

        # Load images
        rgb_img = Image.open(rgb_path).convert('RGB')
        tir_img = Image.open(tir_path).convert('RGB')

        # Optional safety: if size mismatch, resize TIR to RGB size
        if rgb_img.size != tir_img.size:
            tir_img = tir_img.resize(rgb_img.size, Image.Resampling.BILINEAR)

        # Keep original PIL for potential debugging; training uses tensors
        if self.flip_p > 0.0:
            # no flip for cached setup; preserve alignment
            pass

        rgb_tensor = self.transform(rgb_img)
        tir_tensor = self.transform(tir_img)

        # Load cached hidden states [L, 4096] (dtype can be fp32/bf16/fp16)
        hs = torch.load(cache_path, map_location='cpu')
        if not isinstance(hs, torch.Tensor):
            hs = torch.tensor(hs)
        
        # Safety check: detect NaN values in cached LLaVA features
        if torch.isnan(hs).any():
            print(f"WARNING: NaN detected in cached LLaVA features: {cache_path}")
            print(f"  RGB: {rgb_path}")
            print(f"  TIR: {tir_path}")
            print(f"  Shape: {hs.shape}, dtype: {hs.dtype}")
            print(f"  NaN count: {torch.isnan(hs).sum().item()}")
            
            # Option 1: Replace NaN with zeros (conservative)
            hs = torch.where(torch.isnan(hs), torch.zeros_like(hs), hs)
            print(f"  -> Replaced NaN with zeros")
            
            # Option 2: Skip this sample entirely (uncomment to use)
            # print(f"  -> Skipping this sample")
            # return self._create_dummy_sample()
        
        # Additional safety: check for infinite values
        if torch.isinf(hs).any():
            print(f"WARNING: Inf detected in cached LLaVA features: {cache_path}")
            print(f"  Inf count: {torch.isinf(hs).sum().item()}")
            hs = torch.where(torch.isinf(hs), torch.zeros_like(hs), hs)
            print(f"  -> Replaced Inf with zeros")
        
        # Check for extremely large values that might cause instability
        max_val = hs.abs().max().item()
        if max_val > 100.0:  # Arbitrary threshold
            print(f"WARNING: Large values in cached LLaVA features: {cache_path}")
            print(f"  Max absolute value: {max_val}")
            # Optionally clamp extreme values
            hs = torch.clamp(hs, min=-50.0, max=50.0)
            print(f"  -> Clamped to [-50, 50]")

        sample = {
            'rgb': rgb_tensor,
            'thermal': tir_tensor,
            'vlm_hidden': hs,  # [L, 4096], variable L
            'sequence': it.get('sequence', 'unknown'),
            'dataset': it.get('dataset', 'unknown'),
            'id': it.get('id', os.path.basename(rgb_path)),
            'rgb_path': rgb_path,
            'tir_path': tir_path,
            'cache_path': cache_path,
        }
        return sample


def pad_collate_cached(batch: List[Dict]) -> Dict:
    """Pad variable-length vlm_hidden to max L in batch and build mask.

    Returns keys:
      - rgb: [B, 3, H, W]
      - thermal: [B, 3, H, W]
      - vlm_hidden: [B, L_max, 4096]
      - vlm_mask: [B, L_max] (1 for real, 0 for pad)
      - ids, sequence, dataset, paths
    """
    B = len(batch)
    rgb = torch.stack([b['rgb'] for b in batch], dim=0)
    thermal = torch.stack([b['thermal'] for b in batch], dim=0)

    lengths = [b['vlm_hidden'].shape[0] for b in batch]
    L_max = max(lengths)
    D = batch[0]['vlm_hidden'].shape[1]

    vlm = torch.zeros((B, L_max, D), dtype=batch[0]['vlm_hidden'].dtype)
    mask = torch.zeros((B, L_max), dtype=torch.bool)
    for i, b in enumerate(batch):
        L = b['vlm_hidden'].shape[0]
        vlm[i, :L] = b['vlm_hidden']
        mask[i, :L] = True

    out = {
        'rgb': rgb,
        'thermal': thermal,
        'vlm_hidden': vlm,
        'vlm_mask': mask,
        'id': [b['id'] for b in batch],
        'sequence': [b['sequence'] for b in batch],
        'dataset': [b['dataset'] for b in batch],
        'rgb_path': [b['rgb_path'] for b in batch],
        'tir_path': [b['tir_path'] for b in batch],
        'cache_path': [b['cache_path'] for b in batch],
    }
    return out


def create_cached_dataloaders(
    pairs_json: str,
    image_roots: List[str],
    batch_size: int = 16,
    num_workers: int = 8,
    size: int = 256,
    dataset_filter: Optional[str] = "VIVID",
) -> Tuple[DataLoader, DataLoader]:
    train_ds = FrozenVLMIP2PCachedDataset(
        pairs_json=pairs_json,
        image_roots=image_roots,
        split="train",
        size=size,
        flip_p=0.0,
        dataset_filter=dataset_filter,
    )
    val_ds = FrozenVLMIP2PCachedDataset(
        pairs_json=pairs_json,
        image_roots=image_roots,
        split="val",
        size=size,
        flip_p=0.0,
        dataset_filter=dataset_filter,
    )

    # Use DistributedSampler when running with DDP/torchrun
    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    # Shuffle val to get diverse sequences across ranks
    val_sampler = DistributedSampler(val_ds, shuffle=True) if use_ddp else None

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate_cached,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False if val_sampler is not None else True,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate_cached,
    )
    return train_dl, val_dl

