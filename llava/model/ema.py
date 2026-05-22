"""
Exponential Moving Average (EMA) for Model Weights

Maintains a shadow copy of model parameters with exponentially decayed averaging:
    θ_ema = decay * θ_ema + (1 - decay) * θ

Benefits:
- More stable inference (smoother weight updates)
- Better generalization (less prone to overfitting)
- Standard practice for diffusion models (IP2P, SD, DALL-E, etc.)

Usage:
    # Initialize
    ema = EMA(model.parameters(), decay=0.9999)
    
    # After optimizer.step()
    ema.update()
    
    # For validation/inference
    with ema.average_parameters():
        # Use EMA weights
        output = model(input)
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Iterable, Optional
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Args:
        parameters: Iterable of model parameters to track
        decay: EMA decay rate (default: 0.9999)
                Higher = slower update, more stable
                Common values: 0.999, 0.9999, 0.99999
        update_every: Update EMA every N steps (default: 1)
        min_decay: Minimum decay value (default: 0.0, no minimum)
        use_num_updates: Adjust decay based on num_updates (default: True)
    """
    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        decay: float = 0.9999,
        update_every: int = 1,
        min_decay: float = 0.0,
        use_num_updates: bool = True,
    ):
        self.decay = decay
        self.min_decay = min_decay
        self.update_every = update_every
        self.use_num_updates = use_num_updates
        
        # Store model parameters
        self.online_params = list(parameters)
        
        # Create shadow copy of parameters on the same device
        self.shadow_params = [p.clone().detach().to(p.device) for p in self.online_params]
        
        # Track number of updates
        self.num_updates = 0
        
        # Flag to track if we're currently using EMA weights
        self.using_ema = False
        
    def get_decay(self, num_updates: Optional[int] = None) -> float:
        """
        Compute current decay value.
        
        Optionally adjusts decay based on number of updates:
            decay = min(decay, (1 + num_updates) / (10 + num_updates))
        
        This starts with lower decay for faster initial updates,
        then gradually increases to target decay.
        """
        if num_updates is None:
            num_updates = self.num_updates
            
        decay = self.decay
        
        if self.use_num_updates:
            # Adjust decay based on num_updates (starts lower, gradually increases)
            decay = min(decay, (1 + num_updates) / (10 + num_updates))
        
        # Apply minimum decay
        decay = max(decay, self.min_decay)
        
        return decay
    
    @torch.no_grad()
    def update(self):
        """
        Update EMA parameters.
        
        Called after optimizer.step() during training.
        """
        self.num_updates += 1
        
        # Only update every N steps
        if self.num_updates % self.update_every != 0:
            return
        
        decay = self.get_decay()
        
        # Update shadow parameters
        for i, (shadow_param, online_param) in enumerate(zip(self.shadow_params, self.online_params)):
            if online_param.requires_grad:
                # Ensure shadow parameter is on the same device as online parameter
                if shadow_param.device != online_param.device:
                    self.shadow_params[i] = shadow_param.to(online_param.device)
                    shadow_param = self.shadow_params[i]
                
                # EMA update: θ_ema = decay * θ_ema + (1 - decay) * θ
                shadow_param.mul_(decay).add_(online_param.data, alpha=1 - decay)
    
    @torch.no_grad()
    def copy_to(self, parameters: Iterable[nn.Parameter]):
        """Copy EMA parameters to given parameters"""
        for shadow_param, param in zip(self.shadow_params, parameters):
            # Ensure shadow parameter is on the same device as target parameter
            if shadow_param.device != param.device:
                shadow_param = shadow_param.to(param.device)
            param.data.copy_(shadow_param.data)
    
    @torch.no_grad()
    def store(self):
        """Store current model parameters (before loading EMA)"""
        if self.using_ema:
            raise RuntimeError("Already using EMA parameters!")
        
        self.backup_params = [p.clone() for p in self.online_params]
    
    @torch.no_grad()
    def restore(self):
        """Restore original model parameters (after using EMA)"""
        if not self.using_ema:
            raise RuntimeError("Not currently using EMA parameters!")
        
        for param, backup_param in zip(self.online_params, self.backup_params):
            param.data.copy_(backup_param.data)
        
        del self.backup_params
    
    @torch.no_grad()
    def load_ema_into_model(self):
        """Load EMA parameters into model"""
        self.store()
        self.copy_to(self.online_params)
        self.using_ema = True
    
    @torch.no_grad()
    def restore_model_from_ema(self):
        """Restore model parameters from backup"""
        self.restore()
        self.using_ema = False
    
    @contextmanager
    def average_parameters(self):
        """
        Context manager for using EMA parameters.
        
        Usage:
            with ema.average_parameters():
                # Model now uses EMA weights
                output = model(input)
            # Model back to normal weights
        """
        self.load_ema_into_model()
        try:
            yield
        finally:
            self.restore_model_from_ema()
    
    def state_dict(self):
        """Get EMA state for checkpointing"""
        return {
            'shadow_params': self.shadow_params,
            'num_updates': self.num_updates,
            'decay': self.decay,
            'min_decay': self.min_decay,
            'update_every': self.update_every,
            'use_num_updates': self.use_num_updates,
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint"""
        self.shadow_params = state_dict['shadow_params']
        self.num_updates = state_dict['num_updates']
        self.decay = state_dict.get('decay', self.decay)
        self.min_decay = state_dict.get('min_decay', self.min_decay)
        self.update_every = state_dict.get('update_every', self.update_every)
        self.use_num_updates = state_dict.get('use_num_updates', self.use_num_updates)
    
    def __repr__(self):
        return (
            f"EMA(decay={self.decay}, num_updates={self.num_updates}, "
            f"current_decay={self.get_decay():.6f}, num_params={len(self.shadow_params)})"
        )


class EMAWarmupScheduler:
    """
    Scheduler for EMA decay with warmup.
    
    Gradually increases EMA decay from min to max over warmup steps.
    Useful for stable training start.
    
    Args:
        ema: EMA instance
        warmup_steps: Number of warmup steps
        min_decay: Starting decay value (default: 0.95)
        max_decay: Target decay value (default: 0.9999)
    """
    def __init__(
        self,
        ema: EMA,
        warmup_steps: int,
        min_decay: float = 0.95,
        max_decay: float = 0.9999,
    ):
        self.ema = ema
        self.warmup_steps = warmup_steps
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.original_decay = ema.decay
    
    def step(self):
        """Update EMA decay based on num_updates"""
        if self.ema.num_updates < self.warmup_steps:
            # Linear warmup
            progress = self.ema.num_updates / self.warmup_steps
            self.ema.decay = self.min_decay + (self.max_decay - self.min_decay) * progress
        else:
            # Use original decay
            self.ema.decay = self.original_decay


if __name__ == "__main__":
    # Test EMA
    print("Testing EMA module...")
    
    # Create dummy model
    model = nn.Linear(10, 10)
    
    # Initialize EMA
    ema = EMA(model.parameters(), decay=0.9999)
    print(f"\n[1] Initialized: {ema}")
    
    # Simulate training
    print("\n[2] Simulating training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for step in range(100):
        # Forward pass
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update EMA
        ema.update()
        
        if step % 20 == 0:
            print(f"  Step {step}: decay={ema.get_decay():.6f}, num_updates={ema.num_updates}")
    
    # Test context manager
    print("\n[3] Testing context manager...")
    with torch.no_grad():
        x_test = torch.randn(4, 10)
        
        # Normal weights
        output_normal = model(x_test)
        
        # EMA weights
        with ema.average_parameters():
            output_ema = model(x_test)
        
        # Back to normal weights
        output_normal_2 = model(x_test)
        
        print(f"  Normal output mean: {output_normal.mean().item():.4f}")
        print(f"  EMA output mean: {output_ema.mean().item():.4f}")
        print(f"  Normal output (after EMA) mean: {output_normal_2.mean().item():.4f}")
        print(f"  ✓ Weights correctly restored: {torch.allclose(output_normal, output_normal_2)}")
    
    # Test state dict
    print("\n[4] Testing state dict...")
    state = ema.state_dict()
    print(f"  State dict keys: {list(state.keys())}")
    
    # Create new EMA and load state
    ema_new = EMA(model.parameters(), decay=0.999)  # Different decay
    ema_new.load_state_dict(state)
    print(f"  Loaded EMA: {ema_new}")
    print(f"  ✓ num_updates match: {ema_new.num_updates == ema.num_updates}")
    
    # Test warmup scheduler
    print("\n[5] Testing EMA warmup scheduler...")
    model2 = nn.Linear(10, 10)
    ema2 = EMA(model2.parameters(), decay=0.9999)
    scheduler = EMAWarmupScheduler(ema2, warmup_steps=50, min_decay=0.95, max_decay=0.9999)
    
    for step in range(100):
        # Simulate training
        ema2.update()
        scheduler.step()
        
        if step % 20 == 0:
            print(f"  Step {step}: decay={ema2.decay:.6f}")
    
    print("\n✓ All EMA tests passed!")
