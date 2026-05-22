"""
Progress Tracking Utilities with tqdm

Provides:
- Epoch-level progress bars
- Step-level metrics display
- ETA calculation
- Formatted logging
"""

import time
from typing import Dict, Optional
from tqdm import tqdm


class TrainingProgressTracker:
    """
    Manages training progress with tqdm bars and ETA.
    """
    def __init__(
        self,
        total_epochs: int,
        steps_per_epoch: int,
        log_interval: int = 100,
        val_interval: int = 2000,
    ):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.log_interval = log_interval
        self.val_interval = val_interval
        
        # Progress bars
        self.epoch_bar = None
        self.step_bar = None
        
        # Timing
        self.epoch_start_time = None
        self.training_start_time = time.time()
        
        # Metrics
        self.current_epoch = 0
        self.global_step = 0
    
    def start_epoch(self, epoch: int):
        """Start a new epoch"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
        # Create epoch-level progress bar
        if self.epoch_bar is not None:
            self.epoch_bar.close()
        
        self.epoch_bar = tqdm(
            total=self.steps_per_epoch,
            desc=f"Epoch {epoch}/{self.total_epochs}",
            position=0,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    
    def update_step(
        self,
        metrics: Dict[str, float],
        global_step: int,
    ):
        """
        Update progress for a single step.
        
        Args:
            metrics: Dict of metrics to display (loss, lr, etc.)
            global_step: Global training step
        """
        self.global_step = global_step
        
        # Format metrics for display
        metric_str = {}
        for key, value in metrics.items():
            if 'loss' in key.lower():
                metric_str[key] = f"{value:.4f}"
            elif 'lr' in key.lower():
                metric_str[key] = f"{value:.2e}"
            elif 'scale' in key.lower():
                metric_str[key] = f"{value:.2f}"
            else:
                metric_str[key] = f"{value:.4f}"
        
        # Add global step
        metric_str['step'] = f"{global_step}"
        
        # Update epoch bar
        if self.epoch_bar is not None:
            self.epoch_bar.set_postfix(metric_str)
            self.epoch_bar.update(1)
    
    def end_epoch(self):
        """End current epoch"""
        if self.epoch_bar is not None:
            self.epoch_bar.close()
            self.epoch_bar = None
        
        # Print epoch summary
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.training_start_time
        
        print(f"\n{'='*60}")
        print(f"Epoch {self.current_epoch} complete!")
        print(f"  Epoch time: {self.format_time(epoch_time)}")
        print(f"  Total training time: {self.format_time(total_time)}")
        print(f"  Global step: {self.global_step}")
        print(f"{'='*60}\n")
    
    def close(self):
        """Close all progress bars"""
        if self.epoch_bar is not None:
            self.epoch_bar.close()
        
        total_time = time.time() - self.training_start_time
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"  Total time: {self.format_time(total_time)}")
        print(f"  Total steps: {self.global_step}")
        print(f"  Average steps/sec: {self.global_step / total_time:.2f}")
        print(f"{'='*60}\n")
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds into human-readable string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def get_eta(self, current_step: int, total_steps: int) -> str:
        """Calculate ETA for remaining steps"""
        if current_step == 0:
            return "calculating..."
        
        elapsed = time.time() - self.training_start_time
        steps_per_sec = current_step / elapsed
        remaining_steps = total_steps - current_step
        eta_seconds = remaining_steps / steps_per_sec
        
        return self.format_time(eta_seconds)


def format_metrics_for_wandb(metrics: Dict[str, float], prefix: str = "") -> Dict[str, float]:
    """
    Format metrics for W&B logging.
    
    Args:
        metrics: Raw metrics dict
        prefix: Prefix to add to keys (e.g., "train/", "val/")
    
    Returns:
        formatted_metrics: Formatted for W&B
    """
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted[f"{prefix}{key}"] = value
    return formatted


def print_training_header(config: Dict):
    """Print formatted training configuration"""
    print("\n" + "="*80)
    print("FROZEN VLM-IP2P TRAINING".center(80))
    print("="*80)
    print("\nConfiguration:")
    print("-" * 80)
    
    # Group configs
    groups = {
        "Data": ["data_path", "image_folder", "batch_size", "num_workers"],
        "Model": ["resolution", "num_vlm_tokens", "resampler_depth"],
        "Training": ["num_epochs", "lr_perceiver", "lr_adapter", "weight_decay"],
        "CFG": ["cfg_dropout", "image_guidance_scale", "text_guidance_scale", "vlm_guidance_scale"],
        "Validation": ["val_interval", "num_val_samples", "val_scheduler", "num_inference_steps"],
        "Output": ["output_dir", "save_interval", "log_interval"],
    }
    
    for group_name, keys in groups.items():
        print(f"\n{group_name}:")
        for key in keys:
            if key in config:
                value = config[key]
                print(f"  {key:30s}: {value}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Test progress tracker
    print("Testing progress tracker...")
    
    tracker = TrainingProgressTracker(
        total_epochs=3,
        steps_per_epoch=100,
        log_interval=10,
    )
    
    # Simulate training
    import random
    
    for epoch in range(1, 4):
        tracker.start_epoch(epoch)
        
        for step in range(100):
            global_step = (epoch - 1) * 100 + step
            
            # Simulate metrics
            metrics = {
                'loss': random.uniform(0.1, 0.5),
                'lr': 1e-4 * (0.99 ** global_step),
                'ema_decay': 0.9999,
            }
            
            tracker.update_step(metrics, global_step)
            time.sleep(0.01)  # Simulate training time
        
        tracker.end_epoch()
    
    tracker.close()
    
    print("\n✓ Progress tracker test complete!")


