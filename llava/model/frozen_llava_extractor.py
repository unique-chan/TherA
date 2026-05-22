"""
Frozen LLaVA Feature Extractor

Loads a pretrained LLaVA model (with LoRA) and extracts hidden states before the language model head.
The entire model is frozen - no gradients computed.
Hidden states serve as a consistent "codebook" for guiding InstructPix2Pix.
"""

import torch
import torch.nn as nn
import os
from typing import Optional, Tuple

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from peft import PeftModel


class FrozenLLaVAExtractor(nn.Module):
    """
    Frozen LLaVA model for extracting hidden states as conditioning signals.
    
    Workflow:
    1. Load pretrained LLaVA + LoRA weights
    2. Freeze all parameters
    3. Extract hidden states before lm_head (shape: B, L, 4096)
    4. These act as consistent semantic "codebook" for diffusion guidance
    
    Args:
        llava_base_path: Path to base LLaVA model (e.g., "llava-1.5-7b-hf")
        llava_lora_path: Path to finetuned LoRA weights (e.g., "llava-miragehd-prior-bbox")
        device: Device to load model on
        load_8bit: Whether to load in 8-bit (for memory efficiency)
        load_4bit: Whether to load in 4-bit
    """
    def __init__(
        self,
        llava_base_path: str,
        llava_lora_path: Optional[str] = None,
        device: str = "cuda",
        load_8bit: bool = False,
        load_4bit: bool = False,
        merge_lora: bool = True,
    ):
        super().__init__()
        
        self.device = device
        self.llava_base_path = llava_base_path
        self.llava_lora_path = llava_lora_path
        
        print(f"Loading frozen LLaVA from: {llava_base_path}")
        if llava_lora_path:
            print(f"  + LoRA weights from: {llava_lora_path}")
        
        # Load base model
        # IMPORTANT: Load directly on the target CUDA device to avoid multi-GPU splits
        model_name = get_model_name_from_path(llava_base_path)
        target_device_map = {"": device} if isinstance(device, str) and device.startswith("cuda") else None
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            llava_base_path,
            None,
            model_name,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            device_map=target_device_map if target_device_map is not None else device,
            device=device
        )
        
        # Load LoRA weights if provided
        if llava_lora_path:
            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(self.model, llava_lora_path)
            
            # Load non-LoRA trainables (mm_projector weights)
            non_lora_path = os.path.join(llava_lora_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_path):
                print("Loading mm_projector weights...")
                non_lora_weights = torch.load(non_lora_path, map_location='cpu')
                # Convert to fp16 if needed
                if not load_8bit and not load_4bit:
                    non_lora_weights = {k: v.to(torch.float16) for k, v in non_lora_weights.items()}
                self.model.load_state_dict(non_lora_weights, strict=False)
            else:
                print("Warning: non_lora_trainables.bin not found, mm_projector may not be updated")

            # Optionally merge LoRA to avoid runtime PEFT wrappers on vision/LLM
            if merge_lora:
                try:
                    print("Merging LoRA adapters into base model and unloading...")
                    self.model = self.model.merge_and_unload()
                except Exception as e:
                    print(f"Warning: merge_and_unload failed ({e}); continuing with PEFT wrappers.")
        
        # Freeze everything
        # Ensure model is on the target device (safety in multi-GPU runs)
        try:
            if isinstance(self.device, str) and self.device.startswith('cuda') and torch.cuda.is_available():
                # Set current device context to reduce cuBLAS init issues on some systems
                try:
                    local_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
                    torch.cuda.set_device(local_idx)
                except Exception:
                    pass
            self.model.to(self.device)
        except Exception:
            # Some models already sharded by device_map; skip explicit .to()
            pass

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        print("✓ LLaVA model loaded and frozen")
        
    def prepare_inputs(
        self,
        images,  # Either list of PIL images or torch.Tensor
        prompts: list,
        conv_mode: str = "llava_v1"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for LLaVA forward pass.
        
        Args:
            images: Either list of PIL images or image tensors (B, 3, H, W)
            prompts: List of text prompts (B,)
            conv_mode: Conversation template mode
        
        Returns:
            input_ids: Tokenized inputs (B, L)
            images: Processed images (B, 3, 336, 336)
        """
        batch_size = len(prompts)
        
        # Process images: convert PIL to tensors if needed
        if isinstance(images, list):
            # List of PIL images - process each one
            from PIL import Image
            processed_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    # Preprocess PIL image (LLaVA's CLIP processor)
                    img_tensor = self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
                    processed_images.append(img_tensor)
                else:
                    # Already a tensor
                    processed_images.append(img)
            images = torch.stack(processed_images).to(self.device)
        else:
            # Already a batched tensor
            if images.device != self.device:
                images = images.to(self.device)
        
        # Tokenize prompts with image tokens
        input_ids_list = []
        for prompt in prompts:
            # Add image token
            if self.model.config.mm_use_im_start_end:
                prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
            else:
                prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            
            # Tokenize
            input_ids = tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            )
            input_ids_list.append(input_ids)
        
        # Pad to same length
        max_len = max(ids.size(0) for ids in input_ids_list)
        input_ids_padded = []
        for ids in input_ids_list:
            padding = torch.full((max_len - ids.size(0),), self.tokenizer.pad_token_id, dtype=ids.dtype)
            input_ids_padded.append(torch.cat([ids, padding]))
        
        input_ids = torch.stack(input_ids_padded).to(self.device)
        
        return input_ids, images
    
    @torch.no_grad()
    def extract_hidden_states(
        self,
        images,  # Either list of PIL images or torch.Tensor
        prompts: list,
    ) -> torch.Tensor:
        """
        Extract hidden states from LLaVA model (before lm_head).
        
        Args:
            images: Either list of PIL images or image tensors (B, 3, H, W)
            prompts: Text prompts (B,)
        
        Returns:
            hidden_states: Hidden states before lm_head, shape (B, L, 4096)
        """
        # Prepare inputs (handles PIL->tensor conversion if needed)
        input_ids, image_tensors = self.prepare_inputs(images, prompts)
        
        # Ensure FP16 for efficiency
        if image_tensors.device.type == "cpu":
            # Avoid half on CPU (causes slow_conv2d_cpu not implemented for Half)
            pass
        else:
            if image_tensors.dtype != torch.float16:
                image_tensors = image_tensors.half()
        
        # Forward through LLaVA; use top-level model so it accepts `images`
        # hidden_states are from the base model (before lm_head)
        def _forward_with(images_tensor: torch.Tensor):
            return self.model(
                input_ids=input_ids,
                images=images_tensor,
                return_dict=True,
                output_hidden_states=True
            )

        try:
            images_fwd = image_tensors
            if images_fwd.device.type == 'cuda':
                images_fwd = images_fwd.contiguous()
            outputs = _forward_with(images_fwd)
        except RuntimeError as e:
            msg = str(e)
            # Fallback: retry in float32 on the same device for broader kernel support
            if 'GET was unable to find an engine' in msg or 'CUBLAS_STATUS_NOT_INITIALIZED' in msg:
                try:
                    vt = None
                    try:
                        vt = self.model.get_vision_tower()
                    except Exception:
                        vt = None
                    if vt is not None:
                        vt.to(device=self.device, dtype=torch.float32)
                    images_f32 = image_tensors.to(device=self.device, dtype=torch.float32).contiguous()
                    outputs = _forward_with(images_f32)
                except Exception:
                    raise
            else:
                raise
        
        # Get last hidden states (before lm_head)
        hidden_states = outputs.hidden_states[-1]  # Shape: (B, L, 4096)
        
        return hidden_states
    
    def forward(self, images, prompts: list) -> torch.Tensor:
        """
        Forward pass - extract hidden states.
        
        Args:
            images: Either list of PIL images or image tensors (B, 3, H, W)
            prompts: Text prompts (B,)
        
        Returns:
            hidden_states: Hidden states, shape (B, L, 4096)
        """
        return self.extract_hidden_states(images, prompts)


def test_extractor():
    """Test the frozen LLaVA extractor"""
    import torch
    from PIL import Image
    import requests
    from io import BytesIO
    
    print("Testing FrozenLLaVAExtractor...")
    
    # Paths (adjust as needed)
    llava_base = "checkpoints/llava-1.5-7b-hf"
    llava_lora = "checkpoints/llava-miragehd-prior-bbox"
    
    # Create extractor
    extractor = FrozenLLaVAExtractor(
        llava_base_path=llava_base,
        llava_lora_path=llava_lora,
        device="cuda"
    )
    
    # Test with dummy data
    batch_size = 2
    prompts = [
        "How would this RGB scene appear in long-wave thermal infrared spectrum",
        "Describe the thermal characteristics of this scene"
    ]
    
    # Create dummy images
    dummy_images = torch.randn(batch_size, 3, 336, 336).cuda()
    
    # Extract hidden states
    hidden_states = extractor(dummy_images, prompts)
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Expected: (batch_size, variable_length, 4096)")
    print(f"✓ Frozen LLaVA extractor test passed!")
    
    # Verify frozen
    assert not any(p.requires_grad for p in extractor.parameters()), "Model should be frozen!"
    print("✓ All parameters frozen")


if __name__ == "__main__":
    test_extractor()

