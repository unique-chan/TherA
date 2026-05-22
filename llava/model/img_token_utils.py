"""
Utility functions for adding and managing IMG tokens for VLM-guided diffusion.
"""

import torch
import torch.nn as nn
from typing import List
from llava.constants import IMG_TOKENS


def add_img_tokens_to_tokenizer(tokenizer, img_tokens: List[str] = None, 
                                special_tokens: bool = True):
    """
    Add IMG tokens to the tokenizer (safe for resume - won't double-add).
    
    Args:
        tokenizer: HuggingFace tokenizer
        img_tokens: List of IMG tokens (default: IMG_TOKENS from constants)
        special_tokens: If True, tokens are added as special tokens (auto-stripped with skip_special_tokens=True)
                       If False, tokens are normal vocabulary (always kept in decode)
    
    Returns:
        num_added: Number of tokens added (0 if already present)
        token_ids: List of token IDs for the IMG tokens
    
    Note:
        - When decoding for debugging/metrics, use skip_special_tokens=False to see IMG tokens
        - MGIE uses special_tokens=True (default behavior)
        - Safe for resume: checks existing vocab to avoid double-adding
    """
    if img_tokens is None:
        img_tokens = IMG_TOKENS
    
    # Check for existing tokens (CRITICAL for resume!)
    existing_vocab = set(tokenizer.get_vocab().keys())
    tokens_to_add = [t for t in img_tokens if t not in existing_vocab]
    
    if not tokens_to_add:
        # All tokens already exist (likely resuming from checkpoint)
        print(f"IMG tokens already in tokenizer (resuming from checkpoint)")
        token_ids = tokenizer.convert_tokens_to_ids(img_tokens)
        print(f"  Tokens: {img_tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Vocab size: {len(tokenizer)}")
        return 0, token_ids
    
    # Add only new tokens
    num_added = tokenizer.add_tokens(tokens_to_add, special_tokens=special_tokens)
    
    # Get all token IDs (including previously existing ones)
    token_ids = tokenizer.convert_tokens_to_ids(img_tokens)
    
    print(f"Added {num_added} IMG tokens to tokenizer")
    print(f"  New tokens added: {tokens_to_add}")
    print(f"  All IMG token IDs: {token_ids}")
    print(f"  New vocab size: {len(tokenizer)}")
    print(f"  Added as: {'special tokens' if special_tokens else 'normal tokens'}")
    
    if special_tokens:
        print(f"  ⚠️  Remember: Use skip_special_tokens=False when decoding for debugging")
    
    return num_added, token_ids


def save_tokenizer_with_img_tokens(tokenizer, output_dir: str):
    """
    Save tokenizer with IMG tokens for later use.
    
    This is critical for:
    - Resuming training from checkpoint
    - Evaluation/inference
    - Preventing silent vocab mismatch errors
    
    Args:
        tokenizer: Tokenizer with IMG tokens added
        output_dir: Directory to save tokenizer
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Saved tokenizer with {len(tokenizer)} tokens to: {output_dir}")


def resize_token_embeddings_and_init(model, tokenizer, init_method: str = "mean", init_std: float = 0.02):
    """
    Resize model token embeddings and initialize new IMG token embeddings.
    
    Args:
        model: LLaVA model
        tokenizer: Tokenizer (after adding IMG tokens)
        init_method: "mean" (MGIE-style, average of existing) or "normal" (N(0, std))
        init_std: Standard deviation for normal initialization (only used if init_method="normal")
    
    Returns:
        new_token_ids: List of new token IDs
    """
    old_vocab_size = model.config.vocab_size
    new_vocab_size = len(tokenizer)
    num_new_tokens = new_vocab_size - old_vocab_size
    
    if num_new_tokens == 0:
        print("No new tokens to add")
        return []
    
    print(f"\nResizing token embeddings:")
    print(f"  Old vocab size: {old_vocab_size}")
    print(f"  New vocab size: {new_vocab_size}")
    print(f"  Adding {num_new_tokens} new tokens")
    print(f"  Initialization method: {init_method}")
    
    # Remember original dtype before resize
    original_dtype = model.get_model().embed_tokens.weight.dtype
    
    # Resize embeddings
    model.resize_token_embeddings(new_vocab_size)
    
    # CRITICAL: resize_token_embeddings may reset entire model to float32
    # This is especially important when model is in float16/bfloat16
    if model.get_model().embed_tokens.weight.dtype != original_dtype:
        print(f"  ! Warning: resize changed dtype from {original_dtype} to {model.get_model().embed_tokens.weight.dtype}")
        print(f"  ! Restoring entire model to {original_dtype}...")
        
        # Convert entire model back to original dtype
        if original_dtype == torch.float16:
            model = model.half()
        elif original_dtype == torch.bfloat16:
            model = model.bfloat16()
        elif original_dtype == torch.float32:
            model = model.float()
        
        print(f"  ✓ Model restored to {original_dtype}")
    
    # ALSO convert vision tower if it exists (resize doesn't touch it, but we need consistency)
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        vision_dtype = next(vision_tower.parameters()).dtype
        if vision_dtype != original_dtype:
            print(f"  ! Vision tower dtype mismatch: {vision_dtype} vs expected {original_dtype}")
            print(f"  ! Converting vision tower to {original_dtype}...")
            
            if original_dtype == torch.float16:
                vision_tower = vision_tower.half()
            elif original_dtype == torch.bfloat16:
                vision_tower = vision_tower.bfloat16()
            elif original_dtype == torch.float32:
                vision_tower = vision_tower.float()
            
            print(f"  ✓ Vision tower converted to {original_dtype}")
    
    # Sanity check: ensure config is updated (HF usually does this automatically)
    assert model.config.vocab_size == new_vocab_size, \
        f"Config vocab size mismatch! model.config.vocab_size={model.config.vocab_size}, expected={new_vocab_size}"
    
    # Get new token IDs (the last num_new_tokens)
    new_token_ids = list(range(old_vocab_size, new_vocab_size))
    
    # Initialize new token embeddings
    with torch.no_grad():
        embed_tokens = model.get_model().embed_tokens
        lm_head = model.lm_head
        
        if init_method == "mean":
            # MGIE-style: mean of existing embeddings (smoother start)
            # Compute mean - stays on same device and dtype as source
            embed_mean = embed_tokens.weight.data[:old_vocab_size].mean(dim=0, keepdim=True)
            lm_mean = lm_head.weight.data[:old_vocab_size].mean(dim=0, keepdim=True)
            
            # Direct assignment (mean is already on correct device from the slice)
            # Important: Don't use .to(device=...) as model may be distributed across GPUs
            for idx in new_token_ids:
                embed_tokens.weight.data[idx] = embed_mean.squeeze(0)
                lm_head.weight.data[idx] = lm_mean.squeeze(0)
            
            print(f"  ✓ Initialized embeddings as mean of existing tokens (dtype: {embed_tokens.weight.dtype})")
        else:
            # Normal initialization - create on same device/dtype as existing weights
            # Use the embedding's own device and dtype
            new_embed = torch.randn(
                len(new_token_ids), embed_tokens.weight.shape[1],
                dtype=embed_tokens.weight.dtype,
                device=embed_tokens.weight.device
            ) * init_std
            
            new_lm = torch.randn(
                len(new_token_ids), lm_head.weight.shape[1],
                dtype=lm_head.weight.dtype,
                device=lm_head.weight.device
            ) * init_std
            
            embed_tokens.weight.data[new_token_ids] = new_embed
            lm_head.weight.data[new_token_ids] = new_lm
            
            print(f"  ✓ Initialized embeddings with N(0, {init_std}) (dtype: {embed_tokens.weight.dtype})")
    
    # Final verification: check dtype consistency across model
    print(f"\n  Verifying dtype consistency across all modules...")
    dtype_consistent, _ = verify_model_dtype_consistency(model, expected_dtype=original_dtype)
    
    if not dtype_consistent:
        print(f"  ✗ WARNING: Model has dtype inconsistencies after resize!")
        print(f"  ✗ This may cause 'mat1 and mat2 must have same dtype' errors!")
        print(f"  ✗ Consider calling model.half() or model.float() to fix.")
    else:
        print(f"  ✓ All modules consistent at {original_dtype}")
    
    return new_token_ids


def setup_llava_trainable_params_mgie_style(model, train_vision_tower: bool = False, 
                                             train_mm_projector: bool = False):
    """
    Setup trainable parameters MGIE-style: train full embed_tokens and lm_head.
    This is simpler and safer than gradient masking for controlled training.
    
    Args:
        model: LLaVA model
        train_vision_tower: Whether to train vision tower (default: False)
        train_mm_projector: Whether to train mm_projector (default: False)
    
    Returns:
        trainable_param_names: List of trainable parameter names
    """
    print(f"\nSetting up MGIE-style trainable parameters:")
    
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    trainable_params = []
    
    # Train embed_tokens (full matrix)
    embed_tokens = model.get_model().embed_tokens
    embed_tokens.weight.requires_grad = True
    trainable_params.append("model.embed_tokens.weight")
    print(f"  ✓ embed_tokens: trainable (full matrix, {embed_tokens.weight.shape})")
    
    # Train lm_head (full matrix)
    lm_head = model.lm_head
    lm_head.weight.requires_grad = True
    trainable_params.append("lm_head.weight")
    print(f"  ✓ lm_head: trainable (full matrix, {lm_head.weight.shape})")
    
    # Optional: train vision tower
    if train_vision_tower:
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            for param in vision_tower.parameters():
                param.requires_grad = True
            trainable_params.append("vision_tower.*")
            print(f"  ✓ vision_tower: trainable")
    
    # Optional: train mm_projector
    if train_mm_projector:
        mm_projector = model.get_model().mm_projector
        if mm_projector is not None:
            for param in mm_projector.parameters():
                param.requires_grad = True
            trainable_params.append("mm_projector.*")
            print(f"  ✓ mm_projector: trainable")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_count:,} ({100*trainable_count/total_params:.2f}%)")
    
    return trainable_params


def setup_img_token_gradient_masks(model, tokenizer, img_tokens: List[str] = None):
    """
    [ADVANCED/OPTIONAL] Setup gradient masks to train only IMG token embeddings and LM head rows.
    
    NOTE: For most use cases, use `setup_llava_trainable_params_mgie_style()` instead.
    This function is for advanced users who want fine-grained control.
    
    This registers hooks that zero out gradients for all non-IMG token rows,
    ensuring only the IMG token embeddings and their LM head projections are trained.
    
    MGIE-style (simpler, recommended): Train full matrices with low LR
    Gradient masking (this function): Train only specific rows with normal LR
    
    Args:
        model: LLaVA model
        tokenizer: Tokenizer (after adding IMG tokens)
        img_tokens: List of IMG tokens (default: IMG_TOKENS from constants)
    
    Returns:
        img_token_ids: List of IMG token IDs
    """
    if img_tokens is None:
        img_tokens = IMG_TOKENS
    
    # Get IMG token IDs
    img_token_ids = tokenizer.convert_tokens_to_ids(img_tokens)
    
    print(f"\nSetting up gradient masks for IMG tokens:")
    print(f"  IMG token IDs: {img_token_ids}")
    
    # Get embedding and lm_head
    embed_tokens = model.get_model().embed_tokens
    lm_head = model.lm_head
    
    # Create masks on the correct device and dtype
    vocab_size = lm_head.weight.size(0)
    
    # LM head mask: (vocab_size, 1) - broadcasts over hidden_dim
    # Match dtype and device of the weights
    lm_mask = torch.zeros(vocab_size, 1, device=lm_head.weight.device, dtype=lm_head.weight.dtype)
    lm_mask[img_token_ids] = 1.0
    
    # Embedding mask: (vocab_size, 1) - broadcasts over embed_dim
    emb_mask = torch.zeros(embed_tokens.num_embeddings, 1, device=embed_tokens.weight.device, dtype=embed_tokens.weight.dtype)
    emb_mask[img_token_ids] = 1.0
    
    # Register hooks
    def lm_head_grad_mask_hook(grad):
        """Zero out gradients for non-IMG token rows in LM head"""
        # Ensure mask matches gradient dtype
        mask = lm_mask.to(dtype=grad.dtype) if lm_mask.dtype != grad.dtype else lm_mask
        return grad * mask
    
    def embed_grad_mask_hook(grad):
        """Zero out gradients for non-IMG token rows in embeddings"""
        # Ensure mask matches gradient dtype
        mask = emb_mask.to(dtype=grad.dtype) if emb_mask.dtype != grad.dtype else emb_mask
        return grad * mask
    
    lm_head.weight.register_hook(lm_head_grad_mask_hook)
    embed_tokens.weight.register_hook(embed_grad_mask_hook)
    
    print(f"  ✓ Registered gradient mask hooks")
    print(f"  ✓ Only {len(img_token_ids)} token rows will be trained")
    
    return img_token_ids


def verify_model_dtype_consistency(model, expected_dtype=None):
    """
    Verify that all model parameters have consistent dtype.
    
    This is critical after resize_token_embeddings which may reset parts to float32.
    
    Args:
        model: LLaVA model
        expected_dtype: Expected dtype (if None, use embed_tokens dtype)
    
    Returns:
        all_consistent: Boolean
        dtype_report: Dict with dtype info per module
    """
    if expected_dtype is None:
        expected_dtype = model.get_model().embed_tokens.weight.dtype
    
    dtype_report = {}
    all_consistent = True
    
    # Check key modules
    modules_to_check = {
        'embed_tokens': model.get_model().embed_tokens,
        'lm_head': model.lm_head,
    }
    
    # Add optional modules
    if hasattr(model.get_model(), 'mm_projector'):
        modules_to_check['mm_projector'] = model.get_model().mm_projector
    
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        modules_to_check['vision_tower'] = vision_tower
    
    for name, module in modules_to_check.items():
        param_dtypes = [p.dtype for p in module.parameters()]
        if param_dtypes:
            unique_dtypes = set(param_dtypes)
            dtype_report[name] = unique_dtypes
            
            if len(unique_dtypes) > 1:
                print(f"  ✗ {name}: MIXED dtypes {unique_dtypes}")
                all_consistent = False
            elif expected_dtype not in unique_dtypes:
                actual = list(unique_dtypes)[0]
                print(f"  ✗ {name}: dtype {actual}, expected {expected_dtype}")
                all_consistent = False
            else:
                print(f"  ✓ {name}: {list(unique_dtypes)[0]}")
    
    return all_consistent, dtype_report


def verify_img_token_setup(model, tokenizer, img_tokens: List[str] = None):
    """
    Comprehensive sanity check for IMG token setup before training.
    
    This checks:
    - Tokenizer vocab matches model config
    - IMG tokens exist in vocabulary
    - IMG tokens are properly initialized
    - Model is ready for training
    
    Args:
        model: LLaVA model
        tokenizer: Tokenizer
        img_tokens: List of IMG tokens (default: IMG_TOKENS from constants)
    
    Returns:
        success: Boolean indicating if all checks passed
    """
    if img_tokens is None:
        img_tokens = IMG_TOKENS
    
    print("\n" + "="*60)
    print("IMG TOKEN SETUP VERIFICATION")
    print("="*60)
    
    all_good = True
    
    # 1. Check tokenizer/model vocab size consistency
    print(f"\n1. Vocab size consistency:")
    tokenizer_size = len(tokenizer)
    model_config_size = model.config.vocab_size
    embed_size = model.get_model().embed_tokens.weight.shape[0]
    lm_head_size = model.lm_head.weight.shape[0]
    
    print(f"  Tokenizer vocab size: {tokenizer_size}")
    print(f"  Model config vocab size: {model_config_size}")
    print(f"  Embedding matrix size: {embed_size}")
    print(f"  LM head matrix size: {lm_head_size}")
    
    if tokenizer_size != model_config_size:
        print(f"  ✗ Tokenizer/config mismatch!")
        all_good = False
    elif embed_size != tokenizer_size:
        print(f"  ✗ Embedding size mismatch!")
        all_good = False
    elif lm_head_size != tokenizer_size:
        print(f"  ✗ LM head size mismatch!")
        all_good = False
    else:
        print(f"  ✓ All sizes match ({tokenizer_size})")
    
    # 2. Check IMG tokens exist
    print(f"\n2. IMG tokens in vocabulary:")
    token_ids = tokenizer.convert_tokens_to_ids(img_tokens)
    unk_id = tokenizer.unk_token_id
    
    missing_tokens = []
    for token, token_id in zip(img_tokens, token_ids):
        if token_id == unk_id:
            print(f"  ✗ {token}: MISSING (maps to UNK)")
            missing_tokens.append(token)
            all_good = False
        else:
            print(f"  ✓ {token}: ID={token_id}")
    
    if missing_tokens:
        print(f"  ✗ Missing tokens: {missing_tokens}")
    else:
        print(f"  ✓ All {len(img_tokens)} IMG tokens present")
    
    # 3. Check initialization
    print(f"\n3. Embedding initialization:")
    with torch.no_grad():
        embed_weights = model.get_model().embed_tokens.weight[token_ids]
        lm_weights = model.lm_head.weight[token_ids]
        
        embed_norms = embed_weights.norm(dim=1)
        lm_norms = lm_weights.norm(dim=1)
        
        print(f"  Embedding norms: mean={embed_norms.mean():.4f}, min={embed_norms.min():.4f}, max={embed_norms.max():.4f}")
        print(f"  LM head norms:   mean={lm_norms.mean():.4f}, min={lm_norms.min():.4f}, max={lm_norms.max():.4f}")
        
        if embed_norms.mean() < 0.001 or lm_norms.mean() < 0.001:
            print(f"  ⚠ Warning: Embeddings may not be initialized (norms very small)")
            all_good = False
        else:
            print(f"  ✓ Embeddings appear initialized")
    
    # 4. Check dtype consistency
    print(f"\n4. Dtype consistency:")
    dtype_consistent, dtype_report = verify_model_dtype_consistency(model)
    
    if not dtype_consistent:
        print(f"  ✗ Model has dtype inconsistencies!")
        all_good = False
    else:
        print(f"  ✓ All modules have consistent dtype")
    
    # 5. Final sanity checks
    print(f"\n5. Final checks:")
    
    # Check that model config is consistent
    if model.config.vocab_size == tokenizer_size:
        print(f"  ✓ Config vocab_size consistent with tokenizer")
    else:
        print(f"  ✗ Config vocab_size inconsistent!")
        all_good = False
    
    # Remind about critical training setup
    print(f"\n6. Training setup reminders:")
    print(f"  ⚠️  Save tokenizer: tokenizer.save_pretrained(output_dir)")
    print(f"  ⚠️  Decode with: skip_special_tokens=False (for debugging)")
    print(f"  ⚠️  Ensure IMG tokens in labels != IGNORE_INDEX")
    print(f"  ⚠️  Use weight_decay=0.0 for embed_tokens and lm_head")
    print(f"  ⚠️  Use low LR (5e-6 to 1e-5) for language matrices")
    
    print("\n" + "="*60)
    if all_good:
        print("✅ ALL CHECKS PASSED - Ready for training!")
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before training!")
    print("="*60 + "\n")
    
    return all_good


def verify_img_tokens(model, tokenizer, img_tokens: List[str] = None):
    """
    Verify that IMG tokens exist and are properly set up.
    
    Args:
        model: LLaVA model
        tokenizer: Tokenizer
        img_tokens: List of IMG tokens (default: IMG_TOKENS from constants)
    
    Returns:
        success: Boolean indicating if all checks passed
    """
    if img_tokens is None:
        img_tokens = IMG_TOKENS
    
    print("\n" + "="*60)
    print("VERIFYING IMG TOKENS")
    print("="*60)
    
    all_good = True
    
    # Check tokenizer
    print(f"\n1. Checking tokenizer:")
    token_ids = tokenizer.convert_tokens_to_ids(img_tokens)
    unk_id = tokenizer.unk_token_id
    
    for token, token_id in zip(img_tokens, token_ids):
        if token_id == unk_id:
            print(f"  ✗ {token}: MISSING (maps to UNK)")
            all_good = False
        else:
            print(f"  ✓ {token}: ID={token_id}")
    
    # Check model vocab size
    print(f"\n2. Checking model:")
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    print(f"  Model vocab size: {model.config.vocab_size}")
    print(f"  Embedding weight shape: {model.get_model().embed_tokens.weight.shape}")
    print(f"  LM head weight shape: {model.lm_head.weight.shape}")
    
    if model.config.vocab_size != len(tokenizer):
        print(f"  ✗ Model vocab size mismatch!")
        all_good = False
    else:
        print(f"  ✓ Vocab sizes match")
    
    # Check embedding norms (should be non-zero if initialized)
    print(f"\n3. Checking embedding initialization:")
    with torch.no_grad():
        embed_weights = model.get_model().embed_tokens.weight[token_ids]
        lm_weights = model.lm_head.weight[token_ids]
        
        embed_norms = embed_weights.norm(dim=1)
        lm_norms = lm_weights.norm(dim=1)
        
        print(f"  Embedding norms: mean={embed_norms.mean():.4f}, min={embed_norms.min():.4f}, max={embed_norms.max():.4f}")
        print(f"  LM head norms:   mean={lm_norms.mean():.4f}, min={lm_norms.min():.4f}, max={lm_norms.max():.4f}")
        
        if embed_norms.mean() < 0.001 or lm_norms.mean() < 0.001:
            print(f"  ⚠ Warning: Embeddings may not be initialized (norms very small)")
        else:
            print(f"  ✓ Embeddings appear initialized")
    
    print("\n" + "="*60)
    if all_good:
        print("✓ ALL CHECKS PASSED")
    else:
        print("✗ SOME CHECKS FAILED")
    print("="*60 + "\n")
    
    return all_good


def get_img_token_ids(tokenizer, img_tokens: List[str] = None) -> List[int]:
    """
    Get IMG token IDs from tokenizer.
    
    Args:
        tokenizer: HuggingFace tokenizer
        img_tokens: List of IMG tokens (default: IMG_TOKENS from constants)
    
    Returns:
        token_ids: List of IMG token IDs
    """
    if img_tokens is None:
        img_tokens = IMG_TOKENS
    
    return tokenizer.convert_tokens_to_ids(img_tokens)


def extract_img_token_positions(labels: torch.Tensor, img_token_ids: List[int]) -> torch.Tensor:
    """
    Extract positions of IMG tokens in the labels tensor.
    
    Args:
        labels: (B, L) tensor of token IDs
        img_token_ids: List of IMG token IDs to search for
    
    Returns:
        positions: (B, K) tensor of positions for each IMG token
                   -1 indicates the token was not found
    """
    batch_size = labels.size(0)
    num_img_tokens = len(img_token_ids)
    
    # Initialize with -1 (not found)
    positions = torch.full((batch_size, num_img_tokens), -1, dtype=torch.long, device=labels.device)
    
    # For each batch
    for b in range(batch_size):
        # For each IMG token
        for k, token_id in enumerate(img_token_ids):
            # Find positions where this token appears
            matches = (labels[b] == token_id).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                # Take the first occurrence
                positions[b, k] = matches[0]
    
    return positions


def create_img_token_mask(positions: torch.Tensor) -> torch.Tensor:
    """
    Create a binary mask from IMG token positions.
    
    Args:
        positions: (B, K) tensor from extract_img_token_positions (-1 = not found)
    
    Returns:
        mask: (B, K) binary tensor (1=valid token, 0=missing)
    """
    # positions >= 0 means token was found
    mask = (positions >= 0).to(dtype=torch.float32)
    return mask


def gather_img_hidden_states(hidden_states: torch.Tensor, 
                             positions: torch.Tensor,
                             fill_value: float = 0.0) -> torch.Tensor:
    """
    Gather hidden states at IMG token positions.
    
    Args:
        hidden_states: (B, L, D) tensor of hidden states
        positions: (B, K) tensor of positions (from extract_img_token_positions)
        fill_value: Value to use for missing tokens (default: 0.0)
    
    Returns:
        img_hiddens: (B, K, D) tensor of IMG token hidden states
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    num_img_tokens = positions.size(1)
    
    # Initialize output with fill_value
    img_hiddens = torch.full(
        (batch_size, num_img_tokens, hidden_dim),
        fill_value,
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    
    # Gather hidden states at valid positions
    for b in range(batch_size):
        for k in range(num_img_tokens):
            pos = positions[b, k].item()
            if pos >= 0 and pos < seq_len:
                img_hiddens[b, k] = hidden_states[b, pos]
    
    return img_hiddens


def extract_img_token_hiddens(hidden_states: torch.Tensor,
                              labels: torch.Tensor,
                              img_token_ids: List[int],
                              fill_value: float = 0.0,
                              return_mask: bool = False):
    """
    Combined function to extract IMG token hidden states from labels and hidden states.
    
    Args:
        hidden_states: (B, L, D) tensor from model output (last layer)
        labels: (B, L) tensor of token IDs
        img_token_ids: List of IMG token IDs
        fill_value: Value to use for missing tokens (default: 0.0)
        return_mask: If True, also return a validity mask (default: False)
    
    Returns:
        img_hiddens: (B, K, D) tensor of IMG token hidden states
        mask: (B, K) binary mask (1=valid, 0=missing) - only if return_mask=True
    """
    positions = extract_img_token_positions(labels, img_token_ids)
    img_hiddens = gather_img_hidden_states(hidden_states, positions, fill_value)
    
    if return_mask:
        mask = create_img_token_mask(positions)
        return img_hiddens, mask
    
    return img_hiddens

