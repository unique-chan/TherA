"""
Lazy loader for FrozenLLaVAExtractor.

Reference-cache inference does not need LLaVA weights at runtime. The extractor
is imported only when on-the-fly feature extraction is requested.
"""

from __future__ import annotations

import inspect
from typing import Any


def create_frozen_llava_extractor(**kwargs: Any):
    """Construct FrozenLLaVAExtractor; imports LLaVA stack on first use."""
    from llava.model.frozen_llava_extractor import FrozenLLaVAExtractor

    sig = inspect.signature(FrozenLLaVAExtractor.__init__)
    if "merge_lora" not in sig.parameters:
        kwargs.pop("merge_lora", None)
    return FrozenLLaVAExtractor(**kwargs)
