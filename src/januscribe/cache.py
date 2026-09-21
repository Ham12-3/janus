"""KV-cache construction, isolated from the sampling loop.

The reference loop seeds ``past_key_values=None`` and then feeds back whatever
the model returned. On transformers 4.45 that round-trips through the legacy
tuple-of-tuples format, which emits a deprecation warning and is scheduled for
removal. Handing the model a real ``Cache`` object on the first step keeps the
whole loop in the supported representation, and degrades to ``None`` (the
reference behaviour) on any transformers version that lacks it.
"""

from __future__ import annotations

from typing import Any

from januscribe.logging import get_logger

log = get_logger(__name__)


def new_kv_cache() -> Any:
    """Return a fresh KV cache for a generation run, or None if unsupported."""
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:  # pragma: no cover - very old/new transformers
        log.warning("dynamic_cache_unavailable", falling_back_to="legacy tuple cache")
        return None
    return DynamicCache()
