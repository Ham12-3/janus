"""Load Janus-Pro once and hold it.

A 7B load takes minutes, so nothing in this package may load the model twice.
``get_bundle()`` returns a process-wide singleton keyed on the settings that
actually affect the weights (model id, device, dtype, attention impl).
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from januscribe.config import Settings, resolve_device, resolve_dtype
from januscribe.logging import get_logger

# Importing janus registers MultiModalityConfig/MultiModalityCausalLM with the
# transformers Auto* factories. Without it, from_pretrained cannot resolve
# model_type="multi_modality".
from janus.models import MultiModalityCausalLM, VLChatProcessor  # noqa: E402

log = get_logger(__name__)

_BUNDLE_CACHE: dict[tuple[str, str, str, str], "ModelBundle"] = {}


def flash_attn_available() -> bool:
    """True if the optional flash-attn package is importable."""
    return importlib.util.find_spec("flash_attn") is not None


def resolve_attn_implementation(requested: str, device: torch.device, dtype: torch.dtype) -> str:
    """Pick an attention implementation, warning (not failing) if flash-attn is absent.

    flash-attn is strictly optional: it only works on CUDA with fp16/bf16, and a
    missing install must degrade to sdpa/eager rather than break the run.
    """
    if requested != "auto":
        if requested == "flash_attention_2" and not flash_attn_available():
            log.warning("flash_attn_requested_but_missing", falling_back_to="sdpa")
            return "sdpa"
        return requested

    fa_usable = (
        flash_attn_available() and device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    )
    if fa_usable:
        return "flash_attention_2"
    if device.type == "cuda":
        log.warning(
            "flash_attn_not_used",
            installed=flash_attn_available(),
            reason="not installed" if not flash_attn_available() else f"dtype {dtype} unsupported",
            falling_back_to="sdpa",
        )
    return "sdpa"


@dataclass
class ModelBundle:
    """The loaded model plus everything derived from it that callers need."""

    model: MultiModalityCausalLM
    processor: VLChatProcessor
    device: torch.device
    dtype: torch.dtype
    settings: Settings

    @property
    def tokenizer(self) -> Any:
        return self.processor.tokenizer

    @property
    def image_token_size(self) -> int:
        """Size of the VQ codebook (16384 for Janus-Pro)."""
        return int(self.model.config.gen_vision_config.params.image_token_size)

    @property
    def codebook_embed_dim(self) -> int:
        """Channel count of a quantised latent (8 for Janus-Pro's VQ-16)."""
        return int(self.model.gen_vision_model.config.codebook_embed_dim)

    @property
    def vq_dtype(self) -> torch.dtype:
        return next(self.model.gen_vision_model.parameters()).dtype

    def describe(self) -> dict[str, Any]:
        """Facts worth logging next to every artefact this model produces."""
        n_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model_id": self.settings.model_id,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "vq_dtype": str(self.vq_dtype),
            "params_total": n_params,
            "image_token_size": self.image_token_size,
            "codebook_embed_dim": self.codebook_embed_dim,
            "lm_vocab_size": int(self.model.config.language_config.vocab_size),
            "lm_hidden_size": int(self.model.config.language_config.hidden_size),
        }


def load_bundle(settings: Settings) -> ModelBundle:
    """Load processor + model from scratch. Prefer ``get_bundle`` in application code."""
    device = resolve_device(settings.device)
    dtype = resolve_dtype(settings.dtype, device)
    attn_impl = resolve_attn_implementation(settings.attn_implementation, device, dtype)

    log.info(
        "loading_model", model_id=settings.model_id, device=str(device), dtype=str(dtype),
        attn_implementation=attn_impl,
    )
    t0 = time.perf_counter()

    processor: VLChatProcessor = VLChatProcessor.from_pretrained(settings.model_id)

    # The attention implementation has to be set on the *inner* LlamaConfig, because
    # MultiModalityCausalLM builds LlamaForCausalLM itself from config.language_config.
    # Passing attn_implementation= to from_pretrained would target the outer
    # MultiModalityConfig, which declares no attention support.
    config = AutoConfig.from_pretrained(settings.model_id, trust_remote_code=True)
    config.language_config._attn_implementation = attn_impl

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device=device, dtype=dtype).eval()

    # Run the VQ tokenizer in its own dtype (fp32 by default). It is ~70M params, so
    # the memory cost is small and it keeps roundtrip fidelity out of the dtype's hands.
    vq_dtype = resolve_dtype(settings.vq_dtype, device)
    if vq_dtype != dtype:
        model.gen_vision_model = model.gen_vision_model.to(dtype=vq_dtype)
        log.info("vq_dtype_override", vq_dtype=str(vq_dtype), model_dtype=str(dtype))

    bundle = ModelBundle(
        model=model, processor=processor, device=device, dtype=dtype, settings=settings
    )
    log.info("model_loaded", seconds=round(time.perf_counter() - t0, 1), **bundle.describe())
    return bundle


def get_bundle(settings: Settings) -> ModelBundle:
    """Return a cached ModelBundle for these settings, loading it on first use."""
    key = (settings.model_id, settings.device, settings.dtype, settings.vq_dtype)
    if key not in _BUNDLE_CACHE:
        _BUNDLE_CACHE[key] = load_bundle(settings)
    return _BUNDLE_CACHE[key]


def clear_bundle_cache() -> None:
    """Drop cached models (tests, or switching model id inside one process)."""
    _BUNDLE_CACHE.clear()
