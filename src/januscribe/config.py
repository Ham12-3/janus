"""Configuration for JanusScribe.

Everything tunable lives here. The model id is *never* hardcoded elsewhere:
``Settings.model_id`` is the single source of truth, overridable by YAML file,
environment variable (``JANUSCRIBE_MODEL_ID``) or CLI flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DeviceName = Literal["auto", "cuda", "mps", "cpu"]
DTypeName = Literal["auto", "bfloat16", "float16", "float32"]

_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class GenerationConfig(BaseModel):
    """Text-to-image sampling parameters for the generation pathway."""

    cfg_weight: float = Field(5.0, description="Classifier-free guidance weight.")
    temperature: float = Field(1.0, gt=0.0)
    parallel_size: int = Field(1, ge=1, description="Images sampled per call.")
    image_token_num_per_image: int = Field(576, ge=1)
    img_size: int = Field(384, ge=16)
    patch_size: int = Field(16, ge=1, description="VQ downsample rate.")

    @property
    def grid(self) -> int:
        """Side length of the token grid (24 for 384px / 16)."""
        return self.img_size // self.patch_size

    def validate_geometry(self) -> None:
        """Assert token count matches the image geometry. Called before sampling."""
        expected = self.grid * self.grid
        if self.image_token_num_per_image != expected:
            raise ValueError(
                f"image_token_num_per_image={self.image_token_num_per_image} does not match "
                f"img_size={self.img_size} / patch_size={self.patch_size} -> {self.grid}^2={expected}"
            )


class UnderstandConfig(BaseModel):
    """Sampling parameters for the SigLIP understanding pathway."""

    max_new_tokens: int = Field(512, ge=1)
    temperature: float = Field(0.0, ge=0.0, description="0 means greedy / deterministic.")
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    system_prompt: str = ""


class Settings(BaseSettings):
    """Top-level settings. Immutable once built; pass explicitly, do not mutate globals."""

    model_config = SettingsConfigDict(
        env_prefix="JANUSCRIBE_", extra="forbid", protected_namespaces=()
    )

    model_id: str = "deepseek-ai/Janus-Pro-1B"
    device: DeviceName = "auto"
    dtype: DTypeName = "auto"
    # The VQ tokenizer is ~70M params. Running it in fp32 costs little memory and removes
    # quantisation noise from roundtrip PSNR measurements, so it defaults to fp32.
    vq_dtype: DTypeName = "float32"
    attn_implementation: Literal["auto", "flash_attention_2", "sdpa", "eager"] = "auto"
    seed: int = 42
    output_dir: Path = Path("outputs")
    log_level: str = "INFO"

    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    understand: UnderstandConfig = Field(default_factory=UnderstandConfig)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "Settings":
        """Load settings from a YAML file, then apply keyword overrides."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


def resolve_device(name: DeviceName) -> torch.device:
    """Turn a device name into a concrete ``torch.device``, honouring 'auto'."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
    if name == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        raise RuntimeError("device='mps' requested but MPS is not available")
    return torch.device(name)


def resolve_dtype(name: DTypeName, device: torch.device) -> torch.dtype:
    """Turn a dtype name into a ``torch.dtype``, honouring 'auto' per device.

    'auto' picks bf16 on CUDA (the dtype Janus was released in), fp16 on MPS
    (bf16 support there is patchy), and fp32 on CPU (bf16 CPU matmul on consumer
    Intel is emulated and both slow and lossy).
    """
    if name != "auto":
        return _DTYPES[name]
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32
