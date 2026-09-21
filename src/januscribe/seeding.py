"""Single entry point for every source of randomness.

Reproducibility is load-bearing for this project: ablation numbers are only
comparable if two runs with the same seed produce the same pixels. Everything
that draws a random number must get its generator from here.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from januscribe.logging import get_logger

log = get_logger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices) from one place."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log.debug("seeded", seed=seed, deterministic=deterministic)


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    """Return a seeded generator living on ``device``.

    MPS has no generator of its own in current torch builds, so it falls back to
    a CPU generator; callers must then sample on CPU. Determinism is guaranteed
    per (seed, device, dtype) -- not across devices.
    """
    if device.type == "mps":
        return torch.Generator(device="cpu").manual_seed(seed)
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except (RuntimeError, TypeError):  # pragma: no cover - device-specific
        log.warning("generator_fallback_to_cpu", device=str(device))
        return torch.Generator(device="cpu").manual_seed(seed)


def sample_categorical(probs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Draw one index from a 1-D probability vector using an explicit generator.

    Kept separate so the generator's device and the probs' device can disagree
    (the MPS case), without that detail leaking into the sampling loop.
    """
    if probs.dim() != 1:
        raise ValueError(f"expected a 1-D probability vector, got shape {tuple(probs.shape)}")
    gen_device = generator.device.type
    if gen_device != probs.device.type:
        idx = torch.multinomial(probs.detach().to("cpu", torch.float32), 1, generator=generator)
        return idx.to(probs.device)
    return torch.multinomial(probs.to(torch.float32), 1, generator=generator)
