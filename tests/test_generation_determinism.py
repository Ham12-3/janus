"""Seed determinism through the full generation path, against the real model.

Marked ``slow`` for a reason: on CPU each 576-token sample takes minutes. On a
CUDA box this is a few seconds. There is no cheaper honest version of this test
-- checking a truncated sample would not prove the property that matters, which
is that the whole 576-step loop is reproducible.
"""

from __future__ import annotations

import pytest
import torch

from januscribe.config import GenerationConfig
from januscribe.generate import build_generation_prompt, generate_images

pytestmark = pytest.mark.slow

PROMPT = "a red fox wearing a navy blue scarf, digital art"


def test_generation_prompt_ends_with_image_start_tag(bundle) -> None:
    """Without the trailing begin-of-image tag the model continues in text."""
    prompt = build_generation_prompt(bundle, PROMPT)
    assert prompt.endswith(bundle.processor.image_start_tag)
    assert PROMPT in prompt


@pytest.mark.timeout(3600)
def test_same_seed_reproduces_identical_tokens(bundle) -> None:
    cfg = GenerationConfig(parallel_size=1, cfg_weight=5.0, temperature=1.0)
    first = generate_images(bundle, PROMPT, seed=1234, cfg=cfg)[0]
    second = generate_images(bundle, PROMPT, seed=1234, cfg=cfg)[0]

    assert torch.equal(first.tokens, second.tokens)
    assert first.tokens.shape == (576,)
    assert first.seed == second.seed == 1234


@pytest.mark.timeout(3600)
def test_image_index_is_independent_of_batch_size(bundle) -> None:
    """Image i of a batch must equal the solo sample at seed+i.

    This is the property that makes ablation runs comparable across batch sizes.
    """
    batch = generate_images(bundle, PROMPT, seed=7000, cfg=GenerationConfig(parallel_size=2))
    solo = generate_images(bundle, PROMPT, seed=7001, cfg=GenerationConfig(parallel_size=1))[0]

    assert batch[1].seed == solo.seed == 7001
    assert torch.equal(batch[1].tokens, solo.tokens)
