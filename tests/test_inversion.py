"""Soft-token logic that needs no model: schedule, serialisation, indexing.

The parts that genuinely need the model -- that only one parameter trains, and
that the loss actually falls -- live in ``test_inversion_model.py`` and run
against real weights. Nothing here fakes a forward pass.
"""

from __future__ import annotations

import pytest
import torch

from januscribe.baseline import Tier0Strategy, Tier2Strategy
from januscribe.inversion import (
    DEFAULT_TEMPLATES,
    InversionConfig,
    SoftToken,
    prediction_slice,
    token_for,
)
from januscribe.subjects import Subject


def _subject() -> Subject:
    return Subject(
        id="fox",
        noun="fox",
        canonical_description="a small red fox with one torn left ear",
        base_seed=11000,
        attributes=["a red fox", "a torn left ear", "a navy scarf"],
    )


def test_prediction_slice_is_offset_by_one() -> None:
    """Position T-1 predicts code 0, so the window starts one before the codes.

    An off-by-one here would shift every target by one position and still train
    to a plausible-looking loss, which is exactly why it is tested directly.
    """
    start, end = prediction_slice(n_text=42, n_codes=576)
    assert (start, end) == (41, 617)
    assert end - start == 576

    # The window must end exactly at the input length, T + N - 1.
    assert end == 42 + 576 - 1


@pytest.mark.parametrize("n_text,n_codes", [(1, 1), (5, 3), (61, 576)])
def test_prediction_slice_width_always_matches_code_count(n_text, n_codes) -> None:
    start, end = prediction_slice(n_text, n_codes)
    assert end - start == n_codes
    assert start == n_text - 1


def test_prediction_slice_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="at least one text token"):
        prediction_slice(0, 576)
    with pytest.raises(ValueError, match="at least one image code"):
        prediction_slice(42, 0)


def test_lr_warms_up_then_decays_to_lr_end() -> None:
    cfg = InversionConfig(steps=500, lr=1e-3, lr_end=1e-4, warmup_steps=50)
    assert cfg.lr_at(0) == pytest.approx(1e-3 / 50)
    assert cfg.lr_at(49) == pytest.approx(1e-3)
    assert cfg.lr_at(50) == pytest.approx(1e-3)
    assert cfg.lr_at(499) == pytest.approx(1e-4, abs=1e-6)

    after_warmup = [cfg.lr_at(s) for s in range(50, 500)]
    assert all(a >= b - 1e-12 for a, b in zip(after_warmup, after_warmup[1:]))


def test_lr_schedule_without_warmup_starts_at_peak() -> None:
    cfg = InversionConfig(steps=100, lr=1e-3, lr_end=1e-4, warmup_steps=0)
    assert cfg.lr_at(0) == pytest.approx(1e-3)


def test_templates_all_carry_the_token() -> None:
    """A template without the placeholder would train on nothing."""
    for template in DEFAULT_TEMPLATES:
        assert "{token}" in template
    assert len(set(DEFAULT_TEMPLATES)) == len(DEFAULT_TEMPLATES)
    assert len(DEFAULT_TEMPLATES) >= 3, "one pose would be memorised"


def test_token_naming_is_stable() -> None:
    assert token_for("fox") == "<sbj_fox>"
    assert token_for("courier") == "<sbj_courier>"


def test_soft_token_roundtrips_through_safetensors(tmp_path) -> None:
    vector = torch.randn(2048)
    original = SoftToken(
        subject_id="fox",
        token="<sbj_fox>",
        vector=vector,
        meta={"steps": 500, "lr": 0.001, "final_loss": 5.4321, "reference_paths": ["a.png"]},
    )
    path = original.save(tmp_path / "fox.safetensors")
    assert path.stat().st_size < 100_000, "a soft token is one vector, not a checkpoint"

    restored = SoftToken.load(path)
    assert restored.subject_id == "fox"
    assert restored.token == "<sbj_fox>"
    assert torch.allclose(restored.vector, vector)
    assert restored.meta["steps"] == 500
    assert restored.meta["reference_paths"] == ["a.png"]


def test_tier2_prompt_drops_the_canonical_description() -> None:
    """The whole claim is that one vector replaces the prose.

    Leaving the canonical description in would mean Tier 2 is Tier 0 plus a
    token, and any improvement would be unattributable.
    """
    subject = _subject()
    prompt = Tier2Strategy().prompt(subject, "in a meadow")
    assert prompt == "in a meadow. <sbj_fox>"
    assert subject.canonical_description not in prompt


def test_tier2_ablation_can_keep_the_description() -> None:
    subject = _subject()
    prompt = Tier2Strategy(with_canonical=True).prompt(subject, "in a meadow")
    assert "<sbj_fox>" in prompt
    assert subject.canonical_description in prompt


def test_tiers_share_seeds_so_the_comparison_is_controlled() -> None:
    """Same seed per scene index, so a score delta is strategy, not noise."""
    subject = _subject()
    for index in range(20):
        assert Tier0Strategy().seed(subject, index) == Tier2Strategy().seed(subject, index)
    assert Tier2Strategy().name == "tier2"
