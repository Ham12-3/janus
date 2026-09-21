"""Seeding and sampling determinism at the mechanism level. No model needed.

The property under test is the one the reference implementation does *not*
have: a given image index must sample the same codes regardless of how many
images were requested alongside it.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from januscribe.seeding import make_generator, sample_categorical, seed_everything


def _draws(probs: torch.Tensor, seed: int, steps: int = 64) -> list[int]:
    gen = make_generator(seed, torch.device("cpu"))
    return [int(sample_categorical(probs, gen)) for _ in range(steps)]


def test_same_seed_gives_same_draws() -> None:
    probs = torch.softmax(torch.randn(16384), dim=-1)
    assert _draws(probs, 42) == _draws(probs, 42)


def test_different_seeds_diverge() -> None:
    probs = torch.softmax(torch.randn(16384), dim=-1)
    assert _draws(probs, 42) != _draws(probs, 43)


def test_row_draws_are_independent_of_batch_size() -> None:
    """Row i in a batch of 4 must draw exactly what it draws alone at seed+i.

    This is why generate.py samples row by row with per-row generators instead
    of one batched torch.multinomial call.
    """
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(4, 16384), dim=-1)
    base = 1234

    gens = [make_generator(base + i, torch.device("cpu")) for i in range(4)]
    batched = [[int(sample_categorical(probs[i], gens[i])) for i in range(4)] for _ in range(32)]
    batched_row2 = [step[2] for step in batched]

    solo_gen = make_generator(base + 2, torch.device("cpu"))
    solo_row2 = [int(sample_categorical(probs[2], solo_gen)) for _ in range(32)]

    assert batched_row2 == solo_row2


def test_seed_everything_covers_every_rng() -> None:
    seed_everything(7)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(7)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b
