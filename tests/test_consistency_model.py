"""The two consistency measures against the real model.

These are the tests that would catch a metric which runs but measures nothing --
an embedding that returns constants, or a rubric that answers yes to everything.
Marked ``slow``: they need the weights, and each rubric question costs about 20
seconds on CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from januscribe.consistency import embed_images, score_attributes, score_embedding
from januscribe.subjects import Subject

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[1]
FOX_A = REPO_ROOT / "docs" / "m1" / "fox_seed42.png"
FOX_B = REPO_ROOT / "docs" / "m1" / "fox_seed43.png"
DOGE = REPO_ROOT / "assets" / "doge.png"


@pytest.fixture(scope="module")
def images() -> list[Path]:
    for path in (FOX_A, FOX_B, DOGE):
        if not path.exists():
            pytest.skip(f"missing test image {path}")
    return [FOX_A, FOX_B, DOGE]


def test_embeddings_are_normalised_and_the_right_shape(bundle, images) -> None:
    emb = embed_images(bundle, images)
    assert emb.shape == (3, 1024), "SigLIP-L patch tokens pool to 1024 dims"
    assert emb.dtype == torch.float32
    assert torch.allclose(emb.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_embedding_of_empty_input_is_empty(bundle) -> None:
    assert embed_images(bundle, []).numel() == 0


def test_identical_images_score_exactly_one(bundle, images) -> None:
    emb = embed_images(bundle, [images[0], images[0]])
    assert float(emb[0] @ emb[1]) == pytest.approx(1.0, abs=1e-4)


def test_embedding_discriminates_category(bundle, images) -> None:
    """Two different foxes must sit closer together than either does to a dog.

    This is the sanity check that the metric measures *something*. It is also
    the evidence for the compressed-dynamic-range warning in the M2 report: the
    fox-fox number is high despite the two foxes being visibly different
    individuals, so a raw cosine means nothing without the calibration scale.
    """
    emb = embed_images(bundle, images)
    fox_fox = float(emb[0] @ emb[1])
    fox_dog_a = float(emb[0] @ emb[2])
    fox_dog_b = float(emb[1] @ emb[2])

    assert fox_fox > fox_dog_a
    assert fox_fox > fox_dog_b
    assert fox_fox - max(fox_dog_a, fox_dog_b) > 0.15, "metric has no usable dynamic range"


def test_score_embedding_matches_manual_cosine(bundle, images) -> None:
    refs = embed_images(bundle, [images[1], images[2]])
    score = score_embedding(bundle, images[0], refs)
    query = embed_images(bundle, [images[0]])
    expected = [float(x) for x in (query @ refs.T).squeeze(0)]
    assert score.per_reference == pytest.approx(expected, abs=1e-5)
    assert score.best == pytest.approx(max(expected), abs=1e-5)
    assert score.worst == pytest.approx(min(expected), abs=1e-5)


def test_rubric_distinguishes_present_from_absent(bundle) -> None:
    """The rubric must not simply agree with whatever it is asked.

    The M1 fox genuinely has a navy scarf and genuinely has no goggles, so a
    working rubric answers differently to those two questions.
    """
    if not FOX_A.exists():
        pytest.skip(f"missing {FOX_A}")

    subject = Subject(
        id="probe",
        noun="fox",
        canonical_description="a red fox in a navy blue scarf",
        base_seed=1,
        attributes=[
            "a red fox",
            "a navy blue scarf",
            "round brass goggles",
            "a bicycle",
        ],
    )
    score = score_attributes(bundle, FOX_A, subject)
    verdicts = {v.attribute: v.verdict for v in score.verdicts}

    assert verdicts["a red fox"] == "yes"
    assert verdicts["a navy blue scarf"] == "yes"
    assert verdicts["round brass goggles"] == "no"
    assert verdicts["a bicycle"] == "no"
    assert 0.0 < score.score < 1.0
    assert all(v.raw_answer for v in score.verdicts), "raw answers must be retained"
