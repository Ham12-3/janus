"""Scoring arithmetic, report shape and the resume path. No model needed.

Nothing here fakes a model answer in a way that would let broken scoring pass:
the tests feed in verdicts and embeddings directly and check the arithmetic and
the serialisation, which is exactly the part that has no business touching a
GPU. The measures themselves are exercised against the real model in
``test_consistency_model.py``.
"""

from __future__ import annotations

import json

import pytest
import torch

from januscribe.baseline import (
    BaselineResult,
    Tier0Strategy,
    _report_from_dict,
    attribute_pass_rates,
    write_report,
)
from januscribe.consistency import (
    AttributeVerdict,
    ConsistencyReport,
    EmbeddingScore,
    RubricScore,
    calibrate_similarity,
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


def _report(
    verdicts: list[str], sims: list[float] | None = None, scene: str = "s", seed: int = 1
):
    subject = _subject()
    return ConsistencyReport(
        subject_id=subject.id,
        scene=scene,
        seed=seed,
        image_path="x.png",
        rubric=RubricScore(
            [
                AttributeVerdict(attribute=a, verdict=v, raw_answer=v.title() + ".")
                for a, v in zip(subject.attributes, verdicts)
            ]
        ),
        embedding=EmbeddingScore(per_reference=list(sims if sims is not None else [0.9])),
    )


def test_unclear_counts_against_never_for() -> None:
    """A hedged answer must not be scored as a pass."""
    score = RubricScore(
        [
            AttributeVerdict("a", "yes", "Yes."),
            AttributeVerdict("b", "unclear", "Maybe."),
            AttributeVerdict("c", "no", "No."),
        ]
    )
    assert score.n_yes == 1
    assert score.n_unclear == 1
    assert score.score == pytest.approx(1 / 3)


def test_empty_rubric_scores_zero_not_nan() -> None:
    assert RubricScore([]).score == 0.0


def test_embedding_score_reports_spread_not_just_mean() -> None:
    score = EmbeddingScore([0.9, 0.7, 0.95])
    assert score.mean == pytest.approx(0.85)
    assert score.best == pytest.approx(0.95)
    assert score.worst == pytest.approx(0.70)
    assert EmbeddingScore([]).mean == 0.0


def test_report_keeps_both_measures_separate() -> None:
    """The serialised report must never contain a combined score."""
    data = _report(["yes", "no", "yes"], [0.9, 0.8]).as_dict()
    assert set(data) >= {"rubric", "embedding"}
    assert data["rubric"]["score"] == pytest.approx(2 / 3, abs=1e-4)  # as_dict rounds to 4dp
    assert data["embedding"]["mean"] == pytest.approx(0.85, abs=1e-4)
    flat = json.dumps(data)
    assert "overall" not in flat and "combined" not in flat


def test_raw_answers_survive_serialisation() -> None:
    """A bad parse has to stay auditable after the run finishes."""
    data = _report(["yes", "unclear", "no"], [0.5]).as_dict()
    answers = [v["raw_answer"] for v in data["rubric"]["verdicts"]]
    assert answers == ["Yes.", "Unclear.", "No."]


def test_report_roundtrips_through_json() -> None:
    """Resume reads scores back off disk; the roundtrip must be lossless."""
    original = _report(["yes", "no", "unclear"], [0.91, 0.77], scene="in a meadow", seed=12000)
    restored = _report_from_dict(json.loads(json.dumps(original.as_dict())))
    assert restored.subject_id == original.subject_id
    assert restored.scene == original.scene
    assert restored.seed == original.seed
    assert restored.rubric.score == original.rubric.score
    assert restored.rubric.n_unclear == original.rubric.n_unclear
    assert restored.embedding.per_reference == original.embedding.per_reference
    assert [v.raw_answer for v in restored.rubric.verdicts] == [
        v.raw_answer for v in original.rubric.verdicts
    ]


def test_calibration_separates_identical_from_orthogonal() -> None:
    same = torch.nn.functional.normalize(torch.ones(3, 8), dim=-1)
    other = torch.zeros(3, 8)
    other[:, 0] = 1.0
    other = torch.nn.functional.normalize(other, dim=-1)

    scale = calibrate_similarity({"a": same, "b": other})
    assert scale.within_subject["a"] == pytest.approx(1.0, abs=1e-5)
    assert scale.separation > 0.0
    assert set(scale.cross_subject) == {"a|b"}


def test_calibration_skips_single_image_sheets() -> None:
    lone = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
    pair = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    scale = calibrate_similarity({"lone": lone, "pair": pair})
    assert "lone" not in scale.within_subject, "one image has no within-sheet pair"
    assert "pair" in scale.within_subject


def test_tier0_strategy_matches_the_brief() -> None:
    subject = _subject()
    strategy = Tier0Strategy()
    assert strategy.name == "tier0"
    assert strategy.prompt(subject, "in a meadow") == subject.scene_prompt("in a meadow")
    assert strategy.seed(subject, 2) == subject.scene_seed(2)


def test_attribute_pass_rates_locate_the_failing_detail() -> None:
    """The aggregate says how much survives; this says which detail does not."""
    subject = _subject()
    result = BaselineResult(
        strategy="tier0",
        reports=[
            _report(["yes", "no", "yes"]),
            _report(["yes", "no", "no"]),
        ],
        scale=calibrate_similarity({}),
    )
    rows = {a: rate for a, rate, _, _ in attribute_pass_rates(result, subject)}
    assert rows["a red fox"] == pytest.approx(1.0)
    assert rows["a torn left ear"] == pytest.approx(0.0)
    assert rows["a navy scarf"] == pytest.approx(0.5)


def test_write_report_emits_both_files(tmp_path) -> None:
    subject = _subject()
    result = BaselineResult(
        strategy="tier0",
        reports=[_report(["yes", "no", "yes"], [0.9, 0.8], scene="in a meadow", seed=12000)],
        scale=calibrate_similarity({}),
        meta={"n_subjects": 1, "n_scenes": 1, "n_refs": 4, "model": {"model_id": "test"}},
    )
    json_path, md_path = write_report(result, [subject], tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["strategy"] == "tier0"
    assert payload["per_subject"]["fox"]["rubric_mean"] == pytest.approx(2 / 3, abs=1e-4)
    assert len(payload["per_subject"]["fox"]["attributes"]) == 3

    markdown = md_path.read_text(encoding="utf-8")
    assert "Which attributes Tier 0 drops" in markdown
    assert "Similarity scale" in markdown
    assert "in a meadow" in markdown


def test_single_subject_run_has_no_calibrated_floor() -> None:
    """One subject means no cross-subject pairs, so the floor is unknown.

    Reporting 0.0 there would read as a perfect floor and make `separation`
    equal the within-subject mean, overstating how much room the metric has.
    """
    lone = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    scale = calibrate_similarity({"fox": lone})

    assert scale.within_subject["fox"] is not None
    assert scale.cross_mean is None
    assert scale.separation is None
    assert scale.as_dict()["cross_mean"] is None
    assert scale.as_dict()["separation"] is None


def test_markdown_says_na_instead_of_a_fake_floor(tmp_path) -> None:
    from januscribe.baseline import _render_markdown

    lone = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    result = BaselineResult(
        strategy="tier0",
        reports=[_report(["yes", "no", "yes"])],
        scale=calibrate_similarity({"fox": lone}),
        meta={"n_subjects": 1, "n_scenes": 1, "n_refs": 4, "model": {"model_id": "test"}},
    )
    markdown = _render_markdown(result, [_subject()], {"per_subject": {"fox": {
        "rubric_mean": 0.0, "rubric_min": 0.0, "embedding_mean": 0.0,
        "embedding_worst": 0.0, "n_images": 1, "n_unclear": 0, "attributes": [],
    }}})
    floor_line = next(line for line in markdown.split("\n") if "floor" in line)
    assert "n/a" in floor_line
    assert "0.0000" not in floor_line
