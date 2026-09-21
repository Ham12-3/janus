"""Planner, assembler, retry policy and the eval table. No model needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from januscribe.assemble import Document, RenderedSection, image_to_data_uri, render_html
from januscribe.consistency import AttributeVerdict, ConsistencyReport, EmbeddingScore, RubricScore
from januscribe.generate import GeneratedImage
from januscribe.planner import (
    DocumentPlan,
    ImageSpec,
    Section,
    TemplatePlanner,
    build_planner,
    plan_summary,
)
from januscribe.retry import Attempt, RetryOutcome, RetryPolicy
from januscribe.subjects import Subject

SCENES = ["in a meadow", "on a rainy street", "in a library"]
EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"


def _subject(sid: str = "fox") -> Subject:
    return Subject(
        id=sid,
        noun=sid,
        canonical_description=f"a distinctive {sid}",
        base_seed=11000,
        attributes=["a red fox", "a torn ear", "a navy scarf"],
    )


def _report(n_yes: int, n: int = 4, sim: float = 0.9) -> ConsistencyReport:
    return ConsistencyReport(
        subject_id="fox",
        scene="s",
        seed=1,
        image_path=None,
        rubric=RubricScore(
            [AttributeVerdict(f"a{i}", "yes" if i < n_yes else "no", "") for i in range(n)]
        ),
        embedding=EmbeddingScore([sim]),
    )


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #


def test_template_planner_is_deterministic() -> None:
    """Same inputs, same plan -- so a build difference is images, not prose."""
    planner = TemplatePlanner(SCENES)
    assert planner.plan("foxes", [_subject()], 4).model_dump() == (
        planner.plan("foxes", [_subject()], 4).model_dump()
    )


def test_plan_gives_every_section_a_distinct_seed_index() -> None:
    plan = TemplatePlanner(SCENES).plan("foxes", [_subject()], 5)
    assert len(plan.sections) == 5
    assert plan.n_images == 5
    indices = [s.image.seed_index for s in plan.sections]
    assert indices == [0, 1, 2, 3, 4], "colliding seed indices would repeat an image"


def test_plan_cycles_subjects_and_scenes() -> None:
    subjects = [_subject("fox"), _subject("courier")]
    plan = TemplatePlanner(SCENES).plan("a tale", subjects, 4)
    assert [s.image.subject_ids[0] for s in plan.sections] == [
        "fox",
        "courier",
        "fox",
        "courier",
    ]
    assert [s.image.scene for s in plan.sections] == [
        SCENES[0],
        SCENES[1],
        SCENES[2],
        SCENES[0],
    ]


def test_plan_roundtrips_through_json(tmp_path) -> None:
    """A plan can be hand-edited and fed back, which is the real escape hatch."""
    plan = TemplatePlanner(SCENES).plan("foxes", [_subject()], 3)
    restored = DocumentPlan.load(plan.save(tmp_path / "plan.json"))
    assert restored.model_dump() == plan.model_dump()


def test_image_spec_rejects_an_empty_scene() -> None:
    with pytest.raises(ValueError, match="needs a scene"):
        ImageSpec(scene="   ", subject_ids=["fox"])


def test_scene_text_includes_framing() -> None:
    spec = ImageSpec(scene="in a meadow", framing="close-up portrait", subject_ids=["fox"])
    assert spec.scene_text() == "in a meadow, close-up portrait"


def test_unknown_planner_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown planner"):
        build_planner("gpt-9", None, SCENES)


def test_plan_summary_lists_every_section() -> None:
    summary = plan_summary(TemplatePlanner(SCENES).plan("foxes", [_subject()], 3))
    assert summary.count("\n") == 3
    assert "fox" in summary


def test_planner_rejects_degenerate_requests() -> None:
    with pytest.raises(ValueError, match="at least one scene"):
        TemplatePlanner([])
    with pytest.raises(ValueError, match="at least one subject"):
        TemplatePlanner(SCENES).plan("t", [], 2)
    with pytest.raises(ValueError, match="n_sections"):
        TemplatePlanner(SCENES).plan("t", [_subject()], 0)


# --------------------------------------------------------------------------- #
# retry policy
# --------------------------------------------------------------------------- #


def test_policy_gates_on_the_named_metric_only() -> None:
    rubric_only = RetryPolicy(metric="rubric", rubric_threshold=0.75, embedding_threshold=0.99)
    assert rubric_only.accepts(_report(3, 4, sim=0.10)) is True

    embedding_only = RetryPolicy(metric="embedding", embedding_threshold=0.85)
    assert embedding_only.accepts(_report(0, 4, sim=0.90)) is True
    assert embedding_only.accepts(_report(4, 4, sim=0.10)) is False

    both = RetryPolicy(metric="both", rubric_threshold=0.75, embedding_threshold=0.85)
    assert both.accepts(_report(4, 4, sim=0.90)) is True
    assert both.accepts(_report(4, 4, sim=0.10)) is False


def test_rank_follows_the_gating_metric() -> None:
    """Best-of-failed must mean best by the stated rule, not an invented blend."""
    by_embedding = RetryPolicy(metric="embedding")
    assert by_embedding.rank(_report(0, 4, sim=0.95)) > by_embedding.rank(
        _report(4, 4, sim=0.90)
    )
    by_rubric = RetryPolicy(metric="rubric")
    assert by_rubric.rank(_report(4, 4, sim=0.10)) > by_rubric.rank(_report(0, 4, sim=0.99))


def test_seed_stride_keeps_retries_out_of_other_ranges() -> None:
    """A retry seed must not land on another scene's or subject's seed."""
    policy = RetryPolicy()
    subject = _subject()
    retries = {
        subject.scene_seed(i) + a * policy.seed_stride
        for i in range(20)
        for a in range(1, policy.max_attempts)
    }
    base = {subject.scene_seed(i) for i in range(20)} | {
        subject.reference_seed(i) for i in range(20)
    }
    assert retries.isdisjoint(base)


# --------------------------------------------------------------------------- #
# assembler
# --------------------------------------------------------------------------- #


def _document(low_confidence: bool = False) -> Document:
    from PIL import Image

    plan = TemplatePlanner(SCENES).plan("foxes", [_subject()], 2)
    image = Image.new("RGB", (32, 32), (200, 80, 40))
    outcome = RetryOutcome(
        image=image,
        generated=GeneratedImage(
            image=image,
            tokens=torch.zeros(0),
            seed=12000,
            prompt="p",
            cfg_weight=5.0,
            temperature=1.0,
        ),
        report=_report(3),
        attempts=[Attempt(12000, 0.75, 0.9, not low_confidence, 1.0)],
        low_confidence=low_confidence,
    )
    return Document(
        plan=plan,
        sections=[RenderedSection(section=s, image=image, outcome=outcome) for s in plan.sections],
        meta={"model": {"model_id": "test"}},
    )


def test_html_is_self_contained() -> None:
    """No external asset may be referenced: one file is the deliverable."""
    html = render_html(_document())
    assert "data:image/png;base64," in html
    assert "<img" in html
    assert "http://" not in html
    assert "https://" not in html


def test_debug_mode_shows_scores_and_plain_mode_hides_them() -> None:
    assert "rubric" not in render_html(_document(), debug=False)
    debug = render_html(_document(), debug=True)
    assert "rubric" in debug
    assert "3/4" in debug


def test_low_confidence_is_visible_in_the_artefact() -> None:
    """A weak page must be flagged in the output, not only in a log."""
    assert "Low confidence" in render_html(_document(low_confidence=True))
    assert _document(low_confidence=True).n_low_confidence == 2
    assert _document(low_confidence=False).n_low_confidence == 0


def test_html_escapes_section_text() -> None:
    plan = TemplatePlanner(SCENES).plan("foxes", [_subject()], 1)
    plan.sections[0].heading = "<script>alert(1)</script>"
    html = render_html(
        Document(plan=plan, sections=[RenderedSection(section=plan.sections[0])])
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_data_uri_encodes_a_real_png() -> None:
    import base64

    from PIL import Image

    uri = image_to_data_uri(Image.new("RGB", (4, 4)))
    assert base64.b64decode(uri.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"


def test_section_without_an_image_still_renders() -> None:
    plan = DocumentPlan(
        title="t",
        topic="x",
        subject_ids=["fox"],
        sections=[Section(heading="Intro", body="Just text.")],
    )
    html = render_html(Document(plan=plan, sections=[RenderedSection(section=plan.sections[0])]))
    assert "Just text." in html
    assert "<img" not in html


# --------------------------------------------------------------------------- #
# eval table
# --------------------------------------------------------------------------- #


def _load_compare():
    sys.path.insert(0, str(EVALS_DIR))
    import compare_strategies

    return compare_strategies


def test_unavailable_strategies_appear_as_rows_not_omissions() -> None:
    """A missing row reads as 'not tried'; it must read as 'not possible'."""
    cs = _load_compare()
    table = cs.render_table(
        [
            cs.StrategySummary(name="tier0", note="baseline", available=True),
            cs.StrategySummary(
                name="tier3", note="lora", available=False, reason="peft is not installed"
            ),
        ],
        {"model_id": "m", "n_cells": 18},
    )
    assert "tier3" in table
    assert "unavailable" in table
    assert "peft is not installed" in table
    assert "Not measured" in table


def test_available_strategy_with_no_cells_does_not_crash_the_table() -> None:
    """Available but empty is a real state; it must render, not raise."""
    cs = _load_compare()
    table = cs.render_table(
        [cs.StrategySummary(name="tier0", note="baseline", available=True, cells=[])],
        {"model_id": "m", "n_cells": 0},
    )
    assert "no cells produced" in table


def test_cpu_memory_column_is_labelled_honestly() -> None:
    """On a CPU box the column is RSS, and the table must not imply VRAM."""
    cs = _load_compare()
    summary = cs.StrategySummary(
        name="tier0",
        note="baseline",
        available=True,
        cells=[cs.CellResult("tier0", "fox", 0, 1, 0.8, 0.9, 1, False, 200.0)],
        peak_memory_mb=8400.0,
        memory_kind="cpu_peak_rss_mb",
    )
    table = cs.render_table([summary], {"model_id": "m", "n_cells": 1})
    assert "peak RSS (MB)" in table
    assert "not VRAM" in table


def test_eval_set_scene_indices_are_a_prefix() -> None:
    """A larger eval run must be a superset of a smaller one."""
    import yaml

    spec = yaml.safe_load((EVALS_DIR / "eval_set.yaml").read_text(encoding="utf-8"))
    indices = spec["scene_indices"]
    assert indices == list(range(len(indices)))
    assert spec["strategies"] == ["tier0", "tier2", "tier3", "tier4"]
