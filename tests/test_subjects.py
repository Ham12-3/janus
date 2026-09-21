"""Subject registry and Tier 0 prompt construction. No model needed."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from januscribe.subjects import SCENE_SEED_OFFSET, Subject, SubjectRegistry, load_scenes

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBJECTS_YAML = REPO_ROOT / "configs" / "subjects.yaml"
SCENES_YAML = REPO_ROOT / "configs" / "scenes.yaml"


def _subject(**overrides) -> Subject:
    base = dict(
        id="fox",
        noun="fox",
        canonical_description="a small red fox with one torn left ear",
        base_seed=11000,
        attributes=["a red fox", "a torn left ear", "a navy scarf"],
    )
    base.update(overrides)
    return Subject(**base)


def test_tier0_prompt_is_scene_then_canonical() -> None:
    """Tier 0 is exactly scene + canonical description, in that order."""
    subject = _subject()
    prompt = subject.scene_prompt("in a sunlit meadow")
    assert prompt == "in a sunlit meadow. a small red fox with one torn left ear"
    assert prompt.startswith("in a sunlit meadow")
    assert subject.canonical_description in prompt


def test_scene_seeds_never_collide_with_reference_seeds() -> None:
    """A scene image must never accidentally reproduce a reference image.

    Reference seeds occupy base_seed upward; if scene seeds started there too,
    scene 0 and reference 0 would share a seed and the baseline would be
    measuring an image against a near-copy of itself.
    """
    subject = _subject()
    refs = {subject.reference_seed(i) for i in range(64)}
    scenes = {subject.scene_seed(i) for i in range(64)}
    assert refs.isdisjoint(scenes)
    assert subject.scene_seed(0) == subject.base_seed + SCENE_SEED_OFFSET
    assert subject.reference_seed(3) == 11003


def test_attributes_must_be_distinct() -> None:
    with pytest.raises(ValueError, match="duplicate attributes"):
        _subject(attributes=["a red fox", "a red fox", "a navy scarf"])


def test_attribute_count_is_bounded() -> None:
    with pytest.raises(ValueError, match="atomic attributes"):
        _subject(attributes=["only one", "only two"])


def test_reference_paths_are_sorted_and_empty_when_absent(tmp_path) -> None:
    subject = _subject()
    assert subject.reference_paths(tmp_path) == []
    directory = subject.reference_dir(tmp_path)
    directory.mkdir(parents=True)
    for seed in (11002, 11000, 11001):
        (directory / f"ref_seed{seed}.png").write_bytes(b"")
    names = [p.name for p in subject.reference_paths(tmp_path)]
    assert names == ["ref_seed11000.png", "ref_seed11001.png", "ref_seed11002.png"]


def test_shipped_registry_loads_and_is_well_formed() -> None:
    registry = SubjectRegistry.from_yaml(SUBJECTS_YAML)
    assert len(registry) == 3
    assert {s.id for s in registry} == {"fox", "courier", "botanist"}

    seeds = [s.base_seed for s in registry]
    assert len(set(seeds)) == len(seeds), "subjects must not share a base seed"
    # Seed ranges must not overlap once the scene offset is applied.
    for subject in registry:
        others = [s for s in registry if s.id != subject.id]
        mine = {subject.scene_seed(i) for i in range(100)} | {
            subject.reference_seed(i) for i in range(100)
        }
        for other in others:
            theirs = {other.scene_seed(i) for i in range(100)} | {
                other.reference_seed(i) for i in range(100)
            }
            assert mine.isdisjoint(theirs), f"{subject.id} and {other.id} share seeds"

    for subject in registry:
        assert 5 <= len(subject.attributes) <= 8, f"{subject.id} is outside the brief's 5-8"
        assert subject.noun in subject.canonical_description.lower() or subject.id == "courier"


def test_duplicate_subject_ids_are_rejected(tmp_path) -> None:
    doc = {
        "root": "subjects",
        "subjects": [
            {
                "id": "fox",
                "noun": "fox",
                "canonical_description": "a",
                "base_seed": 1,
                "attributes": ["a", "b", "c"],
            },
            {
                "id": "fox",
                "noun": "fox",
                "canonical_description": "b",
                "base_seed": 2,
                "attributes": ["a", "b", "c"],
            },
        ],
    }
    path = tmp_path / "dup.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate subject ids"):
        SubjectRegistry.from_yaml(path)


def test_scene_limit_takes_a_prefix_of_the_full_list() -> None:
    """A reduced run must be a prefix of the full grid, not a different sample."""
    full = load_scenes(SCENES_YAML)
    assert len(full) == 20
    six = load_scenes(SCENES_YAML, limit=6)
    assert six == full[:6]
    assert load_scenes(SCENES_YAML, limit=100) == full


def test_scene_limit_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="limit must be"):
        load_scenes(SCENES_YAML, limit=0)


def test_registry_select() -> None:
    registry = SubjectRegistry.from_yaml(SUBJECTS_YAML)
    assert [s.id for s in registry.select(None)] == [s.id for s in registry]
    assert [s.id for s in registry.select(["courier"])] == ["courier"]
    with pytest.raises(KeyError, match="unknown subject"):
        registry.select(["nope"])
