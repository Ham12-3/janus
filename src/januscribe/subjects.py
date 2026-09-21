"""Subject registry.

A Subject is the thing that has to look the same across every image in a
document. It carries:

* a **canonical description** that is deliberately over-specified, so that even
  the Tier 0 strategy (paste it into every prompt) has something to grip;
* a set of **atomic binary attributes** decomposed from that description, which
  is what the M2 rubric interrogates one question at a time;
* a **fixed base seed**, offset per image index, so a whole document is
  reproducible from one integer;
* a **reference sheet** of images that later milestones score against and that
  M3 trains its soft token on.

Reference images can be imported from disk or bootstrapped by generating them
from the canonical description. Bootstrapping is not circular for the purpose it
serves: the reference sheet *defines* what the subject is supposed to look like,
and every later tier is measured against that same definition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import yaml
from pydantic import BaseModel, Field, field_validator

from januscribe.logging import get_logger

log = get_logger(__name__)

# Scene images start this far above the base seed so they never collide with the
# reference sheet's seeds, which occupy base_seed .. base_seed + n_refs.
SCENE_SEED_OFFSET = 1000


class Subject(BaseModel):
    """One character, product or person that must stay visually stable."""

    id: str
    noun: str = Field(
        description="Head noun of the subject, e.g. 'fox'. M3 initialises the "
        "soft token from the mean embedding of this word."
    )
    canonical_description: str = Field(
        description="Deliberately over-specified. Pasted into every Tier 0 prompt."
    )
    base_seed: int
    attributes: list[str] = Field(
        description="5-8 atomic, independently checkable binary attributes."
    )
    reference_prompt: str = Field(
        default="{canonical}, plain neutral background, full view, character reference sheet",
        description="Template used when bootstrapping the reference sheet.",
    )

    @field_validator("attributes")
    @classmethod
    def _sane_attribute_count(cls, v: list[str]) -> list[str]:
        if not 3 <= len(v) <= 12:
            raise ValueError(
                f"expected 3-12 atomic attributes (the brief says 5-8), got {len(v)}"
            )
        if len(set(v)) != len(v):
            raise ValueError("duplicate attributes would double-count in the rubric")
        return v

    def scene_prompt(self, scene: str) -> str:
        """The Tier 0 prompt: scene first, then the whole canonical description."""
        return f"{scene}. {self.canonical_description}"

    def reference_prompt_text(self) -> str:
        return self.reference_prompt.format(canonical=self.canonical_description)

    def reference_seed(self, index: int) -> int:
        return self.base_seed + index

    def scene_seed(self, index: int) -> int:
        return self.base_seed + SCENE_SEED_OFFSET + index

    def reference_dir(self, root: Path) -> Path:
        return Path(root) / self.id / "refs"

    def reference_paths(self, root: Path) -> list[Path]:
        """Reference images currently on disk, in stable seed order."""
        directory = self.reference_dir(root)
        if not directory.is_dir():
            return []
        return sorted(directory.glob("ref_seed*.png"), key=lambda p: p.name)


class SubjectRegistry(BaseModel):
    """All subjects plus the root directory their reference sheets live under."""

    root: Path = Path("subjects")
    subjects: list[Subject]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SubjectRegistry":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        registry = cls(**data)
        ids = [s.id for s in registry.subjects]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate subject ids in {path}: {ids}")
        return registry

    def __iter__(self) -> Iterator[Subject]:  # type: ignore[override]
        return iter(self.subjects)

    def __len__(self) -> int:
        return len(self.subjects)

    def get(self, subject_id: str) -> Subject:
        for subject in self.subjects:
            if subject.id == subject_id:
                return subject
        raise KeyError(f"unknown subject {subject_id!r}; have {[s.id for s in self.subjects]}")

    def select(self, ids: list[str] | None) -> list[Subject]:
        """Subjects named by ``ids``, or all of them when ``ids`` is None."""
        return list(self.subjects) if not ids else [self.get(i) for i in ids]


def load_scenes(path: str | Path, limit: int | None = None) -> list[str]:
    """Load the scene list, optionally truncated to the first ``limit`` entries.

    Truncation is a run-time knob rather than a smaller file on purpose: the
    reduced grid and the full grid must draw from the same ordered list, or the
    two runs are not comparable.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    scenes = data["scenes"] if isinstance(data, dict) else data
    if not isinstance(scenes, list) or not all(isinstance(s, str) for s in scenes):
        raise ValueError(f"{path} must contain a list of scene strings")
    if limit is not None:
        if limit < 1:
            raise ValueError(f"scene limit must be >= 1, got {limit}")
        scenes = scenes[:limit]
    log.debug("scenes_loaded", path=str(path), n=len(scenes))
    return scenes
