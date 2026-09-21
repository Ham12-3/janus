"""Topic plus subjects to a structured document plan.

A plan is ordered sections, each with body text and an image spec naming the
scene, the camera framing, and which subjects appear. Everything downstream --
generation, scoring, assembly -- consumes the plan, so the plan is the contract.

**Planners are pluggable by design, not by retrofit.** Janus-Pro-1B is an
image model with a small language model attached; asking it to write structured
prose is asking the weakest part of the system to do the hardest text job. So
``Planner`` is a Protocol, and the default implementation is deterministic:

* ``TemplatePlanner`` composes sections from the scene list and the subject
  registry with no model at all. Reproducible, instant, and good enough to
  exercise the whole pipeline.
* ``JanusPlanner`` asks Janus itself for the prose. Available, and honestly
  labelled as the weaker option.
* Anything else implementing ``Planner`` -- a call to a stronger text model --
  drops in without touching the pipeline.

Plans serialise to JSON, so a plan produced by one planner can be edited by hand
and rendered by the pipeline unchanged. That is the real escape hatch: if no
planner is good enough, write the plan yourself.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Protocol, Sequence

import yaml
from pydantic import BaseModel, Field, field_validator

from januscribe.logging import get_logger
from januscribe.subjects import Subject, SubjectRegistry

log = get_logger(__name__)

FRAMINGS: tuple[str, ...] = (
    "wide establishing shot",
    "medium shot",
    "close-up portrait",
    "over-the-shoulder view",
    "low angle shot",
)


class ImageSpec(BaseModel):
    """Everything needed to generate and score one illustration."""

    scene: str = Field(description="What is happening and where.")
    framing: str = Field("medium shot", description="Camera framing.")
    subject_ids: list[str] = Field(
        default_factory=list, description="Subjects that appear, in order."
    )
    seed_index: int = Field(0, ge=0, description="Offset into the subject's seed range.")

    @field_validator("scene")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("an image spec needs a scene")
        return v.strip()

    def scene_text(self) -> str:
        """Scene plus framing, which is what goes into the prompt."""
        return f"{self.scene}, {self.framing}"

    @property
    def primary_subject(self) -> str | None:
        return self.subject_ids[0] if self.subject_ids else None


class Section(BaseModel):
    """One unit of the document: a heading, body text, and optionally an image."""

    heading: str
    body: str
    image: ImageSpec | None = None


class DocumentPlan(BaseModel):
    """An ordered document, ready to be generated and assembled."""

    title: str
    topic: str
    subject_ids: list[str]
    sections: list[Section]
    planner: str = "unknown"

    @property
    def image_sections(self) -> list[Section]:
        return [s for s in self.sections if s.image is not None]

    @property
    def n_images(self) -> int:
        return len(self.image_sections)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "DocumentPlan":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class Planner(Protocol):
    """Turn a topic and a cast of subjects into a document plan.

    Implement this to swap in a stronger text model. Nothing else in the
    pipeline needs to change.
    """

    name: str

    def plan(
        self, topic: str, subjects: Sequence[Subject], n_sections: int, title: str | None = None
    ) -> DocumentPlan: ...


class TemplatePlanner:
    """Deterministic planner. No model, no network, no variance.

    The default, because it makes the pipeline testable and reproducible: the
    same topic and scene list always yields the same plan, so a difference
    between two document builds is a difference in image generation rather than
    in the prose that framed it.
    """

    name = "template"

    def __init__(self, scenes: Sequence[str], body_sentences: int = 3) -> None:
        if not scenes:
            raise ValueError("TemplatePlanner needs at least one scene")
        self.scenes = list(scenes)
        self.body_sentences = body_sentences

    def plan(
        self, topic: str, subjects: Sequence[Subject], n_sections: int, title: str | None = None
    ) -> DocumentPlan:
        if not subjects:
            raise ValueError("a plan needs at least one subject")
        if n_sections < 1:
            raise ValueError(f"n_sections must be >= 1, got {n_sections}")

        sections: list[Section] = []
        for index in range(n_sections):
            scene = self.scenes[index % len(self.scenes)]
            subject = subjects[index % len(subjects)]
            framing = FRAMINGS[index % len(FRAMINGS)]
            sections.append(
                Section(
                    heading=f"{index + 1}. {scene[0].upper()}{scene[1:]}",
                    body=self._body(topic, subject, scene),
                    image=ImageSpec(
                        scene=scene,
                        framing=framing,
                        subject_ids=[subject.id],
                        seed_index=index,
                    ),
                )
            )

        plan = DocumentPlan(
            title=title or f"{topic.strip().capitalize()}",
            topic=topic,
            subject_ids=[s.id for s in subjects],
            sections=sections,
            planner=self.name,
        )
        log.info(
            "plan_built", planner=self.name, sections=len(plan.sections),
            images=plan.n_images, subjects=plan.subject_ids,
        )
        return plan

    def _body(self, topic: str, subject: Subject, scene: str) -> str:
        sentences = [
            f"This part of {topic} follows {subject.noun} {scene}.",
            f"Here {subject.noun} is shown as {subject.canonical_description}.",
            "The illustration beside this text was generated for this section alone.",
        ]
        return " ".join(sentences[: self.body_sentences])


class JanusPlanner:
    """Planner that asks Janus-Pro itself to write the prose.

    Available and honestly labelled: Janus-Pro-1B's language model is ~1.3B
    parameters trained primarily for multimodal understanding, so its prose is
    the weakest link in the pipeline. The image specs are still built
    deterministically -- only the body text comes from the model -- because a
    hallucinated scene would break generation and scoring downstream, while
    mediocre prose only reads badly.

    Falls back to the template body on any failure rather than emitting an
    empty section.
    """

    name = "janus"

    def __init__(self, bundle, scenes: Sequence[str], max_new_tokens: int = 120) -> None:
        self.bundle = bundle
        self.fallback = TemplatePlanner(scenes)
        self.scenes = list(scenes)
        self.max_new_tokens = max_new_tokens

    def plan(
        self, topic: str, subjects: Sequence[Subject], n_sections: int, title: str | None = None
    ) -> DocumentPlan:
        plan = self.fallback.plan(topic, subjects, n_sections, title=title)
        plan.planner = self.name

        for section in plan.sections:
            scene = section.image.scene if section.image else ""
            section.body = self._write_body(topic, scene) or section.body
        return plan

    def _write_body(self, topic: str, scene: str) -> str:
        from januscribe.config import UnderstandConfig
        from januscribe.understand import ask

        prompt = textwrap.dedent(
            f"""
            Write two short sentences for a page of a document about {topic}.
            The page's illustration shows: {scene}.
            Write only the prose, no heading and no list.
            """
        ).strip()
        try:
            # The understanding path is the only text interface Janus exposes; it
            # runs the vision tower over a blank image for a text-only prompt,
            # which is wasteful but harmless. See NOTES-API.md section 6.
            answer = ask(
                self.bundle,
                [_blank_image()],
                prompt,
                cfg=UnderstandConfig(max_new_tokens=self.max_new_tokens, temperature=0.0),
            )
            text = answer.text.strip()
            return text if len(text) > 20 else ""
        except Exception as exc:  # a planner must not take the build down
            log.warning("janus_planner_failed", error=str(exc), scene=scene)
            return ""


def _blank_image():
    from PIL import Image

    return Image.new("RGB", (384, 384), (255, 255, 255))


def load_document_config(path: str | Path) -> dict:
    """Load a build config: topic, subjects, sections, planner, output options."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    for required in ("topic", "subjects"):
        if required not in data:
            raise ValueError(f"{path} is missing required key {required!r}")
    return data


def build_planner(name: str, bundle, scenes: Sequence[str]) -> Planner:
    """Resolve a planner by name. Unknown names fail loudly rather than defaulting."""
    if name == "template":
        return TemplatePlanner(scenes)
    if name == "janus":
        return JanusPlanner(bundle, scenes)
    raise ValueError(
        f"unknown planner {name!r}; available: template, janus. "
        "To add a stronger text model, implement the Planner protocol and register it here."
    )


def resolve_subjects(registry: SubjectRegistry, ids: Sequence[str]) -> list[Subject]:
    return [registry.get(i) for i in ids]


def plan_summary(plan: DocumentPlan) -> str:
    """One-line-per-section summary, for logs and dry runs."""
    lines = [f"{plan.title}  ({plan.planner}, {len(plan.sections)} sections, {plan.n_images} images)"]
    for i, section in enumerate(plan.sections):
        image = section.image
        detail = f"{image.scene_text()} [{','.join(image.subject_ids)}]" if image else "(no image)"
        lines.append(f"  {i + 1}. {section.heading} -- {detail}")
    return "\n".join(lines)


def json_schema() -> str:
    """The plan schema, so a hand-written or externally generated plan can validate."""
    return json.dumps(DocumentPlan.model_json_schema(), indent=2)
