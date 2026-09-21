"""Plan to finished document.

The one place that joins every other module: planner builds the plan, the
consistency machinery scores each image, the retry loop decides whether to
resample, and the assembler renders the result.

Like the baseline runner, builds are **resumable** -- each section's image and
score are written as produced and reused on restart. A document with ten
illustrations is over an hour of CPU here, so a crash must not cost the run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from januscribe.assemble import Document, RenderedSection
from januscribe.baseline import PromptStrategy, Tier0Strategy, reference_embeddings
from januscribe.config import GenerationConfig, Settings, UnderstandConfig
from januscribe.logging import get_logger
from januscribe.model import ModelBundle
from januscribe.planner import DocumentPlan, Section
from januscribe.retry import Attempt, RetryOutcome, RetryPolicy, generate_with_retry
from januscribe.subjects import Subject, SubjectRegistry

log = get_logger(__name__)


@dataclass
class BuildPaths:
    """Where a build keeps its intermediate work."""

    root: Path

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def outcomes(self) -> Path:
        return self.root / "outcomes"

    def image(self, index: int) -> Path:
        return self.images / f"section_{index:02d}.png"

    def outcome(self, index: int) -> Path:
        return self.outcomes / f"section_{index:02d}.json"

    def ensure(self) -> None:
        self.images.mkdir(parents=True, exist_ok=True)
        self.outcomes.mkdir(parents=True, exist_ok=True)


def _outcome_from_dict(data: dict, image: Image.Image) -> RetryOutcome:
    """Rebuild a cached outcome, keeping the attempt history intact."""
    from januscribe.baseline import _report_from_dict
    from januscribe.generate import GeneratedImage

    report = _report_from_dict(data["report"])
    generated = GeneratedImage(
        image=image,
        tokens=torch.zeros(0, dtype=torch.long),  # not needed once rendered
        seed=data["chosen_seed"],
        prompt=data["report"].get("meta", {}).get("prompt", ""),
        cfg_weight=0.0,
        temperature=0.0,
    )
    return RetryOutcome(
        image=image,
        generated=generated,
        report=report,
        attempts=[
            Attempt(
                seed=a["seed"],
                rubric_score=a["rubric_score"],
                embedding_mean=a["embedding_mean"],
                accepted=a["accepted"],
                seconds=a.get("seconds", 0.0),
            )
            for a in data.get("attempts", [])
        ],
        low_confidence=data.get("low_confidence", False),
    )


def prompt_for_section(
    section: Section, subjects: dict[str, Subject], strategy: PromptStrategy
) -> tuple[str, Subject, int]:
    """Build the image prompt for a section via the active strategy.

    Routing through the same ``PromptStrategy`` the eval harness uses is what
    makes a built document and a measured baseline comparable -- Tier 0 prose,
    a learned soft token, or a LoRA all plug in here unchanged.
    """
    spec = section.image
    if spec is None:
        raise ValueError("section has no image spec")
    if not spec.subject_ids:
        raise ValueError(f"image spec for {section.heading!r} names no subject")

    subject = subjects[spec.primary_subject]
    prompt = strategy.prompt(subject, spec.scene_text())

    # Extra subjects are appended by canonical description: only the primary
    # subject gets the strategy's treatment, because a soft token is trained
    # per subject and combining two in one prompt is untested.
    for extra_id in spec.subject_ids[1:]:
        prompt = f"{prompt}, with {subjects[extra_id].canonical_description}"

    return prompt, subject, strategy.seed(subject, spec.seed_index)


def build_document(
    bundle: ModelBundle,
    plan: DocumentPlan,
    registry: SubjectRegistry,
    out_dir: str | Path,
    strategy: PromptStrategy | None = None,
    policy: RetryPolicy | None = None,
    gen_cfg: GenerationConfig | None = None,
    und_cfg: UnderstandConfig | None = None,
    force: bool = False,
) -> Document:
    """Generate and score every illustration in a plan."""
    strategy = strategy or Tier0Strategy()
    policy = policy or RetryPolicy()
    paths = BuildPaths(Path(out_dir) / "work")
    paths.ensure()

    subjects = {sid: registry.get(sid) for sid in plan.subject_ids}
    ref_emb = reference_embeddings(bundle, list(subjects.values()), registry.root)

    log.info(
        "build_start", title=plan.title, sections=len(plan.sections),
        images=plan.n_images, strategy=strategy.name,
        max_attempts=policy.max_attempts, out=str(out_dir),
    )
    t0 = time.perf_counter()

    rendered: list[RenderedSection] = []
    for index, section in enumerate(plan.sections):
        if section.image is None:
            rendered.append(RenderedSection(section=section))
            continue

        image_path = paths.image(index)
        outcome_path = paths.outcome(index)

        if image_path.exists() and outcome_path.exists() and not force:
            image = Image.open(image_path).convert("RGB")
            outcome = _outcome_from_dict(
                json.loads(outcome_path.read_text(encoding="utf-8")), image
            )
            log.info("section_reused", index=index, heading=section.heading)
            rendered.append(
                RenderedSection(section=section, image=image, outcome=outcome)
            )
            continue

        prompt, subject, seed = prompt_for_section(section, subjects, strategy)
        outcome = generate_with_retry(
            bundle, prompt, subject, ref_emb[subject.id], seed=seed,
            policy=policy, gen_cfg=gen_cfg, und_cfg=und_cfg,
            scene=section.image.scene_text(),
        )
        outcome.image.save(image_path)
        outcome_path.write_text(json.dumps(outcome.as_dict(), indent=2), encoding="utf-8")
        log.info(
            "section_done", index=index, of=len(plan.sections),
            attempts=outcome.n_attempts, low_confidence=outcome.low_confidence,
            rubric=round(outcome.report.rubric.score, 3),
        )
        rendered.append(
            RenderedSection(section=section, image=outcome.image, outcome=outcome)
        )

    elapsed = time.perf_counter() - t0
    document = Document(
        plan=plan,
        sections=rendered,
        meta={
            "model": bundle.describe(),
            "strategy": strategy.name,
            "retry_policy": policy.model_dump(),
            "seconds_total": round(elapsed, 1),
        },
    )
    log.info(
        "build_done", seconds=round(elapsed, 1),
        low_confidence=document.n_low_confidence, sections=len(rendered),
    )
    return document


def build_from_config(
    bundle: ModelBundle,
    config: dict,
    registry: SubjectRegistry,
    scenes: Sequence[str],
    out_dir: str | Path,
    settings: Settings,
    force: bool = False,
) -> tuple[Document, dict[str, Path | None]]:
    """Run a whole build from a parsed document config, then write the outputs."""
    from januscribe.assemble import write_all
    from januscribe.planner import build_planner, resolve_subjects

    subject_ids = list(config["subjects"])
    subjects = resolve_subjects(registry, subject_ids)
    planner = build_planner(config.get("planner", "template"), bundle, scenes)

    plan = planner.plan(
        topic=config["topic"],
        subjects=subjects,
        n_sections=int(config.get("sections", 4)),
        title=config.get("title"),
    )

    policy = RetryPolicy(**config.get("retry", {}))
    strategy = _strategy_from_config(config.get("strategy", "tier0"))

    document = build_document(
        bundle, plan, registry, out_dir,
        strategy=strategy, policy=policy,
        gen_cfg=settings.generation, und_cfg=settings.understand, force=force,
    )
    outputs = write_all(
        document, out_dir,
        stem=config.get("stem", "document"),
        debug=bool(config.get("debug", False)),
    )
    return document, outputs


def _strategy_from_config(name: str) -> PromptStrategy:
    from januscribe.baseline import Tier0Strategy, Tier2Strategy

    if name == "tier0":
        return Tier0Strategy()
    if name == "tier2":
        return Tier2Strategy()
    raise ValueError(
        f"unknown strategy {name!r}; available: tier0, tier2. "
        "tier2 needs its soft tokens applied first (januscribe learn)."
    )
