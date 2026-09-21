"""Tier 0 baseline: the number every later strategy has to beat.

Tier 0 is the dumbest thing that could work -- paste the whole canonical
description into every prompt and offset the subject's fixed seed per image.
No training, no reference images in context. Whatever it scores is the bar.

The prompt-building is already behind a strategy object rather than inlined,
because M3 has to be evaluated by *this same harness* for the comparison to mean
anything. Adding Tier 2 means adding a class here, not editing the runner.

Runs are **resumable**. On the hardware this was written on a full sweep is
hours, so every image and every score is written as it is produced and reused on
re-run unless ``force`` is set. A crash costs the current image, not the run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image

from januscribe.config import GenerationConfig, UnderstandConfig
from januscribe.consistency import (
    ConsistencyReport,
    EmbeddingScore,
    RubricScore,
    SimilarityScale,
    calibrate_similarity,
    embed_images,
    score_image,
)
from januscribe.generate import generate_images
from januscribe.logging import get_logger
from januscribe.model import ModelBundle
from januscribe.subjects import Subject, SubjectRegistry

log = get_logger(__name__)


class PromptStrategy(Protocol):
    """How a (subject, scene) pair becomes a prompt and a seed.

    M3's soft-token strategy implements this and nothing else in the harness
    changes, which is the only way Tier 0 and Tier 2 numbers are comparable.
    """

    name: str

    def prompt(self, subject: Subject, scene: str) -> str: ...

    def seed(self, subject: Subject, index: int) -> int: ...


@dataclass
class Tier0Strategy:
    """Scene text plus the full canonical description, at the subject's seed."""

    name: str = "tier0"

    def prompt(self, subject: Subject, scene: str) -> str:
        return subject.scene_prompt(scene)

    def seed(self, subject: Subject, index: int) -> int:
        return subject.scene_seed(index)


@dataclass
class Tier2Strategy:
    """Scene text plus the learned soft token, at the subject's seed.

    Deliberately does NOT include the canonical description: the whole claim of
    M3 is that one learned vector carries the subject's identity, so leaving the
    prose in would make the comparison against Tier 0 meaningless. Set
    ``with_canonical=True`` to ablate that.

    The soft token must already be installed via
    ``inversion.apply_soft_token``; this only builds the prompt.
    """

    name: str = "tier2"
    with_canonical: bool = False

    def prompt(self, subject: Subject, scene: str) -> str:
        from januscribe.inversion import token_for

        token = token_for(subject.id)
        if self.with_canonical:
            return f"{scene}. {token}, {subject.canonical_description}"
        return f"{scene}. {token}"

    def seed(self, subject: Subject, index: int) -> int:
        # Same seeds as Tier 0, so a difference in score is a difference in
        # strategy rather than a difference in sampling noise.
        return subject.scene_seed(index)


@dataclass
class BaselineResult:
    """Everything a run produced, ready to be written out."""

    strategy: str
    reports: list[ConsistencyReport]
    scale: SimilarityScale
    meta: dict = field(default_factory=dict)

    def for_subject(self, subject_id: str) -> list[ConsistencyReport]:
        return [r for r in self.reports if r.subject_id == subject_id]


def build_reference_sheet(
    bundle: ModelBundle,
    subject: Subject,
    root: Path,
    n_refs: int,
    cfg: GenerationConfig | None = None,
    force: bool = False,
) -> list[Path]:
    """Generate (or reuse) the subject's reference sheet.

    Bootstrapping references from the canonical description is not circular for
    what they are used for: the sheet *defines* the subject's intended look, and
    every tier is then measured against that same definition. Real photographs
    can be dropped into the same directory instead, named ``ref_seed*.png``.
    """
    directory = subject.reference_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = cfg or GenerationConfig(parallel_size=1)

    paths: list[Path] = []
    for i in range(n_refs):
        seed = subject.reference_seed(i)
        path = directory / f"ref_seed{seed}.png"
        if path.exists() and not force:
            log.info("reference_reused", subject=subject.id, path=str(path))
        else:
            single = GenerationConfig(**{**cfg.model_dump(), "parallel_size": 1})
            image = generate_images(
                bundle, subject.reference_prompt_text(), seed=seed, cfg=single
            )[0]
            image.save(path)
            log.info("reference_generated", subject=subject.id, seed=seed, path=str(path))
        paths.append(path)
    return paths


def reference_embeddings(
    bundle: ModelBundle, subjects: list[Subject], root: Path
) -> dict[str, torch.Tensor]:
    """SigLIP embeddings of every subject's reference sheet, keyed by subject id."""
    out: dict[str, torch.Tensor] = {}
    for subject in subjects:
        paths = subject.reference_paths(root)
        if not paths:
            raise FileNotFoundError(
                f"subject {subject.id!r} has no reference images under "
                f"{subject.reference_dir(root)}; run `januscribe build-refs` first"
            )
        out[subject.id] = embed_images(bundle, paths)
        log.info("reference_embedded", subject=subject.id, n=len(paths))
    return out


def _score_path(out_dir: Path, subject_id: str, index: int) -> Path:
    return out_dir / "scores" / f"{subject_id}_{index:02d}.json"


def _image_path(out_dir: Path, subject_id: str, index: int, seed: int) -> Path:
    return out_dir / "images" / f"{subject_id}_{index:02d}_seed{seed}.png"


def _report_from_dict(data: dict) -> ConsistencyReport:
    from januscribe.consistency import AttributeVerdict

    rubric = RubricScore(
        verdicts=[
            AttributeVerdict(
                attribute=v["attribute"], verdict=v["verdict"], raw_answer=v["raw_answer"]
            )
            for v in data["rubric"]["verdicts"]
        ]
    )
    return ConsistencyReport(
        subject_id=data["subject_id"],
        scene=data["scene"],
        seed=data["seed"],
        image_path=data.get("image_path"),
        rubric=rubric,
        embedding=EmbeddingScore(per_reference=data["embedding"]["per_reference"]),
        meta=data.get("meta", {}),
    )


def run_baseline(
    bundle: ModelBundle,
    registry: SubjectRegistry,
    subjects: list[Subject],
    scenes: list[str],
    out_dir: str | Path,
    strategy: PromptStrategy | None = None,
    n_refs: int = 4,
    gen_cfg: GenerationConfig | None = None,
    und_cfg: UnderstandConfig | None = None,
    force: bool = False,
    n_scenes_available: int | None = None,
) -> BaselineResult:
    """Run the grid of subjects x scenes and score every image on both measures."""
    strategy = strategy or Tier0Strategy()
    gen_cfg = gen_cfg or GenerationConfig(parallel_size=1)
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(parents=True, exist_ok=True)

    log.info(
        "baseline_start",
        strategy=strategy.name,
        subjects=[s.id for s in subjects],
        n_scenes=len(scenes),
        n_images=len(subjects) * len(scenes),
        n_refs=n_refs,
        out=str(out),
    )
    t0 = time.perf_counter()

    for subject in subjects:
        build_reference_sheet(bundle, subject, registry.root, n_refs, cfg=gen_cfg, force=force)
    ref_emb = reference_embeddings(bundle, subjects, registry.root)
    scale = calibrate_similarity(ref_emb)

    reports: list[ConsistencyReport] = []
    total = len(subjects) * len(scenes)
    done = 0

    for subject in subjects:
        for index, scene in enumerate(scenes):
            done += 1
            score_file = _score_path(out, subject.id, index)
            if score_file.exists() and not force:
                reports.append(_report_from_dict(json.loads(score_file.read_text("utf-8"))))
                log.info("score_reused", subject=subject.id, scene_index=index, of=total)
                continue

            seed = strategy.seed(subject, index)
            prompt = strategy.prompt(subject, scene)
            image_file = _image_path(out, subject.id, index, seed)

            if image_file.exists() and not force:
                image = Image.open(image_file).convert("RGB")
                log.info("image_reused", subject=subject.id, scene_index=index, path=str(image_file))
            else:
                single = GenerationConfig(**{**gen_cfg.model_dump(), "parallel_size": 1})
                generated = generate_images(bundle, prompt, seed=seed, cfg=single)[0]
                generated.save(image_file)
                image = generated.image
                log.info(
                    "image_generated", subject=subject.id, scene_index=index,
                    progress=f"{done}/{total}", seed=seed, path=str(image_file),
                )

            report = score_image(
                bundle, image, subject, ref_emb[subject.id],
                scene=scene, seed=seed, image_path=str(image_file), cfg=und_cfg,
            )
            report.meta = {"strategy": strategy.name, "prompt": prompt, "scene_index": index}
            score_file.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
            reports.append(report)

    elapsed = time.perf_counter() - t0
    log.info("baseline_done", seconds=round(elapsed, 1), n_reports=len(reports))

    return BaselineResult(
        strategy=strategy.name,
        reports=reports,
        scale=scale,
        meta={
            "model": bundle.describe(),
            "n_scenes": len(scenes),
            "n_scenes_available": n_scenes_available or len(scenes),
            "n_subjects": len(subjects),
            "n_refs": n_refs,
            "cfg_weight": gen_cfg.cfg_weight,
            "temperature": gen_cfg.temperature,
            "seconds_total": round(elapsed, 1),
            "scenes": scenes,
        },
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def attribute_pass_rates(
    result: BaselineResult, subject: Subject
) -> list[tuple[str, float, int, int]]:
    """Per-attribute pass rate across scenes: which details Tier 0 keeps dropping.

    The aggregate rubric score says how much survives; this says *what* does not,
    which is the part that tells you where to aim next.
    """
    reports = result.for_subject(subject.id)
    rows: list[tuple[str, float, int, int]] = []
    for attribute in subject.attributes:
        verdicts = [
            v.verdict for r in reports for v in r.rubric.verdicts if v.attribute == attribute
        ]
        n_yes = sum(1 for v in verdicts if v == "yes")
        n_unclear = sum(1 for v in verdicts if v == "unclear")
        rate = n_yes / len(verdicts) if verdicts else 0.0
        rows.append((attribute, rate, n_yes, n_unclear))
    return rows


def write_report(
    result: BaselineResult, subjects: list[Subject], out_dir: str | Path
) -> tuple[Path, Path]:
    """Write report.json (machine) and report.md (human). Returns both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "strategy": result.strategy,
        "meta": result.meta,
        "similarity_scale": result.scale.as_dict(),
        "per_subject": {},
        "images": [r.as_dict() for r in result.reports],
    }
    for subject in subjects:
        reports = result.for_subject(subject.id)
        payload["per_subject"][subject.id] = {
            "rubric_mean": round(_mean([r.rubric.score for r in reports]), 4),
            "rubric_min": round(min((r.rubric.score for r in reports), default=0.0), 4),
            "embedding_mean": round(_mean([r.embedding.mean for r in reports]), 4),
            "embedding_worst": round(min((r.embedding.worst for r in reports), default=0.0), 4),
            "n_images": len(reports),
            "n_unclear": sum(r.rubric.n_unclear for r in reports),
            "attributes": [
                {"attribute": a, "pass_rate": round(rate, 4), "n_yes": ny, "n_unclear": nu}
                for a, rate, ny, nu in attribute_pass_rates(result, subject)
            ],
        }

    json_path = out / "report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = out / "report.md"
    md_path.write_text(_render_markdown(result, subjects, payload), encoding="utf-8")
    log.info("report_written", json=str(json_path), markdown=str(md_path))
    return json_path, md_path


def _render_markdown(result: BaselineResult, subjects: list[Subject], payload: dict) -> str:
    meta = result.meta
    scale = result.scale
    lines: list[str] = [
        f"# Consistency baseline -- strategy `{result.strategy}`",
        "",
        f"- model: `{meta.get('model', {}).get('model_id', '?')}` "
        f"on {meta.get('model', {}).get('device', '?')} "
        f"({meta.get('model', {}).get('dtype', '?')})",
        f"- grid: {meta.get('n_subjects')} subjects x {meta.get('n_scenes')} scenes "
        f"= {len(result.reports)} images, {meta.get('n_refs')} reference images each",
        *(
            [
                f"- **reduced grid**: the first {meta.get('n_scenes')} of "
                f"{meta.get('n_scenes_available')} scenes. Scenes are taken as an ordered "
                "prefix, so a later full run is directly comparable to this one."
            ]
            if (meta.get("n_scenes_available") or 0) > (meta.get("n_scenes") or 0)
            else []
        ),
        f"- sampling: cfg_weight {meta.get('cfg_weight')}, temperature {meta.get('temperature')}",
        f"- wall clock: {meta.get('seconds_total')} s",
        "",
        "Two measures, reported separately on purpose. The rubric asks whether the",
        "specified details are present; the embedding asks whether it looks like the",
        "same thing. They disagree, and the disagreement is the interesting part.",
        "",
        "## Headline",
        "",
        "| subject | rubric mean | rubric worst | embedding mean | embedding worst | unclear answers |",
        "|---|---|---|---|---|---|",
    ]
    for subject in subjects:
        s = payload["per_subject"][subject.id]
        lines.append(
            f"| `{subject.id}` | {s['rubric_mean']:.3f} | {s['rubric_min']:.3f} | "
            f"{s['embedding_mean']:.4f} | {s['embedding_worst']:.4f} | {s['n_unclear']} |"
        )

    all_rubric = [r.rubric.score for r in result.reports]
    all_emb = [r.embedding.mean for r in result.reports]
    lines += [
        f"| **all** | **{_mean(all_rubric):.3f}** | {min(all_rubric, default=0):.3f} | "
        f"**{_mean(all_emb):.4f}** | {min((r.embedding.worst for r in result.reports), default=0):.4f} | "
        f"{sum(r.rubric.n_unclear for r in result.reports)} |",
        "",
        "## Similarity scale",
        "",
        "A raw cosine is uninterpretable on its own: two *different* foxes already sit",
        "near 0.98 under this embedding. These are the goalposts.",
        "",
        f"- within-subject reference sheets (practical ceiling): "
        f"**{scale.within_mean:.4f}**" if scale.within_subject else "- ceiling: n/a",
        f"- across different subjects (floor): **{scale.cross_mean:.4f}**"
        if scale.cross_mean is not None
        else "- across different subjects (floor): **n/a** -- only one subject in this run, "
        "so the metric has no calibrated floor and the numbers below are not yet interpretable",
        f"- usable separation: **{scale.separation:.4f}**"
        if scale.separation is not None
        else "- usable separation: **n/a**",
        "",
        "| pair | cosine |",
        "|---|---|",
    ]
    for k, v in scale.within_subject.items():
        lines.append(f"| within `{k}` | {v:.4f} |")
    for k, v in scale.cross_subject.items():
        lines.append(f"| across `{k}` | {v:.4f} |")

    lines += ["", "## Which attributes Tier 0 drops", ""]
    for subject in subjects:
        lines += [
            f"### `{subject.id}`",
            "",
            "| attribute | pass rate | yes | unclear |",
            "|---|---|---|---|",
        ]
        for row in payload["per_subject"][subject.id]["attributes"]:
            lines.append(
                f"| {row['attribute']} | {row['pass_rate']:.2f} | "
                f"{row['n_yes']}/{payload['per_subject'][subject.id]['n_images']} | "
                f"{row['n_unclear']} |"
            )
        lines.append("")

    lines += ["## Per image", "", "| subject | scene | seed | rubric | embedding (mean/worst) |", "|---|---|---|---|---|"]
    for r in result.reports:
        lines.append(
            f"| `{r.subject_id}` | {r.scene} | {r.seed} | "
            f"{r.rubric.n_yes}/{r.rubric.n_total} | "
            f"{r.embedding.mean:.4f} / {r.embedding.worst:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)
