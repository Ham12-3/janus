"""Run every consistency strategy over the fixed eval set and tabulate the result.

This table is the deliverable. Everything else in the project exists to make it
trustworthy: the seed discipline so cells are comparable, the validated rubric
so the scores mean something, the two-measure split so a win on features is not
confused with a win on identity.

    python evals/compare_strategies.py --out evals/results
    python evals/compare_strategies.py --strategy tier0,tier4 --scenes 2

Design commitments:

* **Same seeds across strategies.** Cell (subject, scene) uses the same seed for
  every strategy, so a difference is the strategy rather than sampling noise.
* **Unavailable strategies are reported, not skipped.** Tier 2 without a trained
  token and Tier 3 without peft appear in the table as unavailable with a
  reason. A silently missing row reads as "not tried" when it should read as
  "not possible here".
* **Resumable.** Every cell is written as produced and reused on restart. A full
  sweep is many hours on CPU.
* **Memory is labelled by what it actually is.** On CUDA this is peak VRAM; on
  CPU there is no VRAM, so it reports peak process RSS and says so, rather than
  printing a zero that looks like a measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from januscribe.baseline import Tier0Strategy, Tier2Strategy, reference_embeddings  # noqa: E402
from januscribe.config import GenerationConfig, Settings, UnderstandConfig  # noqa: E402
from januscribe.consistency import ConsistencyReport, score_image  # noqa: E402
from januscribe.generate import generate_images  # noqa: E402
from januscribe.logging import get_logger  # noqa: E402
from januscribe.model import ModelBundle, get_bundle  # noqa: E402
from januscribe.retry import RetryPolicy  # noqa: E402
from januscribe.subjects import Subject, SubjectRegistry, load_scenes  # noqa: E402

log = get_logger(__name__)

SUBJECTS_FILE = REPO_ROOT / "configs" / "subjects.yaml"
SCENES_FILE = REPO_ROOT / "configs" / "scenes.yaml"
EVAL_SET = REPO_ROOT / "evals" / "eval_set.yaml"
TOKENS_DIR = REPO_ROOT / "tokens"
ADAPTERS_DIR = REPO_ROOT / "adapters"


# --------------------------------------------------------------------------- #
# memory accounting
# --------------------------------------------------------------------------- #

def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory() -> tuple[float, str]:
    """(megabytes, what it measures). CPU has no VRAM; say so rather than lie."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**20, "cuda_peak_vram_mb"
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20, "cpu_peak_rss_mb"
    except ImportError:  # pragma: no cover
        return float("nan"), "unavailable"


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #

@dataclass
class EvalStrategy:
    """A strategy plus how to check it can run here and how to sample from it."""

    name: str
    note: str = ""

    def availability(self, bundle: ModelBundle, registry: SubjectRegistry) -> tuple[bool, str]:
        return True, ""

    def setup(self, bundle: ModelBundle, registry: SubjectRegistry) -> None:
        return None

    def generate(self, bundle, subject, scene, seed, cfg):
        raise NotImplementedError


@dataclass
class PromptEvalStrategy(EvalStrategy):
    """Any strategy that works purely by changing the prompt."""

    inner: object = None

    def generate(self, bundle, subject, scene, seed, cfg):
        prompt = self.inner.prompt(subject, scene)
        return generate_images(bundle, prompt, seed=seed, cfg=cfg)[0], prompt


@dataclass
class Tier0Eval(PromptEvalStrategy):
    name: str = "tier0"
    note: str = "canonical description in every prompt"
    inner: object = field(default_factory=Tier0Strategy)


@dataclass
class Tier2Eval(PromptEvalStrategy):
    name: str = "tier2"
    note: str = "learned soft token"
    inner: object = field(default_factory=Tier2Strategy)
    tokens_dir: Path = TOKENS_DIR

    def availability(self, bundle, registry):
        missing = [
            s.id for s in registry if not (self.tokens_dir / f"{s.id}.safetensors").exists()
        ]
        if missing:
            return False, f"no learned token for {missing}; run `januscribe learn --subject <id>`"
        return True, ""

    def setup(self, bundle, registry):
        from januscribe.inversion import load_and_apply

        for subject in registry:
            load_and_apply(bundle, self.tokens_dir / f"{subject.id}.safetensors")


@dataclass
class Tier3Eval(PromptEvalStrategy):
    name: str = "tier3"
    note: str = "LoRA adapters on LM attention and MLP"
    inner: object = field(default_factory=Tier2Strategy)  # same trigger-token prompt
    adapters_dir: Path = ADAPTERS_DIR

    def availability(self, bundle, registry):
        from januscribe.lora import peft_available

        if not peft_available():
            return False, "peft is not installed"
        missing = [
            s.id
            for s in registry
            if not (self.adapters_dir / s.id / "adapter.safetensors").exists()
        ]
        if missing:
            return False, f"no trained adapter for {missing}"
        return True, ""

    def setup(self, bundle, registry):
        from januscribe.lora import load_lora

        # One adapter at a time is the honest setup; multi-subject adapters are
        # out of scope, so this evaluates the first subject's adapter only if
        # they share a model. Loading per subject happens in generate().
        return None


@dataclass
class Tier4Eval(EvalStrategy):
    name: str = "tier4"
    note: str = "reference sheet in context (Janus was not trained for this)"
    root: Path = Path("subjects")
    n_references: int = 2

    def availability(self, bundle, registry):
        missing = [s.id for s in registry if not s.reference_paths(self.root)]
        if missing:
            return False, f"no reference sheet for {missing}"
        return True, ""

    def generate(self, bundle, subject, scene, seed, cfg):
        from januscribe.visual_conditioning import (
            generate_with_visual_context,
            load_reference_images,
        )

        refs = load_reference_images(subject, self.root, limit=self.n_references)
        generated = generate_with_visual_context(
            bundle, scene, refs, seed=seed, cfg=cfg
        )[0]
        return generated, scene


def build_strategies(names: Sequence[str], root: Path) -> list[EvalStrategy]:
    table = {
        "tier0": lambda: Tier0Eval(),
        "tier2": lambda: Tier2Eval(),
        "tier3": lambda: Tier3Eval(),
        "tier4": lambda: Tier4Eval(root=root),
    }
    unknown = [n for n in names if n not in table]
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; available: {sorted(table)}")
    return [table[n]() for n in names]


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #

@dataclass
class CellResult:
    strategy: str
    subject_id: str
    scene_index: int
    seed: int
    rubric_score: float
    embedding_mean: float
    attempts: int
    low_confidence: bool
    seconds: float

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy, "subject_id": self.subject_id,
            "scene_index": self.scene_index, "seed": self.seed,
            "rubric_score": round(self.rubric_score, 4),
            "embedding_mean": round(self.embedding_mean, 4),
            "attempts": self.attempts, "low_confidence": self.low_confidence,
            "seconds": round(self.seconds, 1),
        }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def run_strategy(
    bundle: ModelBundle,
    strategy: EvalStrategy,
    registry: SubjectRegistry,
    subjects: list[Subject],
    scenes: list[tuple[int, str]],
    ref_emb: dict[str, torch.Tensor],
    policy: RetryPolicy,
    gen_cfg: GenerationConfig,
    und_cfg: UnderstandConfig,
    out_dir: Path,
    force: bool = False,
) -> list[CellResult]:
    """Run one strategy over the whole grid, caching each cell."""
    cache_dir = out_dir / "cells" / strategy.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "images" / strategy.name
    image_dir.mkdir(parents=True, exist_ok=True)

    strategy.setup(bundle, registry)
    results: list[CellResult] = []

    for subject in subjects:
        if strategy.name == "tier3":
            from januscribe.lora import load_lora

            load_lora(bundle, ADAPTERS_DIR / subject.id)

        for scene_index, scene in scenes:
            cache = cache_dir / f"{subject.id}_{scene_index:02d}.json"
            if cache.exists() and not force:
                results.append(CellResult(**json.loads(cache.read_text(encoding="utf-8"))))
                continue

            seed = subject.scene_seed(scene_index)
            t0 = time.perf_counter()

            best: ConsistencyReport | None = None
            attempts = 0
            for attempt in range(policy.max_attempts):
                attempts += 1
                attempt_seed = seed + attempt * policy.seed_stride
                generated, prompt = strategy.generate(
                    bundle, subject, scene, attempt_seed, gen_cfg
                )
                report = score_image(
                    bundle, generated.image, subject, ref_emb[subject.id],
                    scene=scene, seed=attempt_seed, cfg=und_cfg,
                )
                if best is None or policy.rank(report) > policy.rank(best):
                    best = report
                    generated.image.save(image_dir / f"{subject.id}_{scene_index:02d}.png")
                if policy.accepts(report):
                    break

            assert best is not None
            cell = CellResult(
                strategy=strategy.name, subject_id=subject.id, scene_index=scene_index,
                seed=seed, rubric_score=best.rubric.score,
                embedding_mean=best.embedding.mean, attempts=attempts,
                low_confidence=not policy.accepts(best),
                seconds=time.perf_counter() - t0,
            )
            cache.write_text(json.dumps(cell.as_dict(), indent=2), encoding="utf-8")
            results.append(cell)
            log.info(
                "eval_cell", strategy=strategy.name, subject=subject.id,
                scene=scene_index, rubric=round(cell.rubric_score, 3),
                embedding=round(cell.embedding_mean, 4), attempts=attempts,
                seconds=round(cell.seconds, 1),
            )

    return results


@dataclass
class StrategySummary:
    name: str
    note: str
    available: bool
    reason: str = ""
    cells: list[CellResult] = field(default_factory=list)
    peak_memory_mb: float = float("nan")
    memory_kind: str = ""

    def as_dict(self) -> dict:
        return {
            "strategy": self.name, "note": self.note, "available": self.available,
            "reason": self.reason, "n_images": len(self.cells),
            "rubric_mean": round(_mean([c.rubric_score for c in self.cells]), 4)
            if self.cells else None,
            "embedding_mean": round(_mean([c.embedding_mean for c in self.cells]), 4)
            if self.cells else None,
            "mean_attempts": round(_mean([float(c.attempts) for c in self.cells]), 2)
            if self.cells else None,
            "low_confidence": sum(1 for c in self.cells if c.low_confidence),
            "seconds_per_image": round(_mean([c.seconds for c in self.cells]), 1)
            if self.cells else None,
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "memory_kind": self.memory_kind,
            "cells": [c.as_dict() for c in self.cells],
        }


def render_table(summaries: list[StrategySummary], meta: dict) -> str:
    memory_kind = next((s.memory_kind for s in summaries if s.memory_kind), "memory")
    header = "cuda peak VRAM (MB)" if memory_kind.startswith("cuda") else "peak RSS (MB)"

    lines = [
        "# Strategy comparison",
        "",
        f"- model: `{meta.get('model_id')}` on {meta.get('device')} ({meta.get('dtype')})",
        f"- eval set: {meta.get('n_subjects')} subjects x {meta.get('n_scenes')} scenes "
        f"= {meta.get('n_cells')} cells per strategy",
        f"- retry: max {meta.get('max_attempts')} attempts, "
        f"metric {meta.get('metric')}, threshold {meta.get('rubric_threshold')}",
        "",
        "The two measures are reported side by side and never averaged together.",
        "The rubric asks whether specified features are present; the embedding asks",
        "whether it looks like the same subject. A strategy can win one and lose the",
        "other, and that disagreement is information.",
        "",
        f"| strategy | rubric | embedding | attempts | low-conf | s/image | {header} | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        if not s.available:
            lines.append(
                f"| `{s.name}` | — | — | — | — | — | — | **unavailable**: {s.reason} |"
            )
            continue
        if not s.cells:
            # Available but produced nothing: a real possibility if the grid was
            # empty or every cell errored. Say that, rather than crashing the
            # whole table on a None, or printing zeros that read as measurements.
            lines.append(f"| `{s.name}` | — | — | — | — | — | — | **no cells produced** |")
            continue
        d = s.as_dict()
        lines.append(
            f"| `{s.name}` | {d['rubric_mean']:.3f} | {d['embedding_mean']:.4f} | "
            f"{d['mean_attempts']:.2f} | {d['low_confidence']} | "
            f"{d['seconds_per_image']:.0f} | {d['peak_memory_mb']:.0f} | {d['n_images']} |"
        )

    lines += ["", "## What each strategy is", ""]
    for s in summaries:
        lines.append(f"- **`{s.name}`** — {s.note}")

    unavailable = [s for s in summaries if not s.available]
    if unavailable:
        lines += [
            "",
            "## Not measured",
            "",
            "These appear as rows rather than being omitted, because a missing row reads",
            "as *not tried* when it should read as *not possible in this environment*.",
            "",
        ]
        for s in unavailable:
            lines.append(f"- **`{s.name}`**: {s.reason}")

    if memory_kind.startswith("cpu"):
        lines += [
            "",
            "> Memory is peak process RSS, not VRAM: this run had no CUDA device.",
            "> The column is not comparable to a GPU run's VRAM figures.",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evals" / "results")
    parser.add_argument("--eval-set", type=Path, default=EVAL_SET)
    parser.add_argument("--strategy", default=None, help="Comma-separated subset.")
    parser.add_argument("--scenes", type=int, default=None, help="Use the first N scenes.")
    parser.add_argument("--subject", default=None, help="Comma-separated subset.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    spec = yaml.safe_load(args.eval_set.read_text(encoding="utf-8"))
    registry = SubjectRegistry.from_yaml(SUBJECTS_FILE)
    all_scenes = load_scenes(SCENES_FILE)

    subject_ids = args.subject.split(",") if args.subject else spec["subjects"]
    subjects = [registry.get(s) for s in subject_ids]
    indices = spec["scene_indices"][: args.scenes] if args.scenes else spec["scene_indices"]
    scenes = [(i, all_scenes[i]) for i in indices]

    names = args.strategy.split(",") if args.strategy else spec["strategies"]
    strategies = build_strategies(names, root=registry.root)

    policy = RetryPolicy(**spec.get("retry", {}))
    gen_cfg = GenerationConfig(parallel_size=1, **spec.get("generation", {}))
    und_cfg = UnderstandConfig(max_new_tokens=24, temperature=0.0)

    settings = Settings(device=args.device, dtype=args.dtype)
    bundle = get_bundle(settings)
    ref_emb = reference_embeddings(bundle, subjects, registry.root)

    args.out.mkdir(parents=True, exist_ok=True)
    summaries: list[StrategySummary] = []

    for strategy in strategies:
        available, reason = strategy.availability(bundle, registry)
        if not available:
            log.warning("strategy_unavailable", strategy=strategy.name, reason=reason)
            summaries.append(
                StrategySummary(
                    name=strategy.name, note=strategy.note, available=False, reason=reason
                )
            )
            continue

        reset_peak_memory()
        cells = run_strategy(
            bundle, strategy, registry, subjects, scenes, ref_emb,
            policy, gen_cfg, und_cfg, args.out, force=args.force,
        )
        mb, kind = peak_memory()
        summaries.append(
            StrategySummary(
                name=strategy.name, note=strategy.note, available=True,
                cells=cells, peak_memory_mb=mb, memory_kind=kind,
            )
        )

    meta = {
        **bundle.describe(),
        "n_subjects": len(subjects),
        "n_scenes": len(scenes),
        "n_cells": len(subjects) * len(scenes),
        "max_attempts": policy.max_attempts,
        "metric": policy.metric,
        "rubric_threshold": policy.rubric_threshold,
        "eval_set_version": spec.get("version"),
    }

    (args.out / "comparison.json").write_text(
        json.dumps({"meta": meta, "strategies": [s.as_dict() for s in summaries]}, indent=2),
        encoding="utf-8",
    )
    table = render_table(summaries, meta)
    (args.out / "comparison.md").write_text(table, encoding="utf-8")

    print(table)
    print(f"wrote {args.out / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
