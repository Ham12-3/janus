"""Score-then-resample loop.

The rule from the brief, and the one that matters: **never silently ship a
failed image**. If every attempt falls short, the best one is still returned --
but flagged low-confidence, so a weak page is visible in the output rather than
quietly folded into an average.

Scoring costs about as much as generating on CPU (six rubric questions at ~30 s
against ~200 s for a sample), so the policy defaults to few attempts. It is
config rather than code, because M6 ablates it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import torch
from PIL import Image
from pydantic import BaseModel, Field

from januscribe.config import GenerationConfig, UnderstandConfig
from januscribe.consistency import ConsistencyReport, score_image
from januscribe.generate import GeneratedImage, generate_images
from januscribe.logging import get_logger
from januscribe.model import ModelBundle
from januscribe.subjects import Subject

log = get_logger(__name__)

Metric = Literal["rubric", "embedding", "both"]


class RetryPolicy(BaseModel):
    """When to resample, how often, and what counts as good enough."""

    max_attempts: int = Field(3, ge=1, description="1 disables retrying.")
    metric: Metric = "rubric"
    rubric_threshold: float = Field(0.75, ge=0.0, le=1.0)
    embedding_threshold: float = Field(
        0.85,
        ge=-1.0,
        le=1.0,
        description="Absolute cosine. Calibrate against the run's similarity scale.",
    )
    seed_stride: int = Field(
        100_000,
        ge=1,
        description="Seed offset per attempt, large enough that a retry never "
        "collides with another subject's or scene's seed range.",
    )

    def accepts(self, report: ConsistencyReport) -> bool:
        rubric_ok = report.rubric.score >= self.rubric_threshold
        embedding_ok = report.embedding.mean >= self.embedding_threshold
        if self.metric == "rubric":
            return rubric_ok
        if self.metric == "embedding":
            return embedding_ok
        return rubric_ok and embedding_ok

    def rank(self, report: ConsistencyReport) -> tuple[float, float]:
        """Sort key for picking the best of a failed set.

        Ordered by whichever metric the policy gates on, with the other as a
        tiebreak, so "best" means best by the stated rule rather than by an
        invented blend of two measures this project deliberately keeps apart.
        """
        if self.metric == "embedding":
            return (report.embedding.mean, report.rubric.score)
        return (report.rubric.score, report.embedding.mean)


@dataclass
class Attempt:
    """One sample and what it scored."""

    seed: int
    rubric_score: float
    embedding_mean: float
    accepted: bool
    seconds: float

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "rubric_score": round(self.rubric_score, 4),
            "embedding_mean": round(self.embedding_mean, 4),
            "accepted": self.accepted,
            "seconds": round(self.seconds, 1),
        }


@dataclass
class RetryOutcome:
    """The image actually shipped, plus the full attempt history."""

    image: Image.Image
    generated: GeneratedImage
    report: ConsistencyReport
    attempts: list[Attempt] = field(default_factory=list)
    low_confidence: bool = False

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    def as_dict(self) -> dict:
        return {
            "low_confidence": self.low_confidence,
            "n_attempts": self.n_attempts,
            "chosen_seed": self.generated.seed,
            "attempts": [a.as_dict() for a in self.attempts],
            "report": self.report.as_dict(),
        }


def generate_with_retry(
    bundle: ModelBundle,
    prompt: str,
    subject: Subject,
    reference_embeddings: torch.Tensor,
    seed: int,
    policy: RetryPolicy | None = None,
    gen_cfg: GenerationConfig | None = None,
    und_cfg: UnderstandConfig | None = None,
    scene: str = "",
) -> RetryOutcome:
    """Sample, score, resample with a fresh seed until the policy is satisfied.

    Returns the first accepted image, or the best-scoring one flagged
    ``low_confidence`` when every attempt falls short.
    """
    policy = policy or RetryPolicy()
    gen_cfg = gen_cfg or GenerationConfig(parallel_size=1)
    single = GenerationConfig(**{**gen_cfg.model_dump(), "parallel_size": 1})

    attempts: list[Attempt] = []
    scored: list[tuple[GeneratedImage, ConsistencyReport]] = []

    for attempt_index in range(policy.max_attempts):
        attempt_seed = seed + attempt_index * policy.seed_stride
        t0 = time.perf_counter()
        generated = generate_images(bundle, prompt, seed=attempt_seed, cfg=single)[0]
        report = score_image(
            bundle,
            generated.image,
            subject,
            reference_embeddings,
            scene=scene,
            seed=attempt_seed,
            cfg=und_cfg,
        )
        accepted = policy.accepts(report)
        attempts.append(
            Attempt(
                seed=attempt_seed,
                rubric_score=report.rubric.score,
                embedding_mean=report.embedding.mean,
                accepted=accepted,
                seconds=time.perf_counter() - t0,
            )
        )
        scored.append((generated, report))
        log.info(
            "retry_attempt",
            subject=subject.id,
            attempt=attempt_index + 1,
            of=policy.max_attempts,
            seed=attempt_seed,
            rubric=round(report.rubric.score, 3),
            embedding=round(report.embedding.mean, 4),
            accepted=accepted,
        )
        if accepted:
            return RetryOutcome(
                image=generated.image,
                generated=generated,
                report=report,
                attempts=attempts,
                low_confidence=False,
            )

    # Nothing cleared the bar. Ship the best one, but say so loudly.
    best_generated, best_report = max(scored, key=lambda pair: policy.rank(pair[1]))
    log.warning(
        "retry_exhausted_low_confidence",
        subject=subject.id,
        attempts=policy.max_attempts,
        best_rubric=round(best_report.rubric.score, 3),
        best_embedding=round(best_report.embedding.mean, 4),
        rubric_threshold=policy.rubric_threshold,
        metric=policy.metric,
    )
    return RetryOutcome(
        image=best_generated.image,
        generated=best_generated,
        report=best_report,
        attempts=attempts,
        low_confidence=True,
    )
