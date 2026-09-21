"""Consistency measurement.

Two independent measures, reported side by side and **never averaged together**.
They fail in different directions and collapsing them would hide that:

1. **Attribute rubric.** Decompose the canonical description into atomic binary
   attributes and ask the understanding path one strict yes/no question per
   attribute. Score is the fraction answered yes. Every raw answer is kept, so a
   bad parse or a hedging model stays visible rather than baked into a number.
   Sensitive to *whether the specified details are present*. Blind to whether two
   images showing those details depict the same individual.

2. **Embedding similarity.** Cosine similarity between the SigLIP embedding of a
   generated image and each image in the subject's reference sheet. Sensitive to
   overall visual likeness. Has a *compressed dynamic range* on same-category
   images -- see ``calibrate_similarity``, which is why a raw cosine of 0.98 is
   meaningless without a reference scale.

Pooling choice: the embedding is the **mean over the 576 patch tokens** of
``vision_model``'s output. Janus builds its SigLIP tower with ``ignore_head=True``,
so those patch tokens are exactly the representation the model itself consumes;
the MAP attention-pool head is present in the checkpoint but unused by Janus.
Measured on this machine, mean-pooling separates two different foxes from a dog
at 0.980 / 0.603, while the attention head gives 0.990 / 0.839 -- 2.5x less
spread. Mean-pooling it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from januscribe.config import UnderstandConfig
from januscribe.logging import get_logger
from januscribe.model import ModelBundle
from januscribe.subjects import Subject
from januscribe.understand import YesNo, ask_yes_no

log = get_logger(__name__)

ImageInput = str | Path | Image.Image


@dataclass
class AttributeVerdict:
    """One rubric question and what the model actually said."""

    attribute: str
    verdict: YesNo
    raw_answer: str


@dataclass
class RubricScore:
    """Fraction of atomic attributes the model confirms, plus every raw answer."""

    verdicts: list[AttributeVerdict]

    @property
    def n_total(self) -> int:
        return len(self.verdicts)

    @property
    def n_yes(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "yes")

    @property
    def n_unclear(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "unclear")

    @property
    def score(self) -> float:
        """Fraction answered yes. An unclear answer counts against, never for."""
        return self.n_yes / self.n_total if self.n_total else 0.0

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "n_yes": self.n_yes,
            "n_total": self.n_total,
            "n_unclear": self.n_unclear,
            "verdicts": [
                {"attribute": v.attribute, "verdict": v.verdict, "raw_answer": v.raw_answer}
                for v in self.verdicts
            ],
        }


@dataclass
class EmbeddingScore:
    """Cosine similarity against every image in the reference sheet."""

    per_reference: list[float]

    @property
    def mean(self) -> float:
        return sum(self.per_reference) / len(self.per_reference) if self.per_reference else 0.0

    @property
    def best(self) -> float:
        return max(self.per_reference) if self.per_reference else 0.0

    @property
    def worst(self) -> float:
        return min(self.per_reference) if self.per_reference else 0.0

    def as_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "best": round(self.best, 4),
            "worst": round(self.worst, 4),
            "n_references": len(self.per_reference),
            "per_reference": [round(x, 4) for x in self.per_reference],
        }


@dataclass
class ConsistencyReport:
    """Both measures for one image. Deliberately not reduced to one number."""

    subject_id: str
    scene: str
    seed: int
    image_path: str | None
    rubric: RubricScore
    embedding: EmbeddingScore
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {
            "subject_id": self.subject_id,
            "scene": self.scene,
            "seed": self.seed,
            "image_path": self.image_path,
            "rubric": self.rubric.as_dict(),
            "embedding": self.embedding.as_dict(),
        }
        if self.meta:
            out["meta"] = self.meta
        return out


def _as_pil(images: Sequence[ImageInput]) -> list[Image.Image]:
    return [
        img if isinstance(img, Image.Image) else Image.open(img).convert("RGB") for img in images
    ]


@torch.inference_mode()
def embed_images(bundle: ModelBundle, images: Sequence[ImageInput]) -> torch.Tensor:
    """SigLIP embeddings for a list of images: [N, 1024], L2-normalised, float32.

    Preprocessing goes through ``processor.image_processor`` (the same path the
    understanding pathway uses), because the vision tower carries no
    normalisation of its own -- ``CLIPVisionTower.image_norm`` is None for
    Janus-Pro, so the mean/std live entirely in the image processor.
    """
    pil = _as_pil(list(images))
    if not pil:
        return torch.zeros((0, 1), dtype=torch.float32)
    pixel_values = bundle.processor.image_processor(pil, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device=bundle.device, dtype=bundle.dtype)
    patch_tokens = bundle.model.vision_model(pixel_values)  # [N, 576, 1024]
    pooled = patch_tokens.float().mean(dim=1)
    return F.normalize(pooled, dim=-1)


def score_attributes(
    bundle: ModelBundle,
    image: ImageInput,
    subject: Subject,
    cfg: UnderstandConfig | None = None,
) -> RubricScore:
    """Ask one strict yes/no question per atomic attribute."""
    cfg = cfg or UnderstandConfig(max_new_tokens=24, temperature=0.0)
    pil = _as_pil([image])[0]
    verdicts: list[AttributeVerdict] = []
    for attribute in subject.attributes:
        verdict, answer = ask_yes_no(bundle, pil, attribute, cfg=cfg)
        verdicts.append(
            AttributeVerdict(attribute=attribute, verdict=verdict, raw_answer=answer.text)
        )
    score = RubricScore(verdicts=verdicts)
    log.info(
        "rubric_scored",
        subject=subject.id,
        score=round(score.score, 3),
        n_yes=score.n_yes,
        n_total=score.n_total,
        n_unclear=score.n_unclear,
    )
    return score


def score_embedding(
    bundle: ModelBundle, image: ImageInput, reference_embeddings: torch.Tensor
) -> EmbeddingScore:
    """Cosine similarity of one image against a matrix of reference embeddings."""
    if reference_embeddings.numel() == 0:
        return EmbeddingScore(per_reference=[])
    query = embed_images(bundle, [image])  # [1, D]
    sims = (query @ reference_embeddings.to(query.dtype).T).squeeze(0)
    return EmbeddingScore(per_reference=[float(x) for x in sims])


def score_image(
    bundle: ModelBundle,
    image: ImageInput,
    subject: Subject,
    reference_embeddings: torch.Tensor,
    scene: str = "",
    seed: int = -1,
    image_path: str | None = None,
    cfg: UnderstandConfig | None = None,
) -> ConsistencyReport:
    """Run both measures on one generated image."""
    return ConsistencyReport(
        subject_id=subject.id,
        scene=scene,
        seed=seed,
        image_path=image_path,
        rubric=score_attributes(bundle, image, subject, cfg=cfg),
        embedding=score_embedding(bundle, image, reference_embeddings),
    )


@dataclass
class SimilarityScale:
    """Reference scale that makes a raw cosine number interpretable.

    Without this, 0.98 says nothing: two *different* foxes already sit at 0.98
    under this embedding. The numbers that matter are the gaps.
    """

    within_subject: dict[str, float]
    cross_subject: dict[str, float]

    @property
    def within_mean(self) -> float:
        vals = list(self.within_subject.values())
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def cross_mean(self) -> float | None:
        """None when there is only one subject, so there is nothing to compare.

        Returning 0.0 here would be a lie that reads as a perfect floor, and it
        would make ``separation`` report the within-subject mean as if the metric
        had that much room. A single-subject run simply has no floor.
        """
        vals = list(self.cross_subject.values())
        return sum(vals) / len(vals) if vals else None

    @property
    def separation(self) -> float | None:
        """How much room the metric has to work in, or None if uncalibrated."""
        cross = self.cross_mean
        return None if cross is None or not self.within_subject else self.within_mean - cross

    def as_dict(self) -> dict:
        cross, sep = self.cross_mean, self.separation
        return {
            "within_subject": {k: round(v, 4) for k, v in self.within_subject.items()},
            "cross_subject": {k: round(v, 4) for k, v in self.cross_subject.items()},
            "within_mean": round(self.within_mean, 4) if self.within_subject else None,
            "cross_mean": round(cross, 4) if cross is not None else None,
            "separation": round(sep, 4) if sep is not None else None,
        }


def calibrate_similarity(reference_embeddings: dict[str, torch.Tensor]) -> SimilarityScale:
    """Establish what similar and different actually look like on this metric.

    ``within_subject`` is the mean pairwise similarity inside each subject's own
    reference sheet -- the practical ceiling, since those images already differ
    by pose and framing. ``cross_subject`` is the mean similarity between
    different subjects' sheets -- the floor any consistency score has to beat.
    """
    within: dict[str, float] = {}
    for sid, emb in reference_embeddings.items():
        n = int(emb.shape[0])
        if n < 2:
            continue
        sims = emb @ emb.T
        off_diagonal = sims[~torch.eye(n, dtype=torch.bool, device=sims.device)]
        within[sid] = float(off_diagonal.mean())

    cross: dict[str, float] = {}
    ids = sorted(reference_embeddings)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            ea, eb = reference_embeddings[a], reference_embeddings[b]
            if ea.numel() and eb.numel():
                cross[f"{a}|{b}"] = float((ea @ eb.T).mean())

    scale = SimilarityScale(within_subject=within, cross_subject=cross)
    summary = scale.as_dict()
    log.info(
        "similarity_calibrated",
        within_mean=summary["within_mean"],
        cross_mean=summary["cross_mean"],
        separation=summary["separation"],
    )
    return scale
