"""Tier 4: put the reference sheet in context through the understanding path.

The idea is the obvious one: before generating the next image, show the model
what the subject looks like. Janus has a SigLIP pathway that consumes images, so
the reference sheet can be prepended to the prompt.

**Janus-Pro was never trained to do this.** Its training mixes text-to-image
generation and image-to-text understanding, but not "condition image generation
on reference images in context". Nothing guarantees the generation head attends
usefully to SigLIP embeddings sitting in the prefix. The brief expects this to
be unreliable; this module exists to measure it rather than assume it, and to
report honestly if it fails.

One design decision worth stating, because it is not forced:

The CFG unconditional branch replaces **everything** between BOS and the final
``image_start`` with pad -- including the reference-image positions. The
alternative, keeping the images in the unconditional branch and padding only the
text, would make CFG amplify the difference between "text plus images" and
"images alone", which is not the quantity we want to steer on. Padding both
keeps the unconditional branch meaning "no conditioning at all", exactly as it
does in Tier 0, so Tier 4's guidance scale stays comparable to every other tier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from januscribe.config import GenerationConfig
from januscribe.generate import GeneratedImage, build_generation_prompt, sample_from_prefix
from januscribe.logging import get_logger
from januscribe.model import ModelBundle
from januscribe.subjects import Subject

log = get_logger(__name__)


def build_conditioned_prefix(
    bundle: ModelBundle,
    scene_prompt: str,
    reference_images: Sequence[Image.Image],
    parallel_size: int = 1,
) -> torch.Tensor:
    """Build a [2N, T, H] CFG prefix whose conditional rows include the references.

    Layout: the SFT-formatted prompt carries one ``<image_placeholder>`` per
    reference image, which the processor expands into begin/576 tokens/end. The
    usual trailing ``image_start_tag`` is appended so the model switches into
    code generation.
    """
    if not reference_images:
        raise ValueError("visual conditioning needs at least one reference image")

    processor = bundle.processor
    placeholders = "".join(f"{processor.image_tag}\n" for _ in reference_images)
    conversation = [
        {"role": "User", "content": f"{placeholders}{scene_prompt}"},
        {"role": "Assistant", "content": ""},
    ]
    sft = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation, sft_format=processor.sft_format, system_prompt=""
    )
    prepared = processor(
        prompt=sft, images=list(reference_images), force_batchify=True
    ).to(bundle.device, dtype=bundle.dtype)

    # input_ids is mutated in place by prepare_inputs_embeds, so keep a copy for
    # the unconditional branch (NOTES-API.md section 6).
    input_ids = prepared.input_ids.clone()
    cond = bundle.model.prepare_inputs_embeds(**prepared)[0]  # [T, H]

    embed = bundle.model.language_model.get_input_embeddings()
    start_id = torch.tensor([processor.image_start_id], device=bundle.device)
    start_embed = embed(start_id).to(cond.dtype)  # [1, H]
    cond = torch.cat([cond, start_embed], dim=0)  # [T+1, H]

    # Unconditional branch: keep BOS and the trailing image_start, pad everything
    # between -- references included. See the module docstring.
    uncond_ids = input_ids[0].clone()
    uncond_ids[1:] = processor.pad_id
    uncond = embed(uncond_ids).to(cond.dtype)
    uncond = torch.cat([uncond, start_embed], dim=0)

    prefix = torch.stack([cond, uncond], dim=0)  # [2, T+1, H]
    prefix = prefix.repeat(parallel_size, 1, 1)  # interleaved (cond, uncond) x N
    log.info(
        "visual_prefix_built",
        n_references=len(reference_images),
        prefix_tokens=int(prefix.shape[1]),
        rows=int(prefix.shape[0]),
    )
    return prefix


@torch.inference_mode()
def generate_with_visual_context(
    bundle: ModelBundle,
    scene_prompt: str,
    reference_images: Sequence[Image.Image],
    seed: int,
    cfg: GenerationConfig | None = None,
    progress_every: int = 0,
) -> list[GeneratedImage]:
    """Sample images with the reference sheet in context.

    Goes through the same ``sample_from_prefix`` loop as every other tier, so a
    difference in output is a difference in conditioning rather than in sampling.
    """
    cfg = cfg or GenerationConfig(parallel_size=1)
    cfg.validate_geometry()

    prefix = build_conditioned_prefix(
        bundle, scene_prompt, reference_images, parallel_size=cfg.parallel_size
    )
    generated, images, elapsed = sample_from_prefix(
        bundle, prefix, seed=seed, cfg=cfg, progress_every=progress_every
    )
    return [
        GeneratedImage(
            image=images[i],
            tokens=generated[i].detach().cpu(),
            seed=seed + i,
            prompt=scene_prompt,
            cfg_weight=cfg.cfg_weight,
            temperature=cfg.temperature,
            index=i,
            meta={
                "model_id": bundle.settings.model_id,
                "strategy": "tier4_visual",
                "n_references": len(reference_images),
                "seconds_total": round(elapsed, 2),
                "caveat": "Janus-Pro was not trained for in-context image conditioning",
            },
        )
        for i in range(cfg.parallel_size)
    ]


def load_reference_images(
    subject: Subject, root: str | Path, limit: int | None = None
) -> list[Image.Image]:
    """Load a subject's reference sheet as PIL images.

    Defaults to a small number because every reference costs 576 tokens of
    prefix, and the prefix is recomputed for all 576 generation steps.
    """
    paths = subject.reference_paths(Path(root))
    if not paths:
        raise FileNotFoundError(
            f"subject {subject.id!r} has no reference images; run `januscribe build-refs`"
        )
    if limit is not None:
        paths = paths[:limit]
    return [Image.open(p).convert("RGB") for p in paths]
