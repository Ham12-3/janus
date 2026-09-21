"""Tier 3: LoRA adapters instead of a single embedding row.

Same reconstruction objective as ``inversion.py`` -- teacher-force the 576 VQ
codes of the reference images -- but the capacity lives in low-rank adapters on
the language model's attention and MLP projections rather than in one 2048-d
vector.

The trade the brief asks about: more capacity, more drift risk. A soft token
can only move where the *prompt* lands in embedding space; LoRA can move the
model itself, so it can capture a subject the token cannot, and it can also
degrade everything else the model draws. That is why ``general_quality_probe``
exists -- a Tier 3 that wins on the subject while wrecking unrelated prompts has
not actually won, and this is the cheapest way to see that happen.

The trigger stays a fixed soft token whose embedding is **not** trained. That
keeps the comparison clean: Tier 2 trains the token and freezes the weights,
Tier 3 freezes the token and trains the weights. Whatever differs between them
is the capacity, not the prompt.

peft is an optional dependency: absent, this module raises a clear error instead
of failing obscurely halfway through a run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from pydantic import BaseModel, Field

from januscribe.inversion import (
    DEFAULT_TEMPLATES,
    InversionConfig,
    _noun_mean_embedding,
    build_prompt_ids,
    encode_references,
    forward_loss,
    freeze_everything,
    register_soft_token,
    token_for,
    trainable_parameters,
)
from januscribe.logging import get_logger
from januscribe.model import ModelBundle
from januscribe.seeding import seed_everything
from januscribe.subjects import Subject

log = get_logger(__name__)

# Llama attention and MLP projections. Adapting both is the brief's ask; the
# attention-only subset is the usual cheaper variant and is left as a sweep knob.
LLAMA_TARGETS: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


def peft_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("peft") is not None


def _require_peft():
    if not peft_available():
        raise RuntimeError(
            "Tier 3 needs peft, which is an optional dependency. "
            'Install it with: uv pip install "peft>=0.11,<0.14"'
        )
    import peft

    return peft


class LoraSettings(BaseModel):
    """LoRA shape plus the same optimisation schedule as textual inversion."""

    r: int = Field(8, ge=1)
    alpha: int = Field(16, ge=1)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    target_modules: tuple[str, ...] = LLAMA_TARGETS
    inversion: InversionConfig = Field(default_factory=InversionConfig)

    def to_peft_config(self):
        peft = _require_peft()
        return peft.LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=list(self.target_modules),
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
        )


@dataclass
class LoraResult:
    """A trained adapter plus the loss curve that proves it trained."""

    subject_id: str
    token: str
    adapter_dir: Path | None
    losses: list[float]
    trainable_elements: int
    seconds: float
    meta: dict = field(default_factory=dict)

    @property
    def first_loss(self) -> float:
        return self.losses[0] if self.losses else float("nan")

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")


def attach_lora(bundle: ModelBundle, settings: LoraSettings | None = None):
    """Inject LoRA layers into the language model, in place.

    peft replaces the targeted ``nn.Linear`` modules inside the existing module
    tree, so ``bundle.model.language_model.model(...)`` -- the call the sampler
    makes -- routes through the adapters afterwards with no other change.
    """
    peft = _require_peft()
    settings = settings or LoraSettings()
    freeze_everything(bundle)

    peft_config = settings.to_peft_config()
    peft.inject_adapter_in_model(peft_config, bundle.model.language_model)

    trainable = trainable_parameters(bundle.model)
    n_elements = sum(p.numel() for p in trainable)
    total = sum(p.numel() for p in bundle.model.parameters())
    log.info(
        "lora_attached", r=settings.r, alpha=settings.alpha,
        targets=list(settings.target_modules),
        trainable_tensors=len(trainable), trainable_elements=n_elements,
        fraction_of_model=round(n_elements / max(total, 1), 6),
    )
    if not trainable:
        raise RuntimeError(
            "peft injected no trainable parameters; check target_modules matches "
            f"this architecture (got {settings.target_modules})"
        )
    return trainable


def train_lora(
    bundle: ModelBundle,
    subject: Subject,
    image_paths: Sequence[str | Path],
    settings: LoraSettings | None = None,
    out_dir: str | Path | None = None,
) -> LoraResult:
    """Train LoRA adapters on the same VQ-reconstruction objective as Tier 2."""
    peft = _require_peft()
    settings = settings or LoraSettings()
    cfg = settings.inversion
    seed_everything(cfg.seed)

    token = token_for(subject.id)
    token_id = register_soft_token(bundle, token)
    trainable = attach_lora(bundle, settings)

    # The trigger embedding is frozen at the noun mean. Tier 2 trains this and
    # freezes the weights; Tier 3 does the opposite, so the two differ only in
    # where the capacity sits.
    frozen_trigger = _noun_mean_embedding(bundle, subject.noun).to(bundle.device)

    examples = encode_references(bundle, image_paths)
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)

    was_training = bundle.model.training
    bundle.model.language_model.train()  # LoRA dropout and checkpointing both need it

    losses: list[float] = []
    t0 = time.perf_counter()
    log.info(
        "lora_train_start", subject=subject.id, steps=cfg.steps, lr=cfg.lr,
        effective_batch=cfg.batch_size, n_references=len(examples),
    )

    try:
        for step in range(cfg.steps):
            lr = cfg.lr_at(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(cfg.batch_size):
                ex = examples[int(torch.randint(len(examples), (1,), generator=generator))]
                template = cfg.templates[
                    int(torch.randint(len(cfg.templates), (1,), generator=generator))
                ]
                prompt_ids = build_prompt_ids(bundle, template, token)
                loss = forward_loss(bundle, prompt_ids, ex.codes, frozen_trigger, token_id)
                (loss / cfg.batch_size).backward()
                step_loss += float(loss.detach()) / cfg.batch_size

            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            losses.append(step_loss)

            if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.steps - 1):
                log.info(
                    "lora_step", step=step, loss=round(step_loss, 4), lr=round(lr, 6),
                    seconds=round(time.perf_counter() - t0, 1),
                )
    finally:
        bundle.model.train() if was_training else bundle.model.eval()

    elapsed = time.perf_counter() - t0
    adapter_dir: Path | None = None
    if out_dir is not None:
        adapter_dir = Path(out_dir) / subject.id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        state = peft.get_peft_model_state_dict(bundle.model.language_model)
        from safetensors.torch import save_file

        save_file(
            {k: v.to(torch.float32).cpu().contiguous() for k, v in state.items()},
            adapter_dir / "adapter.safetensors",
        )
        log.info("lora_saved", path=str(adapter_dir), tensors=len(state))

    n_elements = sum(p.numel() for p in trainable)
    log.info(
        "lora_train_done", subject=subject.id,
        first_loss=round(losses[0], 4) if losses else None,
        final_loss=round(losses[-1], 4) if losses else None,
        seconds=round(elapsed, 1), trainable_elements=n_elements,
    )
    return LoraResult(
        subject_id=subject.id, token=token, adapter_dir=adapter_dir, losses=losses,
        trainable_elements=n_elements, seconds=elapsed,
        meta={
            "r": settings.r, "alpha": settings.alpha,
            "target_modules": list(settings.target_modules),
            "steps": cfg.steps, "lr": cfg.lr, "seed": cfg.seed,
            "n_references": len(examples),
            "model_id": bundle.settings.model_id,
        },
    )


def load_lora(bundle: ModelBundle, adapter_dir: str | Path, settings: LoraSettings | None = None):
    """Attach adapters and load trained weights into them."""
    peft = _require_peft()
    from safetensors.torch import load_file

    attach_lora(bundle, settings or LoraSettings())
    state = load_file(Path(adapter_dir) / "adapter.safetensors")
    peft.set_peft_model_state_dict(bundle.model.language_model, state)
    log.info("lora_loaded", path=str(adapter_dir), tensors=len(state))


GENERAL_QUALITY_PROMPTS: tuple[str, ...] = (
    "a bowl of fruit on a wooden table",
    "a lighthouse on a rocky coast at sunset",
    "a city street in the rain",
)


def general_quality_probe(
    bundle: ModelBundle, seed: int = 909, prompts: Sequence[str] = GENERAL_QUALITY_PROMPTS
) -> dict:
    """Embed a few unrelated prompts, so Tier 3's collateral damage is visible.

    Run this before and after attaching a trained adapter and compare the
    embeddings. Large drift on prompts that have nothing to do with the subject
    is the cost side of LoRA's extra capacity, and the brief asks for it
    explicitly. This measures *change*, not beauty -- it cannot tell you the
    images got worse, only that they got different, which is the honest limit of
    a cheap probe.
    """
    from januscribe.config import GenerationConfig
    from januscribe.consistency import embed_images
    from januscribe.generate import generate_images

    images = []
    for i, prompt in enumerate(prompts):
        images.append(
            generate_images(
                bundle, prompt, seed=seed + i, cfg=GenerationConfig(parallel_size=1)
            )[0].image
        )
    embeddings = embed_images(bundle, images)
    return {
        "prompts": list(prompts),
        "seed": seed,
        "embeddings": embeddings.cpu(),
        "images": images,
    }


def general_quality_drift(before: dict, after: dict) -> dict:
    """Cosine similarity per prompt between a before and after probe.

    1.0 means the adapter left unrelated generation untouched. Lower means the
    model moved, and the subject gain has to be worth that.
    """
    a, b = before["embeddings"], after["embeddings"]
    if a.shape != b.shape:
        raise ValueError(f"probe shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")
    sims = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    per_prompt = {p: round(float(s), 4) for p, s in zip(before["prompts"], sims)}
    return {
        "per_prompt": per_prompt,
        "mean": round(float(sims.mean()), 4),
        "worst": round(float(sims.min()), 4),
    }
