"""Textual inversion on the generation pathway.

Learn a single embedding vector -- a soft token like ``<sbj_fox>`` -- that means
"this specific fox", by reconstructing the VQ codes of reference images.

Three facts from NOTES-API.md shape everything here:

1. **The model has two input embedding tables** (§4). ``gen_embed`` (16384 x 8)
   feeds previously-generated image codes; ``language_model``'s table
   (102400 x 2048) feeds the prompt. The soft token goes in the *text* table,
   because it appears in the prompt. Putting it in ``gen_embed`` would train an
   8-d vector in codebook space that no prompt can reference: it would run
   without error and produce plausible garbage.
2. **Image logits come from ``gen_head``, not ``lm_head``** (§3). The training
   objective is cross-entropy over 16384 VQ codes, using the same head the
   sampler uses.
3. **``gen_vision_model.encode`` recovers the code space the model generates in**
   (§5). That is what makes reference images usable as teacher-forcing targets.

## Why the trainable weight is a standalone Parameter

The brief says to add one row to the embedding table and freeze everything else.
Done literally -- ``requires_grad=True`` on the whole 102401 x 2048 table with a
hook zeroing every other row -- autograd allocates a full-size gradient tensor:
**840 MB to carry a 2048-float update**, on a 16 GB CPU box that already holds
8.4 GB of fp32 weights.

So the row is held as a standalone ``nn.Parameter`` of shape [2048] and written
into ``inputs_embeds`` at the placeholder positions. This is *mathematically
identical* -- the same vector receives exactly the same gradient -- but the
gradient is 8 KB. It also makes the brief's assertion tight rather than
approximate: there is exactly one trainable tensor and it has 2048 elements, so
"only that row trains" is checked by construction rather than by trusting a hook.

The token is still registered in the tokenizer and the table is still resized,
because inference needs the token to tokenize to a single id. That resized row
is frozen during training and filled in afterwards.

Note ``tie_word_embeddings`` defaults to True for this config, so the text
table and ``lm_head`` are the same weights. Nothing here trains either.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, Field
from safetensors.torch import load_file, save_file

from januscribe.logging import get_logger
from januscribe.model import ModelBundle
from januscribe.seeding import seed_everything
from januscribe.subjects import Subject
from januscribe.vq import encode_image, load_image_for_vq

log = get_logger(__name__)

# Varied templates so the token learns the subject rather than one pose or
# framing. The brief's pool, plus a few that vary medium and composition.
DEFAULT_TEMPLATES: tuple[str, ...] = (
    "a photo of {token}",
    "{token}",
    "{token} standing in a field",
    "a drawing of {token}",
    "a close-up photo of {token}",
    "{token} on a plain background",
    "an illustration of {token}",
    "a portrait of {token}",
)


class InversionConfig(BaseModel):
    """Hyperparameters for soft-token training.

    Defaults are the brief's starting point. The sweep over lr / steps /
    effective batch is deliberately left to a GPU: a single step costs seconds
    on CPU, so the values here are chosen to be *correct and checkable*, not
    converged.
    """

    steps: int = Field(500, ge=1)
    lr: float = Field(1e-3, gt=0)
    lr_end: float = Field(1e-4, gt=0, description="Cosine decay target.")
    warmup_steps: int = Field(50, ge=0)
    batch_size: int = Field(4, ge=1, description="Effective batch, via grad accumulation.")
    weight_decay: float = Field(0.0, ge=0.0)
    grad_clip: float = Field(1.0, gt=0)
    gradient_checkpointing: bool = True
    templates: tuple[str, ...] = DEFAULT_TEMPLATES
    seed: int = 42
    log_every: int = 25

    def lr_at(self, step: int) -> float:
        """Linear warmup then cosine decay from ``lr`` to ``lr_end``."""
        if self.warmup_steps and step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        span = max(self.steps - self.warmup_steps, 1)
        progress = min((step - self.warmup_steps) / span, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_end + (self.lr - self.lr_end) * cosine


@dataclass
class SoftToken:
    """A learned subject embedding, plus everything needed to reproduce it."""

    subject_id: str
    token: str
    vector: torch.Tensor  # [hidden_size], float32
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        metadata = {k: json.dumps(v) for k, v in self.meta.items()}
        metadata["subject_id"] = self.subject_id
        metadata["token"] = self.token
        save_file({"embedding": self.vector.to(torch.float32).cpu().contiguous()}, out, metadata)
        log.info("soft_token_saved", path=str(out), subject=self.subject_id, token=self.token)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "SoftToken":
        from safetensors import safe_open

        out = Path(path)
        tensors = load_file(out)
        with safe_open(out, framework="pt") as f:
            raw = dict(f.metadata() or {})
        subject_id = raw.pop("subject_id", "unknown")
        token = raw.pop("token", f"<sbj_{subject_id}>")
        meta = {}
        for k, v in raw.items():
            try:
                meta[k] = json.loads(v)
            except json.JSONDecodeError:
                meta[k] = v
        return cls(
            subject_id=subject_id, token=token, vector=tensors["embedding"], meta=meta
        )


def token_for(subject_id: str) -> str:
    """The soft-token string for a subject id."""
    return f"<sbj_{subject_id}>"


def _noun_mean_embedding(bundle: ModelBundle, noun: str) -> torch.Tensor:
    """Mean text-table embedding of the subject's head noun.

    The brief's initialisation, and it matters: starting from the mean of "fox"
    puts the token in a region of embedding space that already decodes to
    fox-shaped images, which converges far faster than random noise.

    This is the 2048-d *text* embedding, not an 8-d VQ codebook vector -- see
    the module docstring.
    """
    ids = bundle.tokenizer.encode(f" {noun}", add_special_tokens=False)
    if not ids:
        raise ValueError(f"noun {noun!r} tokenised to nothing")
    table = bundle.model.language_model.get_input_embeddings().weight
    with torch.no_grad():
        vec = table[torch.tensor(ids, device=table.device)].mean(dim=0).clone()
    log.info("soft_token_init", noun=noun, n_subword_tokens=len(ids), norm=float(vec.norm()))
    return vec.to(torch.float32)


def register_soft_token(bundle: ModelBundle, token: str) -> int:
    """Add the token to the tokenizer and grow the embedding table by one row.

    Returns the new token's id. Idempotent: an already-registered token just
    returns its existing id.
    """
    tokenizer = bundle.tokenizer
    existing = tokenizer.convert_tokens_to_ids(token)
    unk = getattr(tokenizer, "unk_token_id", None)
    if existing is not None and existing != unk and existing >= 0:
        embed_rows = bundle.model.language_model.get_input_embeddings().weight.shape[0]
        if existing < embed_rows:
            return int(existing)

    tokenizer.add_special_tokens({"additional_special_tokens": [token]})
    bundle.model.language_model.resize_token_embeddings(len(tokenizer))
    token_id = int(tokenizer.convert_tokens_to_ids(token))
    log.info(
        "soft_token_registered", token=token, token_id=token_id,
        embedding_rows=bundle.model.language_model.get_input_embeddings().weight.shape[0],
    )
    return token_id


def freeze_everything(bundle: ModelBundle) -> int:
    """Freeze every parameter in the model. Returns how many were frozen."""
    n = 0
    for param in bundle.model.parameters():
        if param.requires_grad:
            param.requires_grad_(False)
        n += 1
    return n


def trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def _assert_no_dropout(bundle: ModelBundle) -> None:
    """Fail if the language model has any active dropout.

    Gradient checkpointing forces train() mode, which is only safe because every
    dropout probability in this Llama config is 0. If that ever changes, the
    training forward would stop matching the sampling forward and the token
    would learn against a distribution it will never be sampled from.
    """
    offenders = [
        (name, module.p)
        for name, module in bundle.model.language_model.named_modules()
        if isinstance(module, torch.nn.Dropout) and module.p > 0
    ]
    attn_dropout = getattr(bundle.model.config.language_config, "attention_dropout", 0.0)
    if offenders or attn_dropout:
        raise RuntimeError(
            "gradient checkpointing needs train() mode, but this model has active "
            f"dropout (attention_dropout={attn_dropout}, modules={offenders}). "
            "Disable gradient_checkpointing so training stays in eval() and matches "
            "the sampling path."
        )


@dataclass
class TrainingExample:
    """One reference image, already encoded to its 576 VQ codes."""

    path: str
    codes: torch.Tensor  # [576] int64


def encode_references(
    bundle: ModelBundle, image_paths: Sequence[str | Path], img_size: int = 384
) -> list[TrainingExample]:
    """Encode every reference image to VQ codes once, up front.

    Encoding is deterministic, so doing it once and reusing the codes for every
    step is both correct and much cheaper than re-encoding per step.
    """
    examples: list[TrainingExample] = []
    for path in image_paths:
        img = load_image_for_vq(path, img_size=img_size)
        codes = encode_image(bundle, img).cpu()
        examples.append(TrainingExample(path=str(path), codes=codes))
    if not examples:
        raise ValueError("no reference images given")
    log.info(
        "references_encoded", n=len(examples), codes_per_image=int(examples[0].codes.numel())
    )
    return examples


def build_prompt_ids(
    bundle: ModelBundle, template: str, token: str, system_prompt: str = ""
) -> torch.Tensor:
    """Tokenise a training prompt exactly as the generation path would.

    Same SFT template, same trailing ``image_start_tag``. If the conditioning
    sequence differs from what sampling produces, the token learns something the
    sampler will never see.
    """
    from januscribe.generate import build_generation_prompt

    text = template.format(token=token)
    full = build_generation_prompt(bundle, text, system_prompt=system_prompt)
    ids = bundle.tokenizer.encode(full)
    return torch.tensor(ids, dtype=torch.long, device=bundle.device)


def prediction_slice(n_text: int, n_codes: int) -> tuple[int, int]:
    """Hidden-state positions whose logits are scored, given the layout below.

    With T text tokens (the last being ``image_start``) and N image codes, the
    input is ``text[0..T-1] + code[0..N-2]`` of length ``T+N-1``. Position
    ``T-1`` predicts ``code_0`` and position ``T-1+i`` predicts ``code_i``, so
    the scored window is ``[T-1, T-1+N)``.

    Extracted so the off-by-one can be tested without a forward pass: an error
    here would shift every target by one position and still train to a
    plausible-looking loss.
    """
    if n_text < 1:
        raise ValueError(f"need at least one text token, got {n_text}")
    if n_codes < 1:
        raise ValueError(f"need at least one image code, got {n_codes}")
    start = n_text - 1
    return start, start + n_codes


def forward_loss(
    bundle: ModelBundle,
    prompt_ids: torch.Tensor,
    codes: torch.Tensor,
    soft_vector: torch.nn.Parameter,
    token_id: int,
) -> torch.Tensor:
    """Teacher-forced cross-entropy over the image codes only.

    Sequence layout, with T text tokens (the last being ``image_start``) and
    N image codes::

        input   [ text_0 .. text_{T-1} ][ code_0 .. code_{N-2} ]     length T+N-1
        predict                  code_0   code_1 ..      code_{N-1}

    So hidden state at position ``T-1`` predicts ``code_0``, and position
    ``T-1+i`` predicts ``code_i``. Taking ``gen_head`` over exactly those N
    positions is what masks the loss to the image portion -- the text positions
    never enter the objective, so no separate mask is needed and none can be
    forgotten.
    """
    embed = bundle.model.language_model.get_input_embeddings()
    text_embeds = embed(prompt_ids)  # [T, H], frozen table

    # Write the trainable vector into every placeholder position. This is the
    # only path by which gradient reaches anything.
    placeholder = prompt_ids == token_id
    if not bool(placeholder.any()):
        raise ValueError(
            f"prompt does not contain the soft token (id {token_id}); "
            "the template must include {token}"
        )
    text_embeds = torch.where(
        placeholder.unsqueeze(-1),
        soft_vector.to(text_embeds.dtype).unsqueeze(0),
        text_embeds,
    )

    codes = codes.to(bundle.device)
    n_codes = int(codes.numel())
    # Teacher forcing feeds codes 0..N-2 and predicts 1..N-1; code 0 is
    # predicted from the trailing image_start token.
    img_embeds = bundle.model.prepare_gen_img_embeds(codes[:-1])  # [N-1, H]

    inputs_embeds = torch.cat([text_embeds, img_embeds.to(text_embeds.dtype)], dim=0).unsqueeze(0)

    outputs = bundle.model.language_model.model(
        inputs_embeds=inputs_embeds, use_cache=False
    )
    hidden = outputs.last_hidden_state[0]  # [T+N-1, H]

    start, end = prediction_slice(int(prompt_ids.numel()), n_codes)
    predict_from = hidden[start:end]  # [N, H]
    if predict_from.shape[0] != n_codes:
        raise RuntimeError(
            f"scored {predict_from.shape[0]} positions for {n_codes} codes; "
            "sequence layout and prediction_slice disagree"
        )
    logits = bundle.model.gen_head(predict_from)  # [N, 16384]

    return F.cross_entropy(logits.to(torch.float32), codes)


@dataclass
class InversionResult:
    """A trained token plus the loss curve that proves it trained."""

    soft_token: SoftToken
    losses: list[float]
    steps: int
    seconds: float

    @property
    def first_loss(self) -> float:
        return self.losses[0] if self.losses else float("nan")

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")


def train_soft_token(
    bundle: ModelBundle,
    subject: Subject,
    image_paths: Sequence[str | Path],
    cfg: InversionConfig | None = None,
) -> InversionResult:
    """Learn a soft token for ``subject`` from its reference images."""
    cfg = cfg or InversionConfig()
    seed_everything(cfg.seed)

    token = token_for(subject.id)
    token_id = register_soft_token(bundle, token)
    freeze_everything(bundle)

    init = _noun_mean_embedding(bundle, subject.noun)
    soft_vector = torch.nn.Parameter(init.clone().to(device=bundle.device, dtype=torch.float32))

    # The brief's check, made tight: exactly one trainable tensor exists in the
    # whole process, and it is this one.
    model_trainable = trainable_parameters(bundle.model)
    if model_trainable:
        raise RuntimeError(
            f"{len(model_trainable)} model parameters still require grad; "
            "the whole model must be frozen before inversion"
        )
    log.info(
        "inversion_trainable_check",
        model_params_requiring_grad=0,
        trainable_tensors=1,
        trainable_elements=int(soft_vector.numel()),
    )

    checkpointing_on = False
    if cfg.gradient_checkpointing:
        try:
            bundle.model.language_model.gradient_checkpointing_enable()
            checkpointing_on = True
            log.info("gradient_checkpointing", enabled=True)
        except Exception as exc:  # pragma: no cover - version dependent
            log.warning("gradient_checkpointing_unavailable", error=str(exc))

    examples = encode_references(bundle, image_paths)
    optimizer = torch.optim.AdamW(
        [soft_vector], lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)

    was_training = bundle.model.training
    bundle.model.eval()

    # transformers only checkpoints when the module is in train() mode
    # (`self.gradient_checkpointing and self.training`), so in eval it is a
    # silent no-op -- and this run needs the memory it saves. Llama here has
    # every dropout at 0, so train() is behaviourally identical to eval(); that
    # is asserted rather than assumed, because if a future config turns dropout
    # on, training would quietly diverge from sampling.
    if checkpointing_on:
        _assert_no_dropout(bundle)
        bundle.model.language_model.train()

    losses: list[float] = []
    t0 = time.perf_counter()
    log.info(
        "inversion_start", subject=subject.id, token=token, steps=cfg.steps,
        lr=cfg.lr, effective_batch=cfg.batch_size, n_references=len(examples),
    )

    try:
        for step in range(cfg.steps):
            lr = cfg.lr_at(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            # Micro-batch of 1 with gradient accumulation. Padding variable-length
            # prompts into a real batch would need an attention mask that the
            # generation path never uses, so accumulation keeps training and
            # sampling on identical sequences -- and it is what fits in memory.
            for _ in range(cfg.batch_size):
                ex = examples[int(torch.randint(len(examples), (1,), generator=generator))]
                template = cfg.templates[
                    int(torch.randint(len(cfg.templates), (1,), generator=generator))
                ]
                prompt_ids = build_prompt_ids(bundle, template, token)
                loss = forward_loss(bundle, prompt_ids, ex.codes, soft_vector, token_id)
                (loss / cfg.batch_size).backward()
                step_loss += float(loss.detach()) / cfg.batch_size

            torch.nn.utils.clip_grad_norm_([soft_vector], cfg.grad_clip)
            optimizer.step()
            losses.append(step_loss)

            if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.steps - 1):
                log.info(
                    "inversion_step", step=step, loss=round(step_loss, 4), lr=round(lr, 6),
                    seconds=round(time.perf_counter() - t0, 1),
                )
    finally:
        if checkpointing_on:
            try:
                bundle.model.language_model.gradient_checkpointing_disable()
            except Exception:  # pragma: no cover
                pass
        bundle.model.train() if was_training else bundle.model.eval()

    elapsed = time.perf_counter() - t0
    soft = SoftToken(
        subject_id=subject.id,
        token=token,
        vector=soft_vector.detach().to(torch.float32).cpu(),
        meta={
            "noun": subject.noun,
            "model_id": bundle.settings.model_id,
            "steps": cfg.steps,
            "lr": cfg.lr,
            "lr_end": cfg.lr_end,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
            "n_references": len(examples),
            "reference_paths": [e.path for e in examples],
            "first_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "seconds": round(elapsed, 1),
            "templates": list(cfg.templates),
        },
    )
    log.info(
        "inversion_done", subject=subject.id, steps=cfg.steps,
        first_loss=round(losses[0], 4) if losses else None,
        final_loss=round(losses[-1], 4) if losses else None,
        seconds=round(elapsed, 1),
    )
    return InversionResult(
        soft_token=soft, losses=losses, steps=cfg.steps, seconds=elapsed
    )


def apply_soft_token(bundle: ModelBundle, soft: SoftToken) -> int:
    """Install a learned token into the model so any prompt can use it.

    Registers the token, grows the table if needed, and writes the learned
    vector into its row. After this, ``generate_images(bundle, "<sbj_fox> in a
    meadow", ...)`` works with no other changes.
    """
    token_id = register_soft_token(bundle, soft.token)
    embed = bundle.model.language_model.get_input_embeddings()
    with torch.no_grad():
        embed.weight[token_id] = soft.vector.to(
            device=embed.weight.device, dtype=embed.weight.dtype
        )
    log.info(
        "soft_token_applied", token=soft.token, token_id=token_id, subject=soft.subject_id
    )
    return token_id


def load_and_apply(bundle: ModelBundle, path: str | Path) -> SoftToken:
    """Load a ``.safetensors`` soft token and install it for inference."""
    soft = SoftToken.load(path)
    apply_soft_token(bundle, soft)
    return soft
