"""Text-to-image on the Janus-Pro generation pathway.

This is the reference CFG sampling loop from ``generation_inference.py``, with
three deliberate changes, all of which matter downstream:

1. **Per-image generators.** The reference calls ``torch.multinomial`` once over
   the whole batch, so the pixels you get for image *i* depend on how many
   images you asked for. Here each row draws from its own generator seeded
   ``base_seed + i``, so image *i* at a given seed is byte-identical whether it
   was sampled alone or in a batch of eight. Ablations need that.
2. **No implicit ``.cuda()``.** Device and dtype come from the bundle.
3. **Tokens are returned, not just pixels.** M3 needs the code ids.

The conditioning sequence is built exactly as the reference does it:
``sft_prompt + image_start_tag``, duplicated into interleaved
(conditional, unconditional) rows, where the unconditional row replaces
everything between BOS and the final token with ``pad_id``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from januscribe.cache import new_kv_cache
from januscribe.config import GenerationConfig
from januscribe.logging import get_logger
from januscribe.model import ModelBundle
from januscribe.seeding import make_generator, sample_categorical
from januscribe.vq import vq_output_to_numpy

log = get_logger(__name__)


@dataclass
class GeneratedImage:
    """One sampled image plus the provenance needed to reproduce it exactly."""

    image: Image.Image
    tokens: torch.Tensor  # [n_tokens] int64 VQ codes
    seed: int
    prompt: str
    cfg_weight: float
    temperature: float
    index: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(out)
        return out


def build_generation_prompt(bundle: ModelBundle, prompt: str, system_prompt: str = "") -> str:
    """Wrap a raw prompt in the SFT template and append the begin-of-image tag.

    The trailing ``image_start_tag`` is what tells the model the next tokens are
    image codes rather than text. Without it you get text continuation.
    """
    conversation = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": ""},
    ]
    sft = bundle.processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=bundle.processor.sft_format,
        system_prompt=system_prompt,
    )
    return sft + bundle.processor.image_start_tag


def _build_cfg_batch(
    bundle: ModelBundle, input_ids: torch.Tensor, parallel_size: int
) -> torch.Tensor:
    """Interleave conditional and unconditional rows: [cond0, uncond0, cond1, uncond1, ...].

    The unconditional row keeps the first token (BOS) and the last token
    (begin_of_image) and pads everything in between, which is how Janus was
    trained to express "no prompt".
    """
    seq_len = input_ids.shape[0]
    tokens = torch.zeros((parallel_size * 2, seq_len), dtype=torch.long, device=bundle.device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = bundle.processor.pad_id
    return tokens


@torch.inference_mode()
def generate_images(
    bundle: ModelBundle,
    prompt: str,
    seed: int,
    cfg: GenerationConfig | None = None,
    system_prompt: str = "",
    progress_every: int = 0,
) -> list[GeneratedImage]:
    """Sample ``cfg.parallel_size`` images for ``prompt``.

    Image *i* uses seed ``seed + i``. Returns them in index order.
    """
    cfg = cfg or GenerationConfig()
    cfg.validate_geometry()
    n = cfg.parallel_size
    n_tokens = cfg.image_token_num_per_image

    full_prompt = build_generation_prompt(bundle, prompt, system_prompt=system_prompt)
    input_ids = torch.tensor(
        bundle.processor.tokenizer.encode(full_prompt), dtype=torch.long, device=bundle.device
    )
    tokens = _build_cfg_batch(bundle, input_ids, n)

    prefix = bundle.model.language_model.get_input_embeddings()(tokens)

    log.info(
        "generation_start", prompt=prompt, seed=seed, parallel_size=n,
        cfg_weight=cfg.cfg_weight, temperature=cfg.temperature, prompt_tokens=int(input_ids.numel()),
    )
    generated, images, elapsed = sample_from_prefix(
        bundle, prefix, seed=seed, cfg=cfg, progress_every=progress_every
    )

    return [
        GeneratedImage(
            image=images[i],
            tokens=generated[i].detach().cpu(),
            seed=seed + i,
            prompt=prompt,
            cfg_weight=cfg.cfg_weight,
            temperature=cfg.temperature,
            index=i,
            meta={
                "model_id": bundle.settings.model_id,
                "device": str(bundle.device),
                "dtype": str(bundle.dtype),
                "base_seed": seed,
                "sft_prompt": full_prompt,
                "seconds_total": round(elapsed, 2),
            },
        )
        for i in range(n)
    ]


@torch.inference_mode()
def sample_from_prefix(
    bundle: ModelBundle,
    prefix: torch.Tensor,
    seed: int,
    cfg: GenerationConfig,
    progress_every: int = 0,
) -> tuple[torch.Tensor, list[Image.Image], float]:
    """Run the CFG sampling loop from an arbitrary conditioning prefix.

    ``prefix`` is [2N, T, H] with rows interleaved (cond, uncond, cond, ...).
    Factored out of ``generate_images`` so alternative conditioning -- notably
    M4's visual self-conditioning, which prepends reference-image embeddings --
    samples through exactly the same loop rather than a near-copy that could
    drift from it.

    Returns (codes [N, n_tokens], images, seconds).
    """
    n = cfg.parallel_size
    n_tokens = cfg.image_token_num_per_image
    generators = [make_generator(seed + i, bundle.device) for i in range(n)]
    inputs_embeds = prefix
    generated = torch.zeros((n, n_tokens), dtype=torch.long, device=bundle.device)

    t0 = time.perf_counter()
    past_key_values = new_kv_cache()

    for step in range(n_tokens):
        outputs = bundle.model.language_model.model(
            inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values
        )
        past_key_values = outputs.past_key_values
        logits = bundle.model.gen_head(outputs.last_hidden_state[:, -1, :])

        logit_cond, logit_uncond = logits[0::2, :], logits[1::2, :]
        guided = logit_uncond + cfg.cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(guided.to(torch.float32) / cfg.temperature, dim=-1)

        # Row-by-row so each image's randomness is independent of the batch size.
        next_token = torch.stack(
            [sample_categorical(probs[i], generators[i]).reshape(()) for i in range(n)]
        )
        generated[:, step] = next_token

        # Feed the same chosen code to both the conditional and unconditional rows.
        doubled = next_token.repeat_interleave(2)
        inputs_embeds = bundle.model.prepare_gen_img_embeds(doubled).unsqueeze(dim=1)

        if progress_every and (step + 1) % progress_every == 0:
            log.info(
                "generation_progress", step=step + 1, of=n_tokens,
                seconds=round(time.perf_counter() - t0, 1),
            )

    images = _decode_batch(bundle, generated, cfg)
    elapsed = time.perf_counter() - t0
    log.info(
        "generation_done", seconds=round(elapsed, 1),
        seconds_per_image=round(elapsed / max(n, 1), 1), parallel_size=n,
    )
    return generated, images, elapsed


def _decode_batch(
    bundle: ModelBundle, generated: torch.Tensor, cfg: GenerationConfig
) -> list[Image.Image]:
    """Turn [n, 576] code ids into PIL images via the VQ decoder."""
    n = generated.shape[0]
    codes = generated.reshape(-1).to(device=bundle.device, dtype=torch.int32)
    dec = bundle.model.gen_vision_model.decode_code(
        codes, shape=[n, bundle.codebook_embed_dim, cfg.grid, cfg.grid]
    )
    return [Image.fromarray(a) for a in vq_output_to_numpy(dec)]


def save_all(images: Sequence[GeneratedImage], out_dir: str | Path, stem: str = "img") -> list[Path]:
    """Save a batch, naming each file after the seed that produced it."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return [g.save(directory / f"{stem}_seed{g.seed}.png") for g in images]
