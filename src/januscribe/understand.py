"""Image + question -> text, on the SigLIP understanding pathway.

This is the other half of Janus-Pro: a SigLIP-L encoder at 384x384 feeding the
same transformer. M2 leans on it hard, because the attribute rubric asks the
model a strict yes/no question per attribute and counts the yeses.

The structured-answer helpers live here rather than in ``consistency.py`` so
that the parsing rules sit next to the prompt that produces them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
from PIL import Image

from januscribe.config import UnderstandConfig
from januscribe.logging import get_logger
from januscribe.model import ModelBundle
from januscribe.seeding import seed_everything

log = get_logger(__name__)

ImageInput = str | Path | Image.Image

YesNo = Literal["yes", "no", "unclear"]

_YES = re.compile(r"\b(yes|yeah|yep|correct|true|affirmative)\b", re.I)
_NO = re.compile(r"\b(no|nope|not|isn'?t|aren'?t|false|negative|none)\b", re.I)


@dataclass
class Answer:
    """A model answer plus the context needed to audit it later."""

    question: str
    text: str
    n_images: int

    def as_yes_no(self) -> YesNo:
        """Coerce free text to yes/no/unclear.

        Deliberately conservative: an answer that matches both or neither is
        'unclear', never silently counted as a pass. M2 logs the raw text too,
        so a bad parse is visible rather than baked into a score.
        """
        return parse_yes_no(self.text)


def parse_yes_no(text: str) -> YesNo:
    """Parse a yes/no answer. The first decisive word wins."""
    stripped = text.strip()
    if not stripped:
        return "unclear"
    head = stripped[:80]
    yes_at = _YES.search(head)
    no_at = _NO.search(head)
    if yes_at and no_at:
        return "yes" if yes_at.start() < no_at.start() else "no"
    if yes_at:
        return "yes"
    if no_at:
        return "no"
    return "unclear"


def _load_images(images: Sequence[ImageInput]) -> list[Image.Image]:
    return [
        img if isinstance(img, Image.Image) else Image.open(img).convert("RGB") for img in images
    ]


@torch.inference_mode()
def ask(
    bundle: ModelBundle,
    images: ImageInput | Sequence[ImageInput],
    question: str,
    cfg: UnderstandConfig | None = None,
    seed: int | None = None,
) -> Answer:
    """Ask a question about one or more images and return the model's answer.

    ``cfg.temperature == 0`` means greedy decoding, which is what the rubric
    scorer uses so that repeated scoring of the same image is stable.
    """
    cfg = cfg or UnderstandConfig()
    if isinstance(images, (str, Path, Image.Image)):
        images = [images]
    pil_images = _load_images(list(images))

    placeholders = "".join(f"{bundle.processor.image_tag}\n" for _ in pil_images)
    conversation = [
        {"role": "User", "content": f"{placeholders}{question}"},
        {"role": "Assistant", "content": ""},
    ]

    prepare_inputs = bundle.processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(bundle.device, dtype=bundle.dtype)

    inputs_embeds = bundle.model.prepare_inputs_embeds(**prepare_inputs)

    if seed is not None:
        seed_everything(seed)

    do_sample = cfg.temperature > 0.0
    tokenizer = bundle.tokenizer
    outputs = bundle.model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=do_sample,
        temperature=cfg.temperature if do_sample else None,
        top_p=cfg.top_p if do_sample else None,
        use_cache=True,
    )
    text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()
    log.info(
        "understand_answer", question=question, answer=text[:200], n_images=len(pil_images),
        do_sample=do_sample,
    )
    return Answer(question=question, text=text, n_images=len(pil_images))


def ask_yes_no(
    bundle: ModelBundle,
    images: ImageInput | Sequence[ImageInput],
    attribute: str,
    cfg: UnderstandConfig | None = None,
) -> tuple[YesNo, Answer]:
    """Ask a strict binary attribute question. Returns the verdict and the raw answer.

    Kept as its own function because M2's rubric must phrase every attribute
    question identically -- a rubric whose wording drifts between runs is not a
    measurement.
    """
    cfg = cfg or UnderstandConfig(max_new_tokens=24, temperature=0.0)
    question = (
        f"Look at the image. Does it show: {attribute}? "
        "Answer with exactly one word, 'yes' or 'no'."
    )
    answer = ask(bundle, images, question, cfg=cfg)
    return answer.as_yes_no(), answer
