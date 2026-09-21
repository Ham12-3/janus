"""Image <-> VQ token roundtrip.

Janus-Pro's generation pathway is a LlamaGen-style VQ tokenizer with downsample
rate 16, so a 384x384 image is exactly 24*24 = 576 discrete codes drawn from a
16384-entry codebook. The tokenizer is invertible, which is what makes textual
inversion (M3) possible at all: reference images become teacher-forcing targets.

Everything here is deliberately explicit about tensor ranges. The VQ encoder
expects pixels in [-1, 1]; the decoder returns pixels in [-1, 1]. Getting that
wrong produces plausible-looking but badly degraded output, so both directions
are asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from januscribe.logging import get_logger
from januscribe.model import ModelBundle

log = get_logger(__name__)


@dataclass
class RoundtripResult:
    """Outcome of encode->decode on a single image."""

    tokens: torch.Tensor  # [n_tokens] int64
    original: Image.Image  # the 384x384 image that was actually encoded
    reconstruction: Image.Image
    psnr: float
    grid: int

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.numel())


def load_image_for_vq(path: str | Path, img_size: int = 384) -> Image.Image:
    """Load and centre-crop-resize an image to the square the VQ tokenizer expects."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((img_size, img_size), Image.LANCZOS)


def pil_to_vq_input(img: Image.Image, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """PIL RGB -> [1, 3, H, W] tensor scaled to [-1, 1], the VQ encoder's input range."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    x = x * 2.0 - 1.0
    return x.to(device=device, dtype=dtype)


def vq_output_to_numpy(dec: torch.Tensor) -> np.ndarray:
    """VQ decoder output in [-1, 1] -> uint8 HWC array, matching the reference code."""
    arr = dec.to(torch.float32).detach().cpu().numpy().transpose(0, 2, 3, 1)
    arr = np.clip((arr + 1.0) / 2.0 * 255.0, 0, 255)
    return arr.astype(np.uint8)


@torch.inference_mode()
def encode_image(bundle: ModelBundle, img: Image.Image) -> torch.Tensor:
    """Encode one PIL image to its flat VQ code indices.

    Returns int64 tensor of shape [grid*grid] (576 for a 384px image).

    The underlying call is ``gen_vision_model.encode(x) -> (quant, losses, info)``
    where ``info[2]`` holds the flattened argmin indices. That third element is
    the only part of the return value we want.
    """
    gen_dtype = bundle.vq_dtype
    x = pil_to_vq_input(img, dtype=gen_dtype, device=bundle.device)
    _quant, _losses, info = bundle.model.gen_vision_model.encode(x)
    indices = info[2]
    return indices.reshape(-1).to(torch.int64)


@torch.inference_mode()
def decode_tokens(
    bundle: ModelBundle, tokens: torch.Tensor, img_size: int = 384, patch_size: int = 16
) -> list[Image.Image]:
    """Decode VQ codes back to images.

    ``tokens`` is [n_tokens] for a single image or [batch, n_tokens] for several.
    """
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    batch, n_tokens = tokens.shape
    grid = img_size // patch_size
    if n_tokens != grid * grid:
        raise ValueError(
            f"expected {grid * grid} tokens for a {img_size}px image at patch {patch_size}, "
            f"got {n_tokens}"
        )
    codes = tokens.reshape(-1).to(device=bundle.device, dtype=torch.int32)
    dec = bundle.model.gen_vision_model.decode_code(
        codes, shape=[batch, bundle.codebook_embed_dim, grid, grid]
    )
    return [Image.fromarray(a) for a in vq_output_to_numpy(dec)]


def psnr(a: Image.Image, b: Image.Image) -> float:
    """Peak signal-to-noise ratio in dB between two same-size RGB images."""
    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    mse = float(np.mean((x - y) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def roundtrip(
    bundle: ModelBundle,
    image: str | Path | Image.Image,
    img_size: int = 384,
    patch_size: int = 16,
) -> RoundtripResult:
    """Encode an image to VQ tokens and decode it back, reporting fidelity."""
    img = image if isinstance(image, Image.Image) else load_image_for_vq(image, img_size)
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.LANCZOS)

    tokens = encode_image(bundle, img)
    recon = decode_tokens(bundle, tokens, img_size=img_size, patch_size=patch_size)[0]
    score = psnr(img, recon)

    grid = img_size // patch_size
    log.info(
        "vq_roundtrip",
        n_tokens=int(tokens.numel()),
        grid=f"{grid}x{grid}",
        psnr_db=round(score, 2),
        unique_codes=int(torch.unique(tokens).numel()),
        token_min=int(tokens.min()),
        token_max=int(tokens.max()),
        vq_dtype=str(bundle.vq_dtype),
    )
    return RoundtripResult(
        tokens=tokens.cpu(), original=img, reconstruction=recon, psnr=score, grid=grid
    )


def save_side_by_side(result: RoundtripResult, path: str | Path, label: bool = True) -> Path:
    """Write original | reconstruction as one PNG so the roundtrip can be eyeballed."""
    from PIL import ImageDraw

    w, h = result.original.size
    gap = 8
    banner = 22 if label else 0
    canvas = Image.new("RGB", (w * 2 + gap, h + banner), (18, 18, 18))
    canvas.paste(result.original, (0, banner))
    canvas.paste(result.reconstruction, (w + gap, banner))
    if label:
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 5), "original", fill=(230, 230, 230))
        draw.text((w + gap + 4, 5), f"VQ roundtrip ({result.psnr:.2f} dB)", fill=(230, 230, 230))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    log.info("saved_side_by_side", path=str(out), psnr_db=round(result.psnr, 2))
    return out
