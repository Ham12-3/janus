"""VQ encode/decode roundtrip against the real tokenizer.

Everything in M3 (textual inversion) depends on the encode path producing the
same 576 codes the model would have generated, so this test is the foundation
of the project rather than a nicety. It is marked ``slow`` because it needs the
weights and a minute of CPU.

The PSNR floor of 17 dB is chosen well below the ~22 dB this VQ-16 tokenizer
actually achieves on photographic content, so the test catches a broken
pipeline (wrong pixel range, transposed grid, fp16 underflow -- all of which
land in single digits) without flapping on content differences.
"""

from __future__ import annotations

import pytest
import torch

from januscribe.vq import decode_tokens, encode_image, load_image_for_vq, psnr, roundtrip

PSNR_FLOOR_DB = 17.0

pytestmark = pytest.mark.slow


def test_roundtrip_shape_and_fidelity(bundle, doge_path, tmp_path) -> None:
    result = roundtrip(bundle, doge_path)

    assert result.n_tokens == 576, "384px at downsample 16 must be a 24x24 = 576 code grid"
    assert result.grid == 24
    assert result.tokens.dtype == torch.int64
    assert int(result.tokens.min()) >= 0
    assert int(result.tokens.max()) < bundle.image_token_size
    assert result.reconstruction.size == (384, 384)
    assert result.psnr > PSNR_FLOOR_DB, f"roundtrip PSNR {result.psnr:.2f} dB is implausibly low"


def test_encode_is_deterministic(bundle, doge_path) -> None:
    img = load_image_for_vq(doge_path)
    assert torch.equal(encode_image(bundle, img), encode_image(bundle, img))


def test_reconstruction_beats_a_trivial_baseline(bundle, doge_path) -> None:
    """Guard against a roundtrip that "works" by returning mush.

    A grey frame is what a silently-broken decoder tends to produce, so the
    reconstruction must beat it by a wide margin.
    """
    from PIL import Image

    img = load_image_for_vq(doge_path)
    result = roundtrip(bundle, img)
    grey = Image.new("RGB", img.size, (128, 128, 128))
    assert result.psnr > psnr(img, grey) + 5.0


def test_decode_rejects_wrong_token_count(bundle) -> None:
    with pytest.raises(ValueError, match="expected 576 tokens"):
        decode_tokens(bundle, torch.zeros(512, dtype=torch.int64))


def test_batched_decode_matches_single(bundle, doge_path) -> None:
    """Decoding a batch must agree with decoding one image at a time.

    Agreement here is "within one LSB", not bit-exact, and that is a measured
    property of the backend rather than slack in the test. On CPU, torch picks a
    different convolution path for batch size 1 than for batch size > 1, which on
    this machine moves 4 of 442368 subpixels by 1/255 (98.6 dB). Batch 2 against
    batch 4 *is* bit-exact, as is the same call repeated, so the discontinuity is
    specifically the N=1 special case.

    Consequence for the project: sampled tokens are unaffected -- those are
    reproducible across batch sizes, which
    ``test_image_index_is_independent_of_batch_size`` pins down. Only the decoded
    PNG bytes can differ, so compare images by score, never by file hash.
    """
    import numpy as np

    tokens = encode_image(bundle, load_image_for_vq(doge_path))
    single = decode_tokens(bundle, tokens)[0]
    batched = decode_tokens(bundle, torch.stack([tokens, tokens]))
    assert len(batched) == 2

    max_abs = int(np.abs(np.asarray(single, np.int16) - np.asarray(batched[0], np.int16)).max())
    assert max_abs <= 1, f"batched decode drifted by {max_abs} levels, not a kernel-choice artefact"
    assert psnr(single, batched[0]) > 60.0

    # Rows inside one batch, and the same call repeated, are genuinely bit-exact.
    assert psnr(batched[0], batched[1]) == float("inf")
    assert psnr(single, decode_tokens(bundle, tokens)[0]) == float("inf")


def test_encoded_codes_are_usable_by_autograd(bundle, doge_path) -> None:
    """Codes must be ordinary tensors, not inference tensors.

    M3 teacher-forces on exactly these codes. Produced under
    torch.inference_mode they are permanently unusable in a backward pass
    ("Inference tensors cannot be saved for backward") -- and the failure
    surfaces only once training runs, far from the cause.
    """
    codes = encode_image(bundle, load_image_for_vq(doge_path))
    assert not torch.is_inference(codes)

    # Prove it end to end: a tracked loss against these targets must backward.
    logits = torch.zeros(codes.numel(), 16384, requires_grad=True)
    torch.nn.functional.cross_entropy(logits, codes).backward()
    assert logits.grad is not None
