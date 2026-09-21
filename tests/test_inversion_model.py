"""Soft-token training against the real model.

These are the two checks the brief asks for by name -- that exactly one
parameter trains, and that training actually reduces the loss -- plus the
alignment check that would catch a teacher-forcing off-by-one.

There is no honest cheap version of these. A mocked forward pass would let a
wrong objective, a mis-scattered embedding, or a shifted target window all pass
green. Marked ``slow``: on CPU each optimisation step is a full forward and
backward through a 1.3B model over ~617 tokens.
"""

from __future__ import annotations

import pytest
import torch

from januscribe.inversion import (
    InversionConfig,
    SoftToken,
    _noun_mean_embedding,
    apply_soft_token,
    build_prompt_ids,
    encode_references,
    forward_loss,
    freeze_everything,
    register_soft_token,
    token_for,
    trainable_parameters,
    train_soft_token,
)
from januscribe.subjects import SubjectRegistry

pytestmark = pytest.mark.slow

SUBJECTS_YAML = "configs/subjects.yaml"


@pytest.fixture(scope="module")
def fox(bundle):
    registry = SubjectRegistry.from_yaml(SUBJECTS_YAML)
    subject = registry.get("fox")
    paths = subject.reference_paths(registry.root)
    if len(paths) < 2:
        pytest.skip("fox reference sheet not built; run `januscribe build-refs` first")
    return subject, paths[:3]


def test_only_one_parameter_trains(bundle, fox) -> None:
    """The brief's central invariant: everything frozen but the new row.

    Checked as a count of tensors *and* of elements, because a frozen-looking
    model with one 210M-element embedding table still requiring grad would pass
    a naive count of one.
    """
    subject, _ = fox
    register_soft_token(bundle, token_for(subject.id))
    freeze_everything(bundle)

    assert trainable_parameters(bundle.model) == [], "model must be fully frozen"

    init = _noun_mean_embedding(bundle, subject.noun)
    soft = torch.nn.Parameter(init.clone())
    trainable = [soft] + trainable_parameters(bundle.model)

    assert len(trainable) == 1
    assert trainable[0].numel() == bundle.model.config.language_config.hidden_size == 2048


def test_embedding_table_and_lm_head_stay_frozen(bundle, fox) -> None:
    """tie_word_embeddings is True here, so the table is also the output head."""
    subject, _ = fox
    register_soft_token(bundle, token_for(subject.id))
    freeze_everything(bundle)

    embed = bundle.model.language_model.get_input_embeddings()
    assert embed.weight.requires_grad is False
    assert bundle.model.gen_embed.weight.requires_grad is False
    assert all(not p.requires_grad for p in bundle.model.gen_head.parameters())


def test_gradient_reaches_only_the_soft_vector(bundle, fox) -> None:
    """One backward pass must leave the model's own grads untouched."""
    subject, paths = fox
    token = token_for(subject.id)
    token_id = register_soft_token(bundle, token)
    freeze_everything(bundle)
    bundle.model.zero_grad(set_to_none=True)

    examples = encode_references(bundle, paths[:1])
    soft = torch.nn.Parameter(_noun_mean_embedding(bundle, subject.noun).clone())
    prompt_ids = build_prompt_ids(bundle, "a photo of {token}", token)

    loss = forward_loss(bundle, prompt_ids, examples[0].codes, soft, token_id)
    loss.backward()

    assert soft.grad is not None
    assert torch.isfinite(soft.grad).all()
    assert float(soft.grad.abs().sum()) > 0, "the soft token received no gradient at all"
    assert all(p.grad is None for p in bundle.model.parameters()), (
        "a frozen model parameter accumulated a gradient"
    )


def test_loss_is_aligned_with_the_targets(bundle, fox) -> None:
    """Real codes must score far better than shuffled ones.

    This is the check that catches a teacher-forcing off-by-one: a window
    shifted by a position still produces a finite, plausible loss, but it would
    no longer be meaningfully better than predicting a permutation.
    """
    subject, paths = fox
    token = token_for(subject.id)
    token_id = register_soft_token(bundle, token)
    freeze_everything(bundle)

    examples = encode_references(bundle, paths[:1])
    codes = examples[0].codes
    soft = torch.nn.Parameter(_noun_mean_embedding(bundle, subject.noun).clone())
    prompt_ids = build_prompt_ids(bundle, "a photo of {token}", token)

    with torch.no_grad():
        real = float(forward_loss(bundle, prompt_ids, codes, soft, token_id))
        shuffled = codes[torch.randperm(codes.numel(), generator=torch.Generator().manual_seed(0))]
        scrambled = float(forward_loss(bundle, prompt_ids, shuffled, soft, token_id))

    assert real < scrambled, f"real codes ({real:.3f}) should beat shuffled ({scrambled:.3f})"
    # Chance over a 16384-way codebook is ln(16384) ~= 9.70.
    assert real < 9.70, f"loss {real:.3f} is no better than uniform chance"


def test_missing_token_in_template_is_an_error(bundle, fox) -> None:
    subject, paths = fox
    token = token_for(subject.id)
    token_id = register_soft_token(bundle, token)
    examples = encode_references(bundle, paths[:1])
    soft = torch.nn.Parameter(_noun_mean_embedding(bundle, subject.noun).clone())
    prompt_ids = build_prompt_ids(bundle, "a photo of a fox", token)

    with pytest.raises(ValueError, match="does not contain the soft token"):
        forward_loss(bundle, prompt_ids, examples[0].codes, soft, token_id)


@pytest.mark.timeout(7200)
def test_training_actually_reduces_loss(bundle, fox) -> None:
    """The brief's other named check. A token that does not train is useless.

    Deliberately short: this asserts the optimisation *works*, not that it has
    converged. Convergence needs the hyperparameter sweep, which needs a GPU.
    """
    subject, paths = fox
    cfg = InversionConfig(
        steps=12, batch_size=1, lr=1e-2, lr_end=1e-3, warmup_steps=2, log_every=2, seed=0
    )
    result = train_soft_token(bundle, subject, paths, cfg=cfg)

    assert len(result.losses) == 12
    assert all(torch.isfinite(torch.tensor(result.losses))), "loss went nan/inf"

    first_third = sum(result.losses[:4]) / 4
    last_third = sum(result.losses[-4:]) / 4
    assert last_third < first_third, (
        f"loss did not fall: first four {first_third:.4f}, last four {last_third:.4f}"
    )

    assert result.soft_token.vector.shape == (2048,)
    assert torch.isfinite(result.soft_token.vector).all()
    # It must have actually moved off its initialisation.
    init = _noun_mean_embedding(bundle, subject.noun).cpu()
    assert not torch.allclose(result.soft_token.vector, init, atol=1e-6)


def test_learned_token_installs_into_the_embedding_table(bundle, fox, tmp_path) -> None:
    """Save, load, apply -- after which any prompt can use the token."""
    subject, _ = fox
    vector = torch.randn(2048)
    saved = SoftToken(subject_id=subject.id, token=token_for(subject.id), vector=vector).save(
        tmp_path / "fox.safetensors"
    )
    loaded = SoftToken.load(saved)
    token_id = apply_soft_token(bundle, loaded)

    embed = bundle.model.language_model.get_input_embeddings()
    installed = embed.weight[token_id].detach().float().cpu()
    assert torch.allclose(installed, vector, atol=1e-2)

    # And the token now survives a tokenizer roundtrip as a single id.
    ids = bundle.tokenizer.encode(f"a photo of {loaded.token}", add_special_tokens=False)
    assert token_id in ids
