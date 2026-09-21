# JanusScribe

Multi-page illustrated documents on DeepSeek **Janus-Pro**, where the same
character or product looks the same in every image.

Janus-Pro has no subject consistency out of the box. Generate "a red fox in a
blue scarf" twice and you get two different foxes — here are the two this repo
actually produced at seeds 42 and 43, same prompt:

| seed 42 | seed 43 |
|---|---|
| ![](docs/m1/fox_seed42.png) | ![](docs/m1/fox_seed43.png) |
| different face, different eyes | different scarf, different pose |

Closing that gap — and **measuring** whether it closed — is the whole project.

---

## Status

| milestone | state |
|---|---|
| **M1** primitives | done, verified against the real model |
| **M2** subject registry, consistency measures, validated judge | done; Tier 0 baseline grid in progress |
| **M3** textual inversion | built, correctness proven on CPU; converged training needs a GPU |
| **M4** retry loop, LoRA, visual conditioning | built |
| **M5** document pipeline | built |
| **M6** eval harness | built; the table needs trained Tier 2/3 artefacts to be meaningful |

**84 fast tests** (no weights needed) and **22 model-backed tests** (never mocked).

---

## Install

Requires Python 3.11 or 3.12 (see the `attrdict` note below for why not 3.13).

```bash
git clone https://github.com/deepseek-ai/Janus vendor/Janus
uv venv --python 3.11 .venv
uv pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu  # or the CUDA index
uv pip install -e .
uv pip install -e ./vendor/Janus --no-deps
```

The `--no-deps` on the vendored package is **not optional**. Janus depends on
`attrdict`, which imports `collections.Mapping` and therefore cannot be
installed on Python 3.10 or later. This project depends on `attrdict3` instead.
Installing Janus with its own dependencies will break the environment. Full
reasoning in [NOTES-API.md](NOTES-API.md) §7.

Optional extras: `.[lora]` for Tier 3 (peft), `.[flash]` for flash-attn on CUDA.
Neither is required; a missing flash-attn falls back to `sdpa` with one warning.

Weights download on first use (~3.9 GB for Janus-Pro-1B): `januscribe info`.

---

## Use

```bash
# primitives
januscribe gen "a red fox wearing a navy blue scarf" --seed 42 -n 2
januscribe ask image.png "what colour is the fox?"
januscribe vq-roundtrip assets/doge.png

# subjects and consistency
januscribe subjects                      # show the registry
januscribe build-refs --subject fox      # generate a reference sheet
januscribe baseline --scenes 20          # Tier 0 grid, resumable
januscribe learn --subject fox           # learn a soft token (M3)

# documents
januscribe build --config configs/fox-story.yaml --dry-run   # plan only, no model
januscribe build --config configs/fox-story.yaml --out dist --debug

# the comparison table
python evals/validate_judge.py           # check the rubric still measures
python evals/compare_strategies.py --out evals/results
```

Model, device and dtype are configuration, never hardcoded:

```bash
januscribe --config configs/cuda-7b.yaml build --config configs/fox-story.yaml
JANUSCRIBE_MODEL_ID=deepseek-ai/Janus-Pro-7B januscribe info
```

`--device` accepts `auto`, `cuda`, `mps`, `cpu`. MPS and CPU are slow but work.

---

## Output

A build emits a **self-contained HTML file** with every image inlined as base64,
a PDF, the plan as JSON, and a score sidecar.
[docs/m5/demo.html](docs/m5/demo.html) is a real example, assembled from Tier 0
images and their real scores.

In `--debug` mode each image carries its consistency scores, and any image that
failed the retry threshold is **flagged in the document itself** — a weak page
announces itself to whoever reads it, not just to a log.

---

## How consistency is measured

Two measures, side by side, **never averaged**:

1. **Attribute rubric** — decompose the canonical description into atomic facts
   and ask the understanding path a strict yes/no per fact.
2. **Embedding similarity** — SigLIP cosine against the subject's reference sheet.

They fail differently. The rubric asks whether specified *features* are present;
the embedding asks whether it *looks like the same subject*. A strategy can win
one and lose the other, and that disagreement is information.

A raw cosine is meaningless on its own — two visibly different foxes already sit
at 0.98 — so every report ships a calibration scale: within-reference-sheet
similarity as ceiling, cross-subject as floor.

### The judge had to be validated before any of it counted

The first rubric agreed with hand-labelled reality **43% of the time**. Two of
six fox attributes were wrong on *every single image*, because they used
prepositional binding (`white stripes on the scarf` → 0/5) and a contrastive
clause (`goggles resting on the forehead rather than over the eyes` → 2/4).

Asked to *describe* rather than confirm, the same model on the same image says
the scarf is "predominantly blue with white stripes". It perceives exactly what
it denies. The questions were malformed, not the vision.

Rewritten against five measured rules, the rubric scores **63/63 = 1.000**, with
a **17/18** false-positive rate on cross-subject negative controls. That check
is now a standing eval: [`evals/validate_judge.py`](evals/validate_judge.py)
fails below 0.90 and refuses to run if the ground-truth keys drift from the
config. Full write-up: [docs/m2/judge-validation.md](docs/m2/judge-validation.md).

**Attribute phrasings are part of the measuring apparatus, not prose.**

---

## Testing

```bash
pytest -m "not slow"     # 84 tests, ~19 s, no weights needed
pytest -m slow           # 22 tests, needs weights; hours on CPU
```

The model is never mocked. Tests needing it are marked `slow` and **skip with an
explicit reason** when weights are absent, so a green run on a machine without
weights cannot be mistaken for one that exercised the model.

They have earned it. Bugs found by the test suite, not by inspection:

- VQ codes were produced under `torch.inference_mode`, making them **unusable as
  training targets** (`Inference tensors cannot be saved for backward`). Only the
  tests that ran a backward pass caught it.
- VQ decode is **not bit-exact between batch size 1 and >1** on CPU — 4 subpixels
  of 442,368 by one level. Tokens are unaffected; compare images by score, never
  by file hash.
- An available strategy with zero cells crashed the comparison table on a `None`.

---

## Layout

```
src/januscribe/
  config.py      pydantic settings, YAML profiles, device/dtype resolution
  logging.py     structlog setup
  seeding.py     every RNG seeded from one place; per-row generators
  cache.py       KV cache construction, isolated from the sampling loop
  model.py       load once, hold in a singleton; flash-attn detection
  generate.py    CFG sampling; sample_from_prefix shared by every strategy
  understand.py  image + question -> text, plus strict yes/no parsing
  vq.py          image <-> 576 VQ tokens, with PSNR
  subjects.py    Subject registry: canonical description, attributes, seeds, refs
  consistency.py attribute rubric + SigLIP similarity, reported separately
  baseline.py    Tier 0/2 strategies and the resumable grid runner
  inversion.py   M3 textual inversion: one trainable vector, everything frozen
  lora.py        M4 Tier 3: LoRA adapters + general-quality drift probe
  visual_conditioning.py  M4 Tier 4: reference sheet in context
  retry.py       score, resample, never silently ship a failure
  planner.py     topic -> plan; Planner is a Protocol, swap in any text model
  pipeline.py    plan -> scored images -> document, resumable
  assemble.py    self-contained HTML + PDF, debug mode with scores
  cli.py         typer CLI
configs/         default, cuda-7b, subjects, scenes, fox-story
evals/           judge validation, fixed eval set, strategy comparison
NOTES-API.md     the verified Janus-Pro API surface
```

---

## Three findings worth knowing before reading the code

**The model has two input embedding tables.** `prepare_gen_img_embeds` goes
through `gen_embed` (16384×8); the prompt goes through the LM table
(102400×2048). M3's soft token belongs in the *text* table. Putting it in
`gen_embed` trains an 8-d vector no prompt can reference — it runs without error
and produces plausible garbage. [NOTES-API.md](NOTES-API.md) §4.

**Per-image generators, not one batched draw.** The reference sampler calls
`torch.multinomial` once per step over the whole batch, so image *i* depends on
how many images you asked for. Here each row has its own generator seeded
`base_seed + i`, so image *i* is identical alone or in a batch of eight.
Ablations across batch sizes are not comparable otherwise.

**Counts are not measurable by this judge**, so they are excluded from rubrics.
Count drift therefore passes silently: a courier reference image has two cyan
eyes where the description says one, and scores a clean pass. A rubric number
means "are the specified features present", not "is this the same individual".

---

## Hardware

Built and verified CPU-only: ~200 s per 384px image, ~30 s per rubric question,
and **~3.4 minutes per inversion training step**. So a converged soft token is
28–113 hours per subject here.

M1, M2 and the document pipeline run fine on CPU. **M3's hyperparameter sweep
and therefore the headline Tier 0 vs Tier 2 comparison need a GPU.** Nothing in
the code assumes one — `configs/cuda-7b.yaml` is ready — but the schedule should.

## Licences

Janus code is MIT. Janus-Pro **weights** are under the DeepSeek Model License,
which is not MIT — check it before shipping anything built on them.
`assets/doge.png` is copied from the upstream Janus repo.
