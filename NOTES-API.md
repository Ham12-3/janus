# NOTES-API.md — the real Janus-Pro API surface

Everything below was read out of `vendor/Janus` at commit `1daa72f` (2025-02-01)
and then **verified by running it** against `deepseek-ai/Janus-Pro-1B` on this
machine (CPU, fp32, torch 2.14.0+cpu, transformers 4.45.2, Python 3.11.15).
Where the project brief's assumed API differs from the source, the correction is
called out explicitly.

---

## 1. Verdict on each assumed API

| Assumption in the brief | Verdict | Reality |
|---|---|---|
| `VLChatProcessor.from_pretrained(path)`, `.tokenizer`, `.image_start_tag` | correct | All three exist. `image_start_tag == "<begin_of_image>"`. |
| `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` | correct, with a caveat | Works only because importing `janus.models` registers `MultiModalityConfig` with `AutoConfig`/`AutoModelForCausalLM`. There is **no remote code in the Hub repo**, so `trust_remote_code=True` is inert — the locally installed `janus` package is what resolves `model_type="multi_modality"`. Forget the import and you get "unrecognised model type". |
| `mmgpt.prepare_inputs_embeds(...)` for understanding | correct | Signature is `(input_ids, pixel_values, images_seq_mask, images_emb_mask, **kwargs)`. |
| `mmgpt.prepare_gen_img_embeds(token_ids)` | correct | But see §4 — it does **not** touch the LM embedding table. |
| `mmgpt.gen_head(hidden)` | correct | `hidden [B, 2048] -> logits [B, 16384]`. |
| `mmgpt.gen_vision_model.decode_code(...)` | correct | Requires an explicit `shape=[B, 8, 24, 24]`. |
| "an encode path on `gen_vision_model`" | exists, but the return shape is awkward | `gen_vision_model.encode(x) -> (quant, losses, info)`. The token ids are `info[2]`, **flattened** to `[B*576]`, not `[B, 576]`. |
| "DeepSeek-LLM-1.5b-base" | slightly off | Janus-Pro-1B's `language_config` is 24 layers x hidden 2048 x vocab 102400, i.e. the **1.3B** DeepSeek-LLM architecture. Whole model = **2,089,232,011** params including SigLIP and VQ. |

Nothing in the plan is invalidated. Two things genuinely change how M3 must be
built — see §4 and §7.

---

## 2. Verified constants (Janus-Pro-1B `config.json`)

```
gen_vision_config.cls                     = "VQ-16"
gen_vision_config.params.image_token_size = 16384   # codebook size
gen_vision_config.params.n_embed          = 8       # codebook_embed_dim
gen_head_config.params.n_embed            = 2048    # LM hidden
gen_head_config.params.image_token_embed  = 2048
gen_head_config.params.image_token_size   = 16384   # output logits
vision_config.params.model_name           = "siglip_large_patch16_384"
language_config: hidden 2048, layers 24, heads 16, vocab_size 102400, bf16
```

Observed at runtime (`januscribe info`):

```
params_total       2089232011
image_token_size   16384
codebook_embed_dim 8
lm_vocab_size      102400
lm_hidden_size     2048
```

Tokenizer ids, read off the loaded processor rather than guessed:

| name | tag | id |
|---|---|---|
| `pad_id` | `<｜▁pad▁｜>` | 100002 |
| `image_start_id` | `<begin_of_image>` | 100003 |
| `image_end_id` | `<end_of_image>` | 100580 |
| `image_id` | `<image_placeholder>` | 100581 |
| `num_image_tokens` | — | 576 |
| `sft_format` | — | `"deepseek"` |

`image_start_id` (100003) is **not** adjacent to `image_end_id` (100580). Do not
compute one from the other.

---

## 3. Generation pathway (text to image), exact call sequence

Verified end to end. Two 384px images took 360 s on this CPU (~180 s/image).

```python
# 1. Prompt: SFT template, then the begin-of-image tag. That tag is what switches
#    the model from text continuation into image-code generation.
sft = processor.apply_sft_template_for_multi_turn_prompts(
    conversations=[{"role": "User", "content": prompt},
                   {"role": "Assistant", "content": ""}],
    sft_format=processor.sft_format, system_prompt="")
full = sft + processor.image_start_tag
input_ids = torch.LongTensor(processor.tokenizer.encode(full))

# 2. CFG batch: rows interleave [cond, uncond, cond, uncond, ...].
#    The uncond row keeps token 0 (BOS) and the final token (begin_of_image)
#    and pads everything in between with pad_id.
tokens = input_ids.repeat(2 * parallel_size, 1)
tokens[1::2, 1:-1] = processor.pad_id

# 3. Text embeddings come from the LM's own input embedding table.
inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

# 4. 576 autoregressive steps.
for step in range(576):
    out = mmgpt.language_model.model(inputs_embeds=inputs_embeds,
                                     use_cache=True, past_key_values=cache)
    cache = out.past_key_values
    logits = mmgpt.gen_head(out.last_hidden_state[:, -1, :])      # [2N, 16384]
    guided = logits[1::2] + cfg_weight * (logits[0::2] - logits[1::2])
    next_token = sample(softmax(guided / temperature))            # [N]
    inputs_embeds = mmgpt.prepare_gen_img_embeds(
        next_token.repeat_interleave(2)).unsqueeze(1)             # [2N, 1, 2048]

# 5. Decode.
dec = mmgpt.gen_vision_model.decode_code(codes.int(), shape=[N, 8, 24, 24])
img = clip((dec + 1) / 2 * 255, 0, 255).astype(uint8)             # NCHW -> NHWC
```

Points that are easy to get wrong:

- Call `mmgpt.language_model.model(...)` (the inner `LlamaModel`), **not**
  `mmgpt.language_model(...)`. The latter applies `lm_head` over the 102400-entry
  text vocab. Image logits come from `gen_head` over 16384 codes instead.
- No `attention_mask` is passed. The unconditional row is expressed by
  *substituting* `pad_id` in place, not by masking.
- `gen_head`'s output space (16384 VQ codes) is disjoint from the LM's text vocab
  (102400). Two different heads over the same hidden state.
- The decoder emits `[-1, 1]`, not `[0, 1]`.

### Deviations JanusScribe makes from the reference loop

1. **Per-image generators.** The reference calls `torch.multinomial` once over the
   whole batch, so image *i*'s pixels depend on `parallel_size`. JanusScribe gives
   each row its own generator seeded `base_seed + i` and samples row by row, so
   image *i* is identical whether sampled alone or in a batch of eight. Ablation
   numbers are not comparable otherwise. The cost is negligible next to the
   transformer forward. Proved by `tests/test_generation_determinism.py`.
2. **`DynamicCache` instead of `None`.** Seeding the loop with `None` makes
   transformers 4.45 round-trip the legacy tuple-of-tuples cache and emit a
   removal warning. Passing a real `Cache` keeps the loop on the supported path.
3. **No implicit `.cuda()`.** Device and dtype come from config.

---

## 4. The finding that shapes M3: there are two input embedding tables

```python
def prepare_gen_img_embeds(self, image_ids):
    return self.gen_aligner(self.gen_embed(image_ids))
```

`self.gen_embed` is `nn.Embedding(16384, 8)`, owned by `MultiModalityCausalLM`
and **entirely separate** from `language_model.get_input_embeddings()`
(`nn.Embedding(102400, 2048)`). The model has two input vocabularies feeding one
transformer:

| | table | shape | used for |
|---|---|---|---|
| text | `language_model.model.embed_tokens` | 102400 x 2048 | the prompt, including any soft token |
| image codes | `mmgpt.gen_embed` then `gen_aligner` | 16384 x 8 -> 2048 | previously generated image codes |

**Consequence for M3.** The plan is still right, but only if the new row goes in
the *text* table, because `<sbj_0>` appears in the prompt:

- The new row is index 102400 in a resized 102401 x 2048 table, and its gradient
  is the only one enabled. `gen_embed` stays frozen.
- `resize_token_embeddings` also grows `lm_head`. Check
  `config.tie_word_embeddings` before assuming the two are linked. The inversion
  loss never touches the text head, so an untrained `lm_head` row is harmless,
  but it must not receive gradient either — the `requires_grad` count assertion
  in the brief will catch it if it does.
- "Initialise from the mean embedding of the subject's noun" means the mean over
  the *text* table's rows for the tokenisation of `" fox"` — a 2048-d vector, not
  an 8-d codebook vector.

Adding the row to `gen_embed` instead would train an 8-d vector in VQ codebook
space that no prompt can ever reference. It would run without error and produce
plausible-looking garbage, which is exactly the failure mode to avoid.

---

## 5. VQ tokenizer: the encode path, verified

From `janus/models/vq_model.py`:

```python
def encode(self, x):            # x: [B, 3, 384, 384] in [-1, 1]
    h = self.encoder(x)         # ch_mult [1,1,2,2,4] -> 4 downsamples -> /16
    h = self.quant_conv(h)      # -> [B, 8, 24, 24]
    quant, emb_loss, info = self.quantize(h)
    return quant, emb_loss, info
#   info == (perplexity, min_encodings, min_encoding_indices)
#   min_encoding_indices: [B*576] int64   <-- the tokens
```

So `tokens = gen_vision_model.encode(x)[2][2].reshape(B, 576)`.

The codebook is **L2-normalised** (`codebook_l2_norm=True`) and
`get_codebook_entry` re-normalises on the way out, so encode and decode stay
consistent as long as you do not hand-roll the lookup.

**Measured roundtrip** (`assets/doge.png`, 384x384, fp32):

```
n_tokens 576   grid 24x24   PSNR 22.09 dB   unique codes 310 / 16384
token range [51, 16351]     deterministic across runs: True
```

22 dB with garbled small text is the correct signature for an f16 VQ tokenizer —
it really is lossy at glyph scale. A broken pipeline (wrong pixel range,
transposed grid, fp16 underflow) lands in single digits, which is why the test
floor sits at 17 dB. Side-by-side proof: [`docs/m1/vq_doge_roundtrip.png`](docs/m1/vq_doge_roundtrip.png).

`decode_code(code_b, shape)` wants **flat** codes `[B*576]` plus
`shape=[B, 8, 24, 24]`; it reshapes internally with `channel_first=True`.

### Measured: VQ decode is not bit-exact across batch size 1 vs >1

Decoding the same 576 codes alone and inside a batch gives images that differ by
**4 subpixels out of 442368, by 1 level (98.57 dB)**. Repeated calls at the same
batch size are bit-identical, rows inside one batch are bit-identical, and batch
2 against batch 4 is bit-identical -- so the discontinuity is specifically
torch's `N == 1` convolution path on CPU, not accumulated drift.

Consequences:

- Sampled **tokens are unaffected**. Image *i* produces the same 576 codes at any
  `parallel_size` (`tests/test_generation_determinism.py` pins this down).
- Decoded **PNG bytes can differ** between a solo run and a batched run. Compare
  generated images by score, never by file hash, and do not make a golden-file
  test out of rendered pixels.

---

## 6. Understanding pathway (image + text to text), verified

```python
prepare = processor(conversations=conv, images=pil_images, force_batchify=True)
prepare = prepare.to(device, dtype=dtype)
inputs_embeds = mmgpt.prepare_inputs_embeds(**prepare)
out = mmgpt.language_model.generate(inputs_embeds=inputs_embeds,
                                    attention_mask=prepare.attention_mask, ...)
```

- The conversation `content` must contain one `<image_placeholder>` per image.
  The processor expands each into `<begin_of_image>` + 576 x `<image_placeholder>`
  + `<end_of_image>`.
- `batchify` **left-pads**, so `attention_mask` matters here, unlike generation.
- `prepare_inputs_embeds` mutates `input_ids` in place
  (`input_ids[input_ids < 0] = 0`). Do not reuse that tensor afterwards.
- Text-only prompts still run the SigLIP tower over a zero image, because
  `batchify` allocates `max(1, n_images)` pixel slots. Harmless but wasteful —
  relevant to M5 if Janus is used for pure-text planning.

Measured: ~20 s per short answer on this CPU.

---

## 7. Environment findings (these cost real time; do not re-derive them)

- **`attrdict` is dead.** It imports `collections.Mapping`, removed in Python 3.10.
  `vendor/Janus/pyproject.toml` requires it, so a plain
  `pip install -e ./vendor/Janus` bricks the install on any modern Python. Fix:
  install the vendored package with `--no-deps` and depend on **`attrdict3`**,
  a maintained fork that patches the `collections` module at import time.
  This is why the brief's step 0 command needs the `--no-deps` flag.
- **transformers is pinned `>=4.38.2,<4.46`.** 4.45.2 is verified. Two known
  breakages above that line: `ProcessorMixin.__init__` tightened its argument
  validation (`VLChatProcessor` passes 8 positional args for 2 declared
  attributes and relies on `zip` truncating silently), and the legacy tuple cache
  the reference loop depends on is slated for removal in 4.47.
- **`tokenizers` pinned `<0.21`** to stay inside that transformers range.
- **Attention implementation must be set on the inner `LlamaConfig`**, not passed
  to `from_pretrained`. `MultiModalityCausalLM` constructs
  `LlamaForCausalLM(config.language_config)` itself, and the outer
  `MultiModalityPreTrainedModel` declares no attention support, so
  `attn_implementation="sdpa"` at the top level raises. Do
  `config.language_config._attn_implementation = "sdpa"` instead. The upstream
  demo `app_januspro.py` does the same thing with `"eager"`.
- **flash-attn is optional and unused here.** This machine is CPU-only, so
  `resolve_attn_implementation` picks `sdpa`. On CUDA without flash-attn
  installed it warns once and falls back to `sdpa`. Install never depends on it.
- **Weights are `pytorch_model.bin`, not safetensors** — 3.9 GB for Janus-Pro-1B.
- **dtype policy.** `auto` gives bf16 on CUDA, fp16 on MPS, **fp32 on CPU**
  (bf16 matmul on consumer Intel is emulated: slower *and* lossier). The VQ
  tokenizer is separately forced to fp32 by default; at ~70M params that costs
  little memory and keeps dtype noise out of roundtrip PSNR.

---

## 8. Hardware reality on this machine

No CUDA device (Intel Iris Xe iGPU only), 16 GB RAM, i5-1135G7, 8 logical cores.
Everything runs CPU fp32. Measured: model load 28-59 s, VQ roundtrip ~11 s,
understanding answer ~20 s, and **image generation ~180 s per 384px image**.

Workable for M1 and M2 spot checks. The M2 baseline report (20 scenes x 3
subjects = 60 images, plus ~6 rubric questions each) would take roughly 5 hours
of generation plus 2 hours of scoring here, and the M6 sweep across four
strategies is firmly a CUDA-box job. Nothing in the code assumes a GPU, but the
schedule should.
