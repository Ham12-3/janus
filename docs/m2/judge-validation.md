# Validating the rubric judge

## Why this document exists

M2's attribute rubric asks Janus-Pro's understanding pathway one strict yes/no
question per atomic attribute and scores the fraction answered yes. Every later
milestone is compared against that number. So before running the baseline, the
question is not "what does Tier 0 score" but **"can this instrument measure
anything at all"**.

It could not, at first. The finding below cost about an hour of CPU and saved a
two-hour baseline run that would have produced a confidently wrong number.

## How it surfaced

The M2 smoke run scored a generated fox image **2/6**. Looking at the image, the
truth is **5/6** — the scarf plainly has white stripes, the goggles are plainly
round and brass, and they are plainly on the forehead. The judge said no to all
three.

![the image the judge scored 2/6](judged_2of6.png)

Had the full grid run at that point, Tier 0 would have reported a rubric score
near 0.33 against a real value near 0.83, and every subsequent tier would have
been measured against the judge's weakness rather than the generator's
inconsistency.

## The experiment

Five hand-labelled fox images (four reference sheet, one scene). Each attribute
asked two ways — the phrasing then in `configs/subjects.yaml`, and a simpler
alternative — plus open-ended description questions to separate *perception*
from *question format*.

Ground truth: all five images show a red fox in a white-striped navy scarf with
round brass goggles on the forehead, and none has a torn ear. So 5/6 for every
image.

## Result

| phrasing set | agreement with ground truth |
|---|---|
| original (`configs/subjects.yaml` v1) | **11/18 = 0.611** |
| simplified | **17/18 = 0.944** |

Per attribute:

| attribute | original | simplified |
|---|---|---|
| a red fox | 3/3 | 3/3 |
| a navy blue scarf | 3/3 | 3/3 |
| white stripes **on the scarf** | **0/3** | "a striped scarf" → 3/3 |
| goggles on the forehead **rather than** over the eyes | **0/3** | "goggles on top of the head" → 3/3 |
| round brass-coloured goggles | 2/3 | "goggles" → 3/3 |
| a fox with one torn or notched ear | 3/3 | "a damaged ear" → **2/3** |

Two attributes were wrong on **every single image**. That is not noise; it is a
deterministic wrong answer to a badly formed question.

## It is question format, not perception

The decisive evidence is the open-ended answers. Asked to *describe* rather than
confirm, the same model on the same image says:

> "The scarf in the image is predominantly blue with **white stripes**."

> "Yes, the animal is wearing **goggles on its head**."

The model perceives the exact facts it denies under the binary question. Nothing
is wrong with Janus-Pro-1B's vision here; the questions were malformed.

## The rules this produced

Now documented at the top of `configs/subjects.yaml` and enforced by
`evals/validate_judge.py`:

1. **One visual fact per question.** A bundled question's "no" tells you nothing
   about which part failed.
2. **No prepositional binding to a second object.** `white stripes ON THE SCARF`
   → 0/3; `a striped scarf` → 3/3. Fold the relationship into the head noun.
3. **No contrastive or negated clauses.** `on the forehead RATHER THAN over the
   eyes` → 0/3; `goggles on top of the head` → 3/3.
4. **Stay specific enough to be falsifiable.** Simplification is not uniformly
   safe: `a damaged ear` produced a false *positive* on an undamaged ear, where
   the specific `a fox with one torn or notched ear` correctly said no. Vague
   questions do not get truer answers, they get agreeable ones.
5. **No counts.** `six small black wheels` asks a 1B VLM to count, which it does
   not do reliably. Counts stay in the canonical description, where they steer
   generation, and out of the attribute list, where they would only add noise.

Rules 2 and 4 pull in opposite directions. That tension is the reason this had
to be measured rather than reasoned about.

## Confirmation in fp32, on the rewritten attributes

The sweep above ran in **bfloat16** to halve resident memory on a 16 GB machine
(~4.2 GB against ~8.4 GB in fp32; two earlier fp32 attempts were killed
mid-run). bf16 matmul is emulated on this CPU, so it is not automatically
equivalent to the fp32 the baseline actually runs in.

`evals/validate_judge.py` was therefore re-run in **fp32** against the rewritten
attribute set, asking the retired phrasings alongside for a direct comparison on
identical images:

| attribute set (fp32) | agreement |
|---|---|
| **rewritten** (current `configs/subjects.yaml`) | **29/29 = 1.000** |
| retired phrasings | **6/14 = 0.429** |

Per retired phrasing:

| retired phrasing | correct |
|---|---|
| `white stripes on the scarf` | **0/5** |
| `goggles resting on the forehead rather than over the eyes` | **2/4** |
| `round brass-coloured goggles` | 4/5 |

Two things worth noting. First, the finding transfers: the failures are the same
ones bf16 found, so the bf16 sweep was a valid diagnostic. Second, in fp32 —
the precision the baseline actually runs in — the original rubric was **worse
than the bf16 numbers suggested**, at 0.429 across those attributes. A rubric
that disagrees with reality more often than it agrees cannot resolve Tier 0 from
Tier 2 at all.

Two images are excluded as ambiguous (`null` in the ground truth) and are not
scored either way. Raw rows: `evals/judge_validation.jsonl`.

## Standing policy

Attribute phrasings are part of the measuring apparatus, not prose. Whenever
`configs/subjects.yaml` attributes change:

```bash
python evals/validate_judge.py --dtype float32
```

It fails below **0.90 agreement**, names the attributes the judge cannot read,
and refuses to run if the ground-truth keys have drifted from the config —
silent drift would leave attributes unmeasured while the headline agreement
still looked healthy.

Ground truth lives in `evals/judge_truth.yaml` and is the only thing in this
project not derived from a model, which is exactly why it can be used to measure
one. `null` marks a genuinely ambiguous case and is excluded from scoring: an
honest abstention rather than a free pass.

## Not yet validated

`courier` and `botanist` have had the rules applied but have **no reference
sheets to label against yet**, so their agreement is unmeasured. Their numbers in
any baseline should be read with that caveat until they get the same treatment.
