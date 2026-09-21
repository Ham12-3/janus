"""Measure the rubric judge against hand-labelled ground truth.

Why this exists as a first-class eval rather than a one-off script: the whole
project compares strategies using the attribute rubric, and an instrument that
agrees with reality only 69% of the time cannot resolve the difference between
Tier 0 and Tier 2. The first version of ``configs/subjects.yaml`` was exactly
that -- two of its six fox attributes were wrong on *every* image, because they
used prepositional binding ("white stripes on the scarf") and a contrastive
clause ("on the forehead rather than over the eyes").

So attribute phrasings are not prose. They are part of the measuring apparatus,
and they get regression-tested like any other part of it.

Run it whenever ``configs/subjects.yaml`` attributes change:

    python evals/validate_judge.py                      # validate current config
    python evals/validate_judge.py --dtype bfloat16     # half the memory, ~4.2 GB
    python evals/validate_judge.py --compare "a striped scarf=white stripes on the scarf"

Results append to ``evals/judge_validation.jsonl`` after every question and are
reused on restart, because a full pass costs about a minute per question on CPU
and being killed part-way must not lose the work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from januscribe.config import Settings, UnderstandConfig  # noqa: E402
from januscribe.logging import get_logger  # noqa: E402
from januscribe.model import get_bundle  # noqa: E402
from januscribe.subjects import SubjectRegistry  # noqa: E402
from januscribe.understand import ask_yes_no  # noqa: E402

log = get_logger(__name__)

TRUTH_FILE = REPO_ROOT / "evals" / "judge_truth.yaml"
RESULTS_FILE = REPO_ROOT / "evals" / "judge_validation.jsonl"
SUBJECTS_FILE = REPO_ROOT / "configs" / "subjects.yaml"

# Agreement below this means the rubric cannot resolve the differences the
# project exists to measure, and any baseline built on it is not trustworthy.
MIN_AGREEMENT = 0.90


def load_truth(path: Path) -> tuple[str, dict[str, dict[str, bool | None]]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["subject"], data["images"]


def check_keys_match_config(
    truth: dict[str, dict[str, Any]], attributes: list[str], subject_id: str
) -> None:
    """Fail loudly if the ground-truth keys have drifted from the config.

    Silent drift is the failure mode that would matter here: unlabelled
    attributes would simply go unmeasured and the agreement number would look
    fine while covering less than it claims.
    """
    configured = set(attributes)
    for image, labels in truth.items():
        labelled = set(labels)
        missing = configured - labelled
        extra = labelled - configured
        if missing or extra:
            raise SystemExit(
                f"ground truth for {image} does not match configs/subjects.yaml "
                f"[{subject_id}].\n  unlabelled attributes: {sorted(missing)}\n"
                f"  stale labels: {sorted(extra)}\n"
                f"Update {TRUTH_FILE.name} by looking at the image, not by guessing."
            )


def load_done(dtype: str) -> dict[tuple[str, str], dict]:
    if not RESULTS_FILE.exists():
        return {}
    done = {}
    for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dtype") == dtype:
            done[(row["image"], row["attribute"])] = row
    return done


def emit(row: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", default="float32", help="float32 | bfloat16 | float16")
    parser.add_argument("--subjects", type=Path, default=SUBJECTS_FILE)
    parser.add_argument("--truth", type=Path, default=TRUTH_FILE)
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="NEW=OLD -- also ask the old phrasing, scored against the same label.",
    )
    args = parser.parse_args()

    subject_id, truth = load_truth(args.truth)
    registry = SubjectRegistry.from_yaml(args.subjects)
    subject = registry.get(subject_id)
    check_keys_match_config(truth, subject.attributes, subject_id)

    alternatives = dict(pair.split("=", 1) for pair in args.compare)
    bundle = get_bundle(Settings(dtype=args.dtype, device="cpu"))
    cfg = UnderstandConfig(max_new_tokens=24, temperature=0.0)
    done = load_done(args.dtype)

    rows: list[dict] = []
    for image_rel, labels in truth.items():
        image_path = REPO_ROOT / image_rel
        if not image_path.exists():
            log.warning("image_missing", path=str(image_path))
            continue

        for attribute, label in labels.items():
            for phrasing, tag in [(attribute, "current")] + (
                [(alternatives[attribute], "alternative")] if attribute in alternatives else []
            ):
                cached = done.get((image_rel, phrasing))
                if cached:
                    rows.append(cached)
                    continue
                verdict, answer = ask_yes_no(bundle, image_path, phrasing, cfg=cfg)
                row = {
                    "dtype": args.dtype, "image": image_rel, "attribute": attribute,
                    "phrasing": phrasing, "tag": tag, "truth": label,
                    "verdict": verdict, "raw": answer.text,
                    "correct": None if label is None else (verdict == "yes") == bool(label),
                }
                emit(row)
                rows.append(row)
                mark = "--" if row["correct"] is None else ("ok" if row["correct"] else "XX")
                print(f"{mark} {Path(image_rel).stem:18} {phrasing:42} -> {verdict}", flush=True)

    return report(rows, subject.attributes)


def report(rows: list[dict], attributes: list[str]) -> int:
    scored = [r for r in rows if r["correct"] is not None and r["tag"] == "current"]
    if not scored:
        print("no scored rows")
        return 1

    hits = sum(1 for r in scored if r["correct"])
    agreement = hits / len(scored)
    skipped = sum(1 for r in rows if r["correct"] is None)

    print("\n=== judge validation ===")
    print(f"agreement: {hits}/{len(scored)} = {agreement:.3f}  (ambiguous, unscored: {skipped})")
    print(f"\n{'attribute':45} {'correct':>9}")
    failing = []
    for attribute in attributes:
        sub = [r for r in scored if r["attribute"] == attribute]
        if not sub:
            continue
        good = sum(1 for r in sub if r["correct"])
        print(f"{attribute:45} {good:>4}/{len(sub)}")
        if good < len(sub):
            failing.append((attribute, good, len(sub)))

    alt = [r for r in rows if r["tag"] == "alternative" and r["correct"] is not None]
    if alt:
        alt_hits = sum(1 for r in alt if r["correct"])
        print(f"\nalternative phrasings: {alt_hits}/{len(alt)} = {alt_hits / len(alt):.3f}")

    if failing:
        print("\nattributes the judge cannot read reliably:")
        for attribute, good, total in failing:
            print(f"  {attribute!r}: {good}/{total}")
        print("Rewrite them per the rules at the top of configs/subjects.yaml.")

    if agreement < MIN_AGREEMENT:
        print(
            f"\nFAIL: agreement {agreement:.3f} is below {MIN_AGREEMENT}. "
            "A rubric this noisy cannot resolve Tier 0 from Tier 2; fix the phrasings "
            "before trusting any baseline built with it."
        )
        return 1
    print(f"\nPASS: agreement {agreement:.3f} >= {MIN_AGREEMENT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
