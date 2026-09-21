"""Measure the rubric judge against hand-labelled ground truth.

Why this exists as a first-class eval rather than a one-off script: the whole
project compares strategies using the attribute rubric, and an instrument that
agrees with reality only 43% of the time cannot resolve the difference between
Tier 0 and Tier 2. The first version of ``configs/subjects.yaml`` was exactly
that -- two of its six fox attributes were wrong on *every* image, because they
used prepositional binding ("white stripes on the scarf") and a contrastive
clause ("on the forehead rather than over the eyes"). See
``docs/m2/judge-validation.md``.

So attribute phrasings are not prose. They are part of the measuring apparatus,
and they get regression-tested like any other part of it.

    python evals/validate_judge.py                       # every labelled subject
    python evals/validate_judge.py --subject courier,botanist
    python evals/validate_judge.py --dtype bfloat16      # ~4.2 GB instead of ~8.4
    python evals/validate_judge.py --compare "a striped scarf=white stripes on the scarf"

Results append to ``evals/judge_validation.jsonl`` after every question and are
reused on restart, because a full pass costs ~30 s per question on CPU and being
killed part-way must not lose the work.
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


def load_truth(path: Path) -> tuple[dict, dict]:
    """Return ({subject_id: {image: {attribute: label}}}, negative_controls)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "subjects" not in data:
        raise SystemExit(
            f"{path} must have a top-level 'subjects:' mapping of subject id -> images"
        )
    return data["subjects"], data.get("negative_controls", {})


def check_keys_match_config(
    images: dict[str, dict[str, Any]], attributes: list[str], subject_id: str
) -> None:
    """Fail loudly if ground-truth keys have drifted from the config.

    Silent drift is the failure mode that matters: unlabelled attributes would
    simply go unmeasured while the headline agreement still looked healthy.
    """
    configured = set(attributes)
    for image, labels in images.items():
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


def load_done(dtype: str, image_to_subject: dict[str, str]) -> dict[tuple[str, str], dict]:
    """Load cached answers, backfilling fields added after they were written.

    Rows from an earlier schema have no ``subject``, which silently dropped a
    whole subject out of the per-subject breakdown while the overall number
    still looked right. Cached rows get the missing field filled in from the
    ground truth rather than being trusted as-is.
    """
    if not RESULTS_FILE.exists():
        return {}
    done = {}
    for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dtype") != dtype:
            continue
        if "subject" not in row:
            row["subject"] = image_to_subject.get(row["image"], "_unknown")
        done[(row["image"], row["phrasing"], row.get("tag", "current"))] = row
    return done


def emit(row: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", default="float32", help="float32 | bfloat16 | float16")
    parser.add_argument("--subject", default=None, help="Comma-separated ids; default all labelled.")
    parser.add_argument("--subjects-file", type=Path, default=SUBJECTS_FILE)
    parser.add_argument("--truth", type=Path, default=TRUTH_FILE)
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="NEW=OLD -- also ask the old phrasing, scored against the same label.",
    )
    args = parser.parse_args()

    truth, negatives = load_truth(args.truth)
    registry = SubjectRegistry.from_yaml(args.subjects_file)

    wanted = args.subject.split(",") if args.subject else list(truth)
    unknown = [s for s in wanted if s not in truth]
    if unknown:
        raise SystemExit(f"no ground truth for {unknown}; labelled subjects are {sorted(truth)}")

    alternatives = dict(pair.split("=", 1) for pair in args.compare)
    bundle = get_bundle(Settings(dtype=args.dtype, device="cpu"))
    cfg = UnderstandConfig(max_new_tokens=24, temperature=0.0)
    image_to_subject = {
        image: sid for sid, images in truth.items() for image in images
    }
    done = load_done(args.dtype, image_to_subject)

    rows: list[dict] = []
    for subject_id in wanted:
        subject = registry.get(subject_id)
        images = truth[subject_id]
        check_keys_match_config(images, subject.attributes, subject_id)

        for image_rel, labels in images.items():
            image_path = REPO_ROOT / image_rel
            if not image_path.exists():
                log.warning("image_missing", path=str(image_path))
                continue

            for attribute, label in labels.items():
                phrasings = [(attribute, "current")]
                if attribute in alternatives:
                    phrasings.append((alternatives[attribute], "alternative"))
                for phrasing, tag in phrasings:
                    cached = done.get((image_rel, phrasing, tag))
                    if cached:
                        rows.append(cached)
                        continue
                    verdict, answer = ask_yes_no(bundle, image_path, phrasing, cfg=cfg)
                    row = {
                        "dtype": args.dtype, "subject": subject_id, "image": image_rel,
                        "attribute": attribute, "phrasing": phrasing, "tag": tag,
                        "truth": label, "verdict": verdict, "raw": answer.text,
                        "correct": None if label is None else (verdict == "yes") == bool(label),
                    }
                    emit(row)
                    rows.append(row)
                    mark = "--" if row["correct"] is None else ("ok" if row["correct"] else "XX")
                    print(
                        f"{mark} {subject_id:9} {Path(image_rel).stem:18} {phrasing:40} -> {verdict}",
                        flush=True,
                    )

    negative_rows: list[dict] = []
    if not args.subject:  # negative controls span subjects, so only on a full run
        for image_rel, labels in negatives.items():
            image_path = REPO_ROOT / image_rel
            if not image_path.exists():
                log.warning("image_missing", path=str(image_path))
                continue
            for attribute, label in labels.items():
                cached = done.get((image_rel, attribute, "negative"))
                if cached:
                    negative_rows.append(cached)
                    continue
                verdict, answer = ask_yes_no(bundle, image_path, attribute, cfg=cfg)
                row = {
                    "dtype": args.dtype, "subject": "_negative", "image": image_rel,
                    "attribute": attribute, "phrasing": attribute, "tag": "negative",
                    "truth": label, "verdict": verdict, "raw": answer.text,
                    "correct": (verdict == "yes") == bool(label),
                }
                emit(row)
                negative_rows.append(row)
                mark = "ok" if row["correct"] else "XX"
                print(
                    f"{mark} NEGATIVE  {Path(image_rel).stem:18} {attribute:40} -> {verdict}",
                    flush=True,
                )

    return report(rows, registry, wanted, negative_rows)


def report(
    rows: list[dict],
    registry: SubjectRegistry,
    wanted: list[str],
    negative_rows: list[dict] | None = None,
) -> int:
    scored = [r for r in rows if r["correct"] is not None and r["tag"] == "current"]
    if not scored:
        print("no scored rows")
        return 1

    print("\n=== judge validation ===")
    failed_subjects: list[tuple[str, float]] = []

    for subject_id in wanted:
        sub = [r for r in scored if r.get("subject") == subject_id]
        if not sub:
            # A requested subject producing no rows means it silently went
            # unmeasured -- the exact failure this eval exists to prevent, so it
            # is an error rather than a skipped line.
            print(f"\n[FAIL] {subject_id}: no scored rows. Unmeasured, not passing.")
            failed_subjects.append((subject_id, 0.0))
            continue
        hits = sum(1 for r in sub if r["correct"])
        agreement = hits / len(sub)
        skipped = sum(
            1 for r in rows if r.get("subject") == subject_id and r["correct"] is None
        )
        flag = "PASS" if agreement >= MIN_AGREEMENT else "FAIL"
        print(f"\n[{flag}] {subject_id}: {hits}/{len(sub)} = {agreement:.3f}"
              f"  (ambiguous, unscored: {skipped})")

        # Positive and negative cases separately: a judge that only ever sees
        # attributes that ARE present is not characterised, because agreeing is
        # its failure mode.
        pos = [r for r in sub if r["truth"]]
        neg = [r for r in sub if not r["truth"]]
        if pos:
            print(f"    present   {sum(1 for r in pos if r['correct'])}/{len(pos)}")
        if neg:
            print(f"    absent    {sum(1 for r in neg if r['correct'])}/{len(neg)}")
        else:
            print("    absent    none labelled -- false positives are untested here")

        for attribute in registry.get(subject_id).attributes:
            per = [r for r in sub if r["attribute"] == attribute]
            if per and sum(1 for r in per if r["correct"]) < len(per):
                good = sum(1 for r in per if r["correct"])
                print(f"    unreadable: {attribute!r} {good}/{len(per)}")

        if agreement < MIN_AGREEMENT:
            failed_subjects.append((subject_id, agreement))

    hits = sum(1 for r in scored if r["correct"])
    overall = hits / len(scored)
    print(f"\noverall (per-subject sheets): {hits}/{len(scored)} = {overall:.3f}")

    negative_rows = negative_rows or []
    if negative_rows:
        neg_hits = sum(1 for r in negative_rows if r["correct"])
        rate = neg_hits / len(negative_rows)
        print(f"negative controls: {neg_hits}/{len(negative_rows)} = {rate:.3f}")
        for r in negative_rows:
            if not r["correct"]:
                print(
                    f"    FALSE POSITIVE  {Path(r['image']).stem}: "
                    f"claimed {r['attribute']!r} is present"
                )
        if rate < MIN_AGREEMENT:
            failed_subjects.append(("negative controls", rate))
    else:
        print("negative controls: NOT RUN -- false-positive rate is unmeasured")

    alt = [r for r in rows if r["tag"] == "alternative" and r["correct"] is not None]
    if alt:
        alt_hits = sum(1 for r in alt if r["correct"])
        print(f"alternative phrasings: {alt_hits}/{len(alt)} = {alt_hits / len(alt):.3f}")

    if failed_subjects:
        print("\nFAIL: these subjects are below the threshold, so a baseline using them")
        print("cannot be trusted to resolve one strategy from another:")
        for subject_id, agreement in failed_subjects:
            print(f"  {subject_id}: {agreement:.3f} < {MIN_AGREEMENT}")
        print("Rewrite the named attributes per the rules in configs/subjects.yaml.")
        return 1

    print(f"\nPASS: every subject at or above {MIN_AGREEMENT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
