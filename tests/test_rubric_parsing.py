"""Yes/no parsing for the M2 attribute rubric. No model needed.

An ambiguous answer must come back 'unclear' rather than quietly scoring as a
pass -- a rubric that guesses is not a measurement.
"""

from __future__ import annotations

import pytest

from januscribe.understand import Answer, parse_yes_no


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yes", "yes"),
        ("Yes.", "yes"),
        ("Yes, the fox wears a navy scarf.", "yes"),
        ("no", "no"),
        ("No.", "no"),
        ("No, there are no goggles.", "no"),
        ("The fox is not wearing goggles.", "no"),
        ("Yeah, definitely.", "yes"),
        ("", "unclear"),
        ("   ", "unclear"),
        ("It is a fox.", "unclear"),
        ("Maybe, hard to tell.", "unclear"),
    ],
)
def test_parse_yes_no(text: str, expected: str) -> None:
    assert parse_yes_no(text) == expected


def test_first_decisive_word_wins() -> None:
    assert parse_yes_no("Yes, but it is not blue.") == "yes"
    assert parse_yes_no("No, yes it is absent.") == "no"


def test_answer_object_delegates() -> None:
    assert Answer(question="q", text="Yes, it does.", n_images=1).as_yes_no() == "yes"
