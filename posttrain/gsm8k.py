"""GSM8K benchmarking helpers (Problem 7).

PROVIDED -- do not modify: standard_prompt_template, extract_answer, is_correct.
YOUR TASK:                 improved_prompt_template.

A prompt template maps a GSM8K question to a list of chat messages.  scripts/eval_gsm8k.py
renders the messages with the model's chat template, decodes greedily, stops after the
first "</answer>", and scores the completion with extract_answer + is_correct.  The
autograder re-scores your predictions.jsonl with its own copy of these two functions.

extract_answer is lenient, like the "flexible extract" rule of common GSM8K harnesses:
it reads the last <answer>...</answer> block, else the last \boxed{...}, else the last
number anywhere in the completion.  has_explicit_answer tells whether the model gave an
explicit final answer (<answer> or \boxed); its mean over the problems is the format rate.
"""

from __future__ import annotations

import re
from typing import Optional

ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def standard_prompt_template(question: str) -> list[dict]:
    """The standard prompt used for Problem 7(a)."""
    return [{
        "role": "user",
        "content": f"{question}\nRespond with only the final answer in the form <answer>NUMBER</answer>.",
    }]


def _first_number(text: str) -> Optional[str]:
    nums = NUMBER.findall(text.replace(",", ""))
    return nums[0] if nums else None


def extract_answer(completion: str) -> Optional[str]:
    """The standard output extractor.  In order of preference:

    1. the first number inside the LAST <answer>...</answer> block;
    2. the first number inside the LAST \\boxed{...};
    3. the LAST number anywhere in the completion.

    Thousands separators are removed first, so "<answer>$1,200</answer>" gives "1200".
    Returns None only when the completion contains no number at all.
    """
    for pattern in (ANSWER_TAG, BOXED):
        blocks = pattern.findall(completion)
        if blocks and _first_number(blocks[-1]) is not None:
            return _first_number(blocks[-1])
    nums = NUMBER.findall(completion.replace(",", ""))
    return nums[-1] if nums else None


def has_explicit_answer(completion: str) -> bool:
    """True if the completion marks a final answer with <answer>...</answer> or \\boxed{...}
    containing a number (used for the format rate; it does not affect accuracy)."""
    return any(_first_number(b) is not None
               for pattern in (ANSWER_TAG, BOXED) for b in pattern.findall(completion))


def is_correct(prediction: Optional[str], gold: str) -> bool:
    """Numeric match: "18", "18.0" and "18.00" all equal the gold answer "18"."""
    if prediction is None:
        return False
    return abs(float(prediction) - float(gold)) < 1e-6


def improved_prompt_template(question: str) -> list[dict]:
    """YOUR prompt for Problem 7(b).  Return a list of chat messages ({"role": ..., "content": ...});
    you may add a system message or few-shot example turns.  Ask for the final answer in
    <answer>NUMBER</answer> (or \\boxed{NUMBER}) so that extract_answer reads the right number.
    """
    return [
        {
            "role": "system",
            "content": (
                "Solve the word problem with short calculations, then put only the "
                "final number inside <answer>NUMBER</answer>."
            ),
        },
        {
            "role": "user",
            "content": (
                "A kitchen filled 6 trays with 8 rolls each, then sold 19 rolls. "
                "How many rolls were left?"
            ),
        },
        {
            "role": "assistant",
            "content": "6 * 8 = 48 rolls. 48 - 19 = 29 rolls left.\n<answer>29</answer>",
        },
        {
            "role": "user",
            "content": (
                "A shop sold 9 notebooks for 4 dollars each and 3 folders for "
                "2 dollars each. How many dollars did the shop collect?"
            ),
        },
        {
            "role": "assistant",
            "content": "9 * 4 = 36 and 3 * 2 = 6. 36 + 6 = 42 dollars.\n<answer>42</answer>",
        },
        {"role": "user", "content": question},
    ]
