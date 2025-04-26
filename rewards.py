"""Answer parsing and reward functions.

This is the only place that decides whether a model output is "correct".
Training rewards and evaluation both import from here, so a parsing bug
shows up the same way everywhere instead of silently inflating one number.

Reward functions follow the TRL GRPOTrainer signature:
    fn(completions, **dataset_columns) -> list[float]
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

CORRECT_REWARD = 1.0
FORMAT_REWARD = 0.2

# A number with optional sign, optional thousands separators, optional decimals.
# "1,000" and "1000" both match; "1,00" does not get read as a thousands group.
_NUMBER_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-?\.\d+")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_FORMAT_RE = re.compile(
    r"^\s*<think>(?:(?!</?think>).)*</think>\s*<answer>(?:(?!</?answer>).)*</answer>\s*$",
    re.DOTALL,
)


def completion_text(completion) -> str:
    """TRL passes plain strings, or a list of chat messages for chat prompts."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "")
    return ""


def extract_answer(text: str) -> str | None:
    """Return the contents of the last <answer>...</answer> block, if any."""
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


def normalize_number(text: str | None) -> Decimal | None:
    """Turn an answer string into a number, or None if it is not exactly one number.

    Accepts "$1,234.50", "18.", "-3", "42 apples". Rejects "3 or 4" and empty
    strings: an answer that lists several numbers should not get credit for
    containing the right one (that would be an easy reward hack).
    """
    if text is None:
        return None
    numbers = _NUMBER_RE.findall(text)
    if len(numbers) != 1:
        return None
    try:
        value = Decimal(numbers[0].replace(",", ""))
    except InvalidOperation:
        return None
    # Decimal("18.00") and Decimal("18") already compare equal; normalize() also
    # makes them hash the same way, which majority voting relies on.
    return value.normalize()


def parse_gold(answer_field: str) -> Decimal | None:
    """GSM8K gold answers end with '#### <number>'. Plain numbers are accepted too."""
    tail = answer_field.split("####")[-1]
    return normalize_number(tail)


def predicted_number(completion) -> Decimal | None:
    return normalize_number(extract_answer(completion_text(completion)))


def is_correct(completion, gold) -> bool:
    pred = predicted_number(completion)
    gold_value = gold if isinstance(gold, Decimal) else parse_gold(str(gold))
    return pred is not None and gold_value is not None and pred == gold_value


def has_valid_format(completion) -> bool:
    return bool(_FORMAT_RE.match(completion_text(completion)))


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    """1.0 if the number in <answer> matches the gold answer, else 0.0."""
    return [CORRECT_REWARD if is_correct(c, a) else 0.0 for c, a in zip(completions, answer)]


def format_reward(completions, **kwargs) -> list[float]:
    """Small bonus for exactly one <think> block followed by one <answer> block.

    Kept small (0.2 vs 1.0) so the model can't win by formatting nicely and
    answering wrong.
    """
    return [FORMAT_REWARD if has_valid_format(c) else 0.0 for c in completions]


REWARD_FUNCS = {"correctness": correctness_reward, "format": format_reward}
