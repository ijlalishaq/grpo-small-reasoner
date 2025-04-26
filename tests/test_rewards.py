from decimal import Decimal

import pytest

from rewards import (
    correctness_reward,
    extract_answer,
    format_reward,
    normalize_number,
    parse_gold,
)

GOOD = "<think>3 boxes of 6 is 18.</think>\n<answer>18</answer>"


def chat(text):
    return [{"role": "assistant", "content": text}]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("18", Decimal(18)),
        ("$18.00", Decimal(18)),
        ("18.", Decimal(18)),
        ("1,234", Decimal(1234)),
        ("1,234.5", Decimal("1234.5")),
        ("-7", Decimal(-7)),
        ("0.25", Decimal("0.25")),
        (".5", Decimal("0.5")),
        ("42 apples", Decimal(42)),
        ("0", Decimal(0)),
        ("", None),
        ("eighteen", None),
        ("3 or 4", None),
        (None, None),
    ],
)
def test_normalize_number(text, expected):
    assert normalize_number(text) == expected


def test_parse_gold_gsm8k_format():
    assert parse_gold("Natalia sold 48/2 = 24 clips.\n#### 1,072") == Decimal(1072)
    assert parse_gold("72") == Decimal(72)


def test_extract_answer_takes_last_block():
    assert extract_answer("<answer>1</answer> wait <answer>2</answer>") == "2"
    assert extract_answer("no tags here") is None


def test_correct_answer():
    assert correctness_reward([GOOD], answer=["... #### 18"]) == [1.0]


def test_wrong_answer():
    assert correctness_reward(["<think>x</think><answer>17</answer>"], answer=["#### 18"]) == [0.0]


def test_comma_and_decimal_answers_match():
    comps = [
        "<think>x</think><answer>1,000</answer>",
        "<think>x</think><answer>2.50</answer>",
    ]
    assert correctness_reward(comps, answer=["#### 1000", "#### 2.5"]) == [1.0, 1.0]


def test_malformed_gets_no_credit():
    comps = [
        "The answer is 18",  # no tags at all
        "<think>x</think><answer></answer>",  # empty answer
        "<think>x</think><answer>17 or 18</answer>",  # hedging
        "<answer>18",  # unclosed tag
    ]
    assert correctness_reward(comps, answer=["#### 18"] * 4) == [0.0] * 4


def test_chat_style_completions():
    assert correctness_reward([chat(GOOD)], answer=["#### 18"]) == [1.0]
    assert format_reward([chat(GOOD)]) == [0.2]


def test_format_reward():
    comps = [
        GOOD,
        "<answer>18</answer>",  # missing think
        "<think>a</think><think>b</think><answer>18</answer>",  # two think blocks
        "<think>a</think><answer>18</answer> extra text",  # trailing text
        "<think>a</think><answer>1</answer><answer>2</answer>",  # two answers
    ]
    assert format_reward(comps) == [0.2, 0.0, 0.0, 0.0, 0.0]
