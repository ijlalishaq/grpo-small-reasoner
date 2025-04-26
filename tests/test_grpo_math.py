from decimal import Decimal

import pytest

from grpo_math import group_advantages, majority_vote, pass_at_k, zero_variance_fraction


def test_advantages_center_each_group():
    adv = group_advantages([1, 0, 0, 0, 1, 1, 1, 1], group_size=4, scale=False)
    assert adv == [0.75, -0.25, -0.25, -0.25, 0, 0, 0, 0]


def test_scaled_advantages_have_unit_std():
    adv = group_advantages([1.0, 0.0, 1.0, 0.0], group_size=4)
    assert adv == pytest.approx([1, -1, 1, -1], abs=1e-3)


def test_zero_variance_fraction():
    # Group 1 all wrong, group 2 mixed, group 3 all right.
    rewards = [0, 0, 1, 0, 1, 1]
    assert zero_variance_fraction(rewards, group_size=2) == pytest.approx(2 / 3)


def test_pass_at_k():
    assert pass_at_k(n=8, c=0, k=1) == 0.0
    assert pass_at_k(n=8, c=8, k=1) == 1.0
    assert pass_at_k(n=8, c=2, k=1) == pytest.approx(0.25)
    assert pass_at_k(n=8, c=1, k=8) == 1.0


def test_majority_vote():
    assert majority_vote([Decimal(3), None, Decimal(4), Decimal(3)]) == Decimal(3)
    assert majority_vote([None, None]) is None
    assert majority_vote([Decimal(1), Decimal(2)]) == Decimal(1)
