"""The small amount of math behind GRPO and the eval metrics, written out plainly.

TRL does the advantage computation for you during training. This file exists
so you can read, test and print exactly what it computes, and so eval.py and
the training logs share one definition of pass@k and majority voting.
"""

from __future__ import annotations

from collections import Counter
from math import comb


def group_advantages(rewards: list[float], group_size: int, scale: bool = True, eps: float = 1e-4) -> list[float]:
    """GRPO's advantage: how much better each answer did than its siblings.

    `rewards` holds group_size consecutive answers per prompt. Each answer's
    advantage is (reward - group mean), optionally divided by the group std.
    No value network is needed: the group mean *is* the baseline.

    If every answer in a group got the same reward, all advantages are 0 and
    that prompt teaches the model nothing (see zero_variance_fraction).
    """
    if len(rewards) % group_size:
        raise ValueError("len(rewards) must be a multiple of group_size")
    out = []
    for i in range(0, len(rewards), group_size):
        group = rewards[i : i + group_size]
        mean = sum(group) / group_size
        # Population std, matching torch.std(unbiased=False) up to the eps.
        std = (sum((r - mean) ** 2 for r in group) / group_size) ** 0.5
        out.extend((r - mean) / (std + eps) if scale else r - mean for r in group)
    return out


def zero_variance_fraction(rewards: list[float], group_size: int) -> float:
    """Share of prompts where all answers scored the same (no learning signal)."""
    groups = [rewards[i : i + group_size] for i in range(0, len(rewards), group_size)]
    return sum(len(set(g)) == 1 for g in groups) / len(groups)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021).

    Probability that at least one of k samples drawn without replacement from
    the n is correct. With n == k this is just "any of them correct".
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def majority_vote(answers: list) -> object | None:
    """Most common non-None answer; ties go to the answer seen first."""
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return None
    top = max(counts.values())
    return next(a for a in answers if a is not None and counts[a] == top)
