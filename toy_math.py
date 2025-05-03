"""A tiny generated dataset of word problems, so the whole project runs offline on a laptop.

GSM8K is great but it's too hard for the models that train in reasonable time on a
CPU: a 0.5B model gets most of it wrong, so most GRPO groups are all-wrong and there's
nothing to learn from (see CONCEPTS.md, section 4). These problems are the same *kind*
of thing (short story, a few arithmetic steps, one number at the end) but with small
numbers and 1-3 steps, so a small model gets some right and some wrong. That mix is
exactly what GRPO needs.

Each row looks like a GSM8K row: a `question`, and an `answer` with a worked solution
followed by "#### <number>". That way the rest of the code can't tell the difference.

Everything is generated from a seed, so train/val/test are the same on every machine
and never overlap.
"""

from __future__ import annotations

import random

NAMES = ["Ali", "Sara", "Omar", "Mina", "Leo", "Ava", "Zain", "Noor", "Sam", "Hina"]
ITEMS = ["apples", "pencils", "stickers", "marbles", "books", "cookies", "coins", "cards"]

SPLIT_SIZES = {"train": 2000, "val": 100, "test": 200}
SPLIT_SEEDS = {"train": 1, "val": 2, "test": 3}


def _one_step(r: random.Random) -> tuple[str, str, int]:
    name, item = r.choice(NAMES), r.choice(ITEMS)
    kind = r.choice(["add", "sub", "mul"])
    if kind == "add":
        a, b = r.randint(5, 60), r.randint(5, 60)
        return f"{name} has {a} {item}. A friend gives {name} {b} more. How many {item} does {name} have now?", f"{a} + {b} = {a + b}", a + b
    if kind == "sub":
        a = r.randint(20, 90)
        b = r.randint(2, a - 1)
        return f"{name} has {a} {item} and gives away {b}. How many {item} are left?", f"{a} - {b} = {a - b}", a - b
    a, b = r.randint(2, 9), r.randint(3, 12)
    return f"{name} buys {a} bags with {b} {item} in each bag. How many {item} is that in total?", f"{a} * {b} = {a * b}", a * b


def _two_step(r: random.Random) -> tuple[str, str, int]:
    name, item = r.choice(NAMES), r.choice(ITEMS)
    kind = r.choice(["mul_add", "mul_sub", "add_div"])
    if kind == "mul_add":
        start, bags, per = r.randint(3, 30), r.randint(2, 6), r.randint(3, 10)
        total = start + bags * per
        return (f"{name} has {start} {item}. Then {name} buys {bags} packs with {per} {item} in each pack. How many {item} does {name} have now?",
                f"{bags} * {per} = {bags * per}\n{start} + {bags * per} = {total}", total)
    if kind == "mul_sub":
        boxes, per = r.randint(3, 8), r.randint(4, 10)
        have = boxes * per
        used = r.randint(1, have - 1)
        return (f"{name} has {boxes} boxes of {item} with {per} in each box. {name} uses {used} of them. How many {item} are left?",
                f"{boxes} * {per} = {have}\n{have} - {used} = {have - used}", have - used)
    friends, share = r.randint(2, 6), r.randint(2, 9)
    a = r.randint(1, friends * share - 1)
    b = friends * share - a
    return (f"{name} has {a} {item} and finds {b} more. {name} shares them equally among {friends} friends. How many {item} does each friend get?",
            f"{a} + {b} = {a + b}\n{a + b} / {friends} = {share}", share)


def _three_step(r: random.Random) -> tuple[str, str, int]:
    name, item = r.choice(NAMES), r.choice(ITEMS)
    start, bags, per = r.randint(5, 40), r.randint(2, 5), r.randint(3, 8)
    gained = bags * per
    lost = r.randint(1, start + gained - 1)
    friends = r.choice([d for d in range(2, 6) if (start + gained - lost) % d == 0] or [1])
    end = (start + gained - lost) // friends
    if friends == 1:
        q = (f"{name} has {start} {item}, buys {bags} packs of {per}, then loses {lost}. How many {item} does {name} have?")
        steps = f"{bags} * {per} = {gained}\n{start} + {gained} = {start + gained}\n{start + gained} - {lost} = {end}"
    else:
        q = (f"{name} has {start} {item}, buys {bags} packs of {per}, then loses {lost}. "
             f"{name} splits what is left equally into {friends} piles. How many {item} are in each pile?")
        steps = (f"{bags} * {per} = {gained}\n{start} + {gained} = {start + gained}\n"
                 f"{start + gained} - {lost} = {start + gained - lost}\n{start + gained - lost} / {friends} = {end}")
    return q, steps, end


def make_problems(split: str, n: int | None = None) -> list[dict]:
    """Return `n` problems for a split, as {"question", "answer"} dicts in GSM8K format."""
    r = random.Random(SPLIT_SEEDS[split])
    n = SPLIT_SIZES[split] if n is None else n
    # Mix of difficulties: roughly a third each.
    makers = [_one_step, _two_step, _three_step]
    rows, seen = [], set()
    # Never let the same question text land in two splits: val skips train, test skips both.
    earlier = {"train": [], "val": ["train"], "test": ["train", "val"]}[split]
    other = {p["question"] for s in earlier for p in make_problems(s)}
    while len(rows) < n:
        q, steps, ans = makers[len(rows) % 3](r)
        if q in seen or q in other:
            continue
        seen.add(q)
        rows.append({"question": q, "answer": f"{steps}\n#### {ans}"})
    return rows


if __name__ == "__main__":
    for row in make_problems("val", 6):
        print(row["question"], "\n ->", row["answer"].replace("\n", " | "), "\n")
