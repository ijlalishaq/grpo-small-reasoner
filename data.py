"""Datasets and the one prompt template every model sees.

The default dataset is "toy": small generated word problems (see toy_math.py).
They need no download and are easy enough for a 0.5B model to learn from on a CPU.

GSM8K is still here as a stretch goal, if you want to see how much harder real
grade-school problems are. Its splits:
  train - GSM8K train minus 200 held-out questions (used for GRPO)
  val   - those 200 questions (used to compare checkpoints and configs)
  test  - GSM8K test (used only for the final results table)

The validation indices live in data/splits.json, which is committed, so the
split never changes between runs or machines.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset

from toy_math import make_problems

# The worked example matters a lot for a 0.5B model. Without it, the model wrote its reasoning
# in <思考> tags (Chinese for "think") 91 times out of 100, never matched the format, and got 39%
# of val right; with it, 54% is right. That leaves room for RL without starting at zero.
SYSTEM_PROMPT = (
    "Solve the math problem. Think step by step inside <think> </think> tags, writing out each calculation. "
    "Then give only the final number inside <answer> </answer> tags.\n\n"
    "Example:\n<think>\n3 * 4 = 12\n12 + 5 = 17\n</think>\n<answer>17</answer>"
)

VAL_SIZE = 200
SPLIT_SEED = 0
SPLITS_FILE = Path(__file__).parent / "data" / "splits.json"


def make_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def _val_indices(n_train: int, splits_file: Path) -> list[int]:
    if splits_file.exists():
        saved = json.loads(splits_file.read_text())
        if saved["n_train"] != n_train:
            raise ValueError(f"{splits_file} was made for {saved['n_train']} rows, got {n_train}")
        return saved["val_indices"]
    idx = sorted(random.Random(SPLIT_SEED).sample(range(n_train), min(VAL_SIZE, n_train // 5)))
    splits_file.parent.mkdir(exist_ok=True)
    splits_file.write_text(json.dumps({"n_train": n_train, "seed": SPLIT_SEED, "val_indices": idx}))
    return idx


def _load_raw(name: str, split: str, local_dir: str | None) -> Dataset:
    """Load from the Hub, or from <local_dir>/<name>_<split>.jsonl for offline runs."""
    if local_dir:
        return load_dataset("json", data_files=str(Path(local_dir) / f"{name}_{split}.jsonl"), split="train")
    if name == "gsm8k":
        return load_dataset("openai/gsm8k", "main", split=split)
    if name == "svamp":
        return load_dataset("ChilleD/SVAMP", split=split)
    raise ValueError(f"unknown dataset {name}")


def _to_rows(raw: Dataset) -> Dataset:
    """GSM8K-style rows -> prompt / gold answer / worked solution."""
    return raw.map(
        lambda ex: {
            "prompt": make_prompt(ex["question"]),
            "answer": ex["answer"].split("####")[-1].strip(),
            "solution": ex["answer"],  # full worked solution, used by sft.py
        },
        remove_columns=raw.column_names,
    )


def load_split(split: str, name: str = "toy", local_dir: str | None = None) -> Dataset:
    """Return a dataset with columns `prompt` (chat messages) and `answer` (gold number as text).

    toy and GSM8K rows also carry `solution`, the reference worked solution.
    """
    if name == "toy":
        return _to_rows(Dataset.from_list(make_problems(split)))
    if name == "gsm8k":
        if split in ("train", "val"):
            raw = _load_raw("gsm8k", "train", local_dir)
            # Offline data gets its own split file so it never clobbers the real one.
            splits_file = Path(local_dir) / "splits.json" if local_dir else SPLITS_FILE
            val = set(_val_indices(len(raw), splits_file))
            keep = [i for i in range(len(raw)) if (i in val) == (split == "val")]
            raw = raw.select(keep)
        else:
            raw = _load_raw("gsm8k", "test", local_dir)
        return _to_rows(raw)
    if name == "svamp":
        # Out-of-distribution check only, so it has no train/val handling.
        raw = _load_raw("svamp", "test", local_dir)
        return raw.map(
            lambda ex: {
                "prompt": make_prompt(f"{ex['Body'].strip()} {ex['Question'].strip()}"),
                "answer": str(ex["Answer"]),
            },
            remove_columns=raw.column_names,
        )
    raise ValueError(f"unknown dataset {name}")
