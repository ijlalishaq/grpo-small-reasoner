"""Fast tests for data splits, SFT targets and eval metric bookkeeping (no model needed)."""

import json

import pytest

pytest.importorskip("datasets")

import eval as eval_mod
from data import load_split
from rewards import has_valid_format, is_correct, parse_gold
from sft import to_target
from toy_math import SPLIT_SIZES, make_problems


@pytest.fixture
def fake_data(tmp_path):
    rows = [{"question": f"What is {i} + {i}?", "answer": f"{i} + {i} = <<{i}+{i}={2 * i}>>{2 * i}\n#### {2 * i}"} for i in range(50)]
    for split in ("train", "test"):
        (tmp_path / f"gsm8k_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def test_train_and_val_are_disjoint_and_stable(fake_data):
    train, val = load_split("train", "gsm8k", fake_data), load_split("val", "gsm8k", fake_data)
    assert len(train) + len(val) == 50 and len(val) == 10
    train_q = {p[-1]["content"] for p in train["prompt"]}
    assert not train_q & {p[-1]["content"] for p in val["prompt"]}
    # Second load reads the saved split file and gives the same questions.
    assert load_split("val", "gsm8k", fake_data)["prompt"] == val["prompt"]
    assert val[0]["prompt"][0]["role"] == "system"


def test_sft_target_is_valid_and_correct(fake_data):
    ex = load_split("train", "gsm8k", fake_data)[0]
    target = to_target(ex["solution"])
    assert "<<" not in target
    assert has_valid_format(target) and is_correct(target, ex["answer"])


def test_eval_metrics(fake_data, tmp_path, monkeypatch):
    def fake_generate(model, prompts, n, temperature, max_new_tokens, batch_size):
        texts = []
        for p in prompts:
            a, b = (int(x.strip("?")) for x in p[-1]["content"].split()[2::2])
            right, wrong = f"<think>add</think><answer>{a + b}</answer>", "<answer>-1</answer>"
            # Greedy is right only for even a; samples are 2 right, 1 wrong.
            texts.append([right if a % 2 == 0 else wrong] if n == 1 else [right, right, wrong][:n])
        return texts, [[10] * len(t) for t in texts]

    monkeypatch.setattr(eval_mod, "generate_hf", fake_generate)
    monkeypatch.setattr(eval_mod, "RESULTS", tmp_path / "results")
    args = eval_mod.parse_args(["--model", "fake", "--split", "test", "--k", "3", "--limit", "10", "--dataset", "gsm8k", "--local-data", str(fake_data)])
    s = eval_mod.evaluate(args)
    assert s["pass@1"] == 0.5 and s["format_rate"] == 0.5
    assert s["maj@3"] == 1.0 and s["pass@3"] == 1.0 and s["mean_length"] == 10
    assert len((tmp_path / "results" / "results.jsonl").read_text().splitlines()) == 1


def test_toy_problems_are_stable_disjoint_and_correct():
    train, val, test = (make_problems(s) for s in ("train", "val", "test"))
    assert [len(train), len(val), len(test)] == [SPLIT_SIZES[s] for s in ("train", "val", "test")]
    assert make_problems("val") == val  # same seed, same problems, on every machine
    train_q = {r["question"] for r in train}
    assert not train_q & {r["question"] for r in val} and not train_q & {r["question"] for r in test}
    for row in train + val + test:
        # Every worked line "a op b = c" is true, and the last result is the gold answer.
        lines = row["answer"].split("####")[0].strip().splitlines()
        for line in lines:
            lhs, rhs = line.split("=")
            assert eval(lhs.replace("/", "//")) == int(rhs) and eval(lhs) == int(rhs)
        gold = parse_gold(row["answer"])
        assert gold is not None and gold >= 0 and gold == int(lines[-1].split("=")[1])


def test_toy_split_loads_with_sft_targets():
    ex = load_split("val", "toy")[0]
    assert ex["prompt"][-1]["content"] and ex["answer"].isdigit()
    target = to_target(ex["solution"])
    assert has_valid_format(target) and is_correct(target, ex["answer"])
