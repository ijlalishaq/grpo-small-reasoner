"""Evaluate any model (base, SFT, GRPO, merged) with the same prompt and parser.

    python eval.py --model Qwen/Qwen2.5-0.5B-Instruct --split val --tag base
    python eval.py --model outputs/main-seed0-merged --split test --tag grpo-seed0

On a laptop CPU, the 100-question val split takes roughly 10-20 minutes with
the default --k 4. Use --k 1 --limit 20 while you are just poking around.

Metrics:
  pass@1   greedy decoding, the headline number
  maj@k    majority vote over k sampled answers
  pass@k   any of the k sampled answers correct
  format   share of greedy outputs with valid <think>/<answer> tags
  length   mean greedy completion length in tokens

One summary line is appended to results/results.jsonl per run, and every
generation is saved under results/samples/ so you can read what the model did.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

from data import load_split
from grpo_math import majority_vote, pass_at_k
from rewards import has_valid_format, parse_gold, predicted_number

RESULTS = Path(__file__).parent / "results"


@lru_cache(maxsize=1)
def load_model(model_name):
    """Cached so the greedy and the sampled pass share one copy of the weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    return model, tok, device


def generate_hf(model_name, prompts, n, temperature, max_new_tokens, batch_size):
    """Return (texts, token_counts), each a list of n-item lists, one per prompt."""
    import torch

    model, tok, device = load_model(model_name)
    texts, lengths = [], []
    for i in range(0, len(prompts), batch_size):
        chats = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts[i : i + batch_size]]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to(device)
        sampling = dict(do_sample=True, temperature=temperature, top_p=1.0) if temperature > 0 else dict(do_sample=False)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, num_return_sequences=n, pad_token_id=tok.pad_token_id, **sampling)
        new = out[:, enc["input_ids"].shape[1] :]
        for j in range(0, len(new), n):
            group = new[j : j + n]
            texts.append(tok.batch_decode(group, skip_special_tokens=True))
            lengths.append([int((seq != tok.pad_token_id).sum()) for seq in group])
    return texts, lengths


def _fmt(x) -> str:
    # Decimal("20").normalize() prints as "2E+1"; "f" formatting keeps it readable.
    return "None" if x is None else format(x, "f")


def evaluate(args) -> dict:
    ds = load_split(args.split, name=args.dataset, local_dir=args.local_data)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    prompts, golds = ds["prompt"], [parse_gold(a) for a in ds["answer"]]

    greedy, greedy_len = generate_hf(args.model, prompts, 1, 0.0, args.max_new_tokens, args.batch_size)
    sampled = [[] for _ in prompts]
    if args.k > 1:
        sampled, _ = generate_hf(args.model, prompts, args.k, args.temperature, args.max_new_tokens, args.batch_size)

    rows, totals = [], dict(pass1=0.0, majk=0.0, passk=0.0, fmt=0.0, length=0.0)
    for prompt, gold, g, glen, samples in zip(prompts, golds, greedy, greedy_len, sampled):
        g_pred = predicted_number(g[0])
        s_preds = [predicted_number(s) for s in samples]
        n_correct = sum(p is not None and p == gold for p in s_preds)
        totals["pass1"] += g_pred is not None and g_pred == gold
        totals["fmt"] += has_valid_format(g[0])
        totals["length"] += glen[0]
        if samples:
            totals["majk"] += majority_vote(s_preds) == gold
            totals["passk"] += pass_at_k(len(samples), n_correct, args.k)
        rows.append(dict(question=prompt[-1]["content"], gold=_fmt(gold), greedy=g[0], greedy_pred=_fmt(g_pred), samples=samples, n_correct=n_correct))

    n = len(prompts)
    summary = {
        "tag": args.tag or Path(args.model).name,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "n": n,
        "pass@1": totals["pass1"] / n,
        "format_rate": totals["fmt"] / n,
        "mean_length": totals["length"] / n,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_new_tokens": args.max_new_tokens,
    }
    if args.k > 1:
        summary |= {f"maj@{args.k}": totals["majk"] / n, f"pass@{args.k}": totals["passk"] / n, "temperature": args.temperature}

    (RESULTS / "samples").mkdir(parents=True, exist_ok=True)
    stem = f"{summary['tag']}_{args.dataset}_{args.split}"
    with open(RESULTS / "samples" / f"{stem}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(RESULTS / "results.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hub id or local path (merge LoRA adapters first)")
    p.add_argument("--dataset", default="toy", choices=["toy", "gsm8k", "svamp"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--tag", help="name for this row in the results table")
    p.add_argument("--k", type=int, default=4, help="samples per question for maj@k / pass@k (1 = greedy only, fastest)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, help="evaluate only the first N questions")
    p.add_argument("--local-data", help="directory of <dataset>_<split>.jsonl files for offline runs")
    return p.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
