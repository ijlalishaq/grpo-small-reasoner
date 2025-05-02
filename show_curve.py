"""Print a training curve in the terminal, no plotting library needed.

    python show_curve.py outputs/main-seed0
    python show_curve.py outputs/main-seed0 --key reward_std --every 5

Reads the log history TRL saves in the newest checkpoint-*/trainer_state.json
and draws one bar per logged step (or a moving average with --every).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_history(run_dir: str) -> list[dict]:
    ckpts = sorted(Path(run_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not ckpts:
        raise SystemExit(f"no checkpoint-* folders in {run_dir} (they're written every save_steps)")
    return json.loads((ckpts[-1] / "trainer_state.json").read_text())["log_history"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--key", default="reward", help="any logged key, e.g. reward, reward_std, completion_length, kl")
    p.add_argument("--every", type=int, default=10, help="average this many steps into one bar")
    args = p.parse_args(argv)

    points = [(h["step"], float(h[args.key])) for h in load_history(args.run_dir) if args.key in h]
    if not points:
        raise SystemExit(f"key {args.key!r} not found in the log")
    rows = []
    for i in range(0, len(points), args.every):
        chunk = points[i : i + args.every]
        rows.append((chunk[0][0], chunk[-1][0], sum(v for _, v in chunk) / len(chunk)))
    top = max(abs(v) for *_, v in rows) or 1.0
    print(f"{args.key} (mean of every {args.every} steps)")
    for first, last, v in rows:
        print(f"steps {first:>4}-{last:<4} {v:8.3f} {'#' * round(40 * abs(v) / top)}")


if __name__ == "__main__":
    main()
