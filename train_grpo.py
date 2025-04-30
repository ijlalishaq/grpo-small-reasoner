"""GRPO training with LoRA, on a laptop CPU.

    python train_grpo.py configs/smoke.yaml            # 3 steps, catches crashes cheaply
    python train_grpo.py configs/main.yaml --seed 1    # the real run (about an hour)

The YAML has a `dataset` key (toy or gsm8k) and four sections:
  model    base model id and dtype
  lora     passed to peft.LoraConfig
  rewards  which reward functions to use, with weights
  grpo     passed straight to trl.GRPOConfig, so any GRPOConfig option works

Every key in `grpo` is a real TRL argument. The ones that matter most:
  num_generations        G, answers sampled per prompt (the "group")
  beta                   KL penalty against the starting model (0 = none)
  max_completion_length  hard cap on answer length in tokens
  temperature            sampling temperature; too low and groups stop varying
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from data import load_split
from rewards import REWARD_FUNCS


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str) -> dict:
    """Load a config; a top-level `extends: other.yaml` inherits from another file."""
    cfg = yaml.safe_load(Path(path).read_text())
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(load_config(str(Path(path).parent / parent)), cfg)
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--seed", type=int, help="overrides grpo.seed and adds -seedN to the output dir")
    p.add_argument("--local-data", help="directory of gsm8k_<split>.jsonl files (only for dataset: gsm8k)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    grpo_args = dict(cfg["grpo"])
    if args.seed is not None:
        grpo_args["seed"] = args.seed
        grpo_args["output_dir"] = f"{grpo_args['output_dir']}-seed{args.seed}"

    # Heavy imports after arg parsing so --help is instant.
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    model_kwargs = {"torch_dtype": cfg["model"].get("dtype", "auto")}

    reward_names = list(cfg["rewards"])
    trainer = GRPOTrainer(
        model=cfg["model"]["name"],
        reward_funcs=[REWARD_FUNCS[name] for name in reward_names],
        args=GRPOConfig(**grpo_args, reward_weights=[cfg["rewards"][n] for n in reward_names], model_init_kwargs=model_kwargs),
        train_dataset=load_split("train", name=cfg.get("dataset", "toy"), local_dir=args.local_data),
        peft_config=LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]),
    )
    trainer.train()
    trainer.save_model(grpo_args["output_dir"])
    print(f"LoRA adapter saved to {grpo_args['output_dir']}. Merge it with merge_lora.py before eval.")


if __name__ == "__main__":
    main()
