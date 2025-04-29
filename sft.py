"""Optional SFT warm start: teach the <think>/<answer> format on a couple of hundred examples.

    python sft.py --out outputs/sft --n 200        # ~10-15 minutes on a laptop CPU
    python merge_lora.py outputs/sft outputs/sft-merged
    python train_grpo.py configs/ablations/from_sft.yaml

The reference worked solutions become the <think> part and the final number the
<answer> part. The point is not to teach math (200 examples barely move
accuracy) but to make the format reliable before RL, so GRPO's early steps are
spent on correctness instead of on discovering the tags.
"""

import argparse
import re

from data import load_split


def to_target(solution: str) -> str:
    reasoning, answer = solution.split("####")
    reasoning = re.sub(r"<<.*?>>", "", reasoning).strip()  # drop GSM8K calculator annotations
    return f"<think>\n{reasoning}\n</think>\n<answer>{answer.strip()}</answer>"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="toy", choices=["toy", "gsm8k"])
    p.add_argument("--out", default="outputs/sft")
    p.add_argument("--n", type=int, default=200, help="number of training examples")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local-data", help="directory of gsm8k_<split>.jsonl files (only for --dataset gsm8k)")
    args = p.parse_args(argv)

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    ds = load_split("train", name=args.dataset, local_dir=args.local_data).shuffle(seed=args.seed)
    ds = ds.select(range(min(args.n, len(ds))))
    ds = ds.map(lambda ex: {"prompt": ex["prompt"], "completion": [{"role": "assistant", "content": to_target(ex["solution"])}]},
                remove_columns=ds.column_names)

    trainer = SFTTrainer(
        model=args.model,
        train_dataset=ds,
        args=SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=1e-4, per_device_train_batch_size=4,
                       gradient_accumulation_steps=2, logging_steps=5, save_strategy="no", seed=args.seed, report_to="none",
                       use_cpu=True, bf16=False),
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    trainer.train()
    trainer.save_model(args.out)


if __name__ == "__main__":
    main()
