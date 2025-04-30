"""Fold a LoRA adapter into its base model and save a standalone model.

    python merge_lora.py outputs/main outputs/main-merged
    python merge_lora.py outputs/main/checkpoint-300 outputs/main-300-merged

LoRA trains two small matrices A and B per layer and leaves the base weights
frozen; the effective weight is W + (alpha / r) * B @ A. Merging computes that
sum once, so the result loads and runs like any normal model.
"""

import argparse

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("adapter", help="directory with adapter_config.json")
    p.add_argument("out")
    args = p.parse_args(argv)

    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter, torch_dtype=torch.float32)
    model.merge_and_unload().save_pretrained(args.out)
    # Checkpoint dirs don't always carry the tokenizer, so fall back to the base model's.
    try:
        tok = AutoTokenizer.from_pretrained(args.adapter)
    except OSError:
        tok = AutoTokenizer.from_pretrained(model.peft_config["default"].base_model_name_or_path)
    tok.save_pretrained(args.out)
    print(f"merged model saved to {args.out}")


if __name__ == "__main__":
    main()
