"""End-to-end plumbing test on CPU with a tiny random model and fake data.

It needs no downloads, so it checks that train -> merge -> eval connect up,
not that anything is learned. Run it with:  pytest -m slow
"""

import json

import pytest

pytest.importorskip("trl")
pytestmark = pytest.mark.slow

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def make_tiny_model(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    specials = ["<pad>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]
    vocab = {t: i for i, t in enumerate(specials + [chr(c) for c in range(32, 127)] + ["\n"])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="?"))
    tk.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<|end|>",
                                  additional_special_tokens=specials[2:])
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(path)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=1024,
                      pad_token_id=0, eos_token_id=1, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(path)


def make_fake_gsm8k(path):
    rows = [{"question": f"What is {i} + {i}?", "answer": f"{i} + {i} = {2 * i}\n#### {2 * i}"} for i in range(40)]
    for split in ("train", "test"):
        (path / f"gsm8k_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_train_merge_eval(tmp_path, monkeypatch):
    import eval as eval_mod
    import merge_lora
    import train_grpo

    model_dir, data_dir, out = tmp_path / "tiny", tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    make_tiny_model(model_dir)
    make_fake_gsm8k(data_dir)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
dataset: gsm8k
model: {{name: {model_dir}, dtype: float32}}
lora: {{r: 4, lora_alpha: 8, target_modules: [q_proj, v_proj]}}
rewards: {{correctness: 1.0, format: 1.0}}
grpo:
  output_dir: {out}
  max_steps: 2
  per_device_train_batch_size: 4
  num_generations: 4
  max_completion_length: 16
  learning_rate: 1.0e-4
  logging_steps: 1
  save_steps: 1
  report_to: none
  use_cpu: true
""")
    train_grpo.main([str(cfg), "--local-data", str(data_dir)])
    assert (out / "adapter_config.json").exists()
    assert (out / "checkpoint-2").exists()

    merge_lora.main([str(out), str(tmp_path / "merged")])

    monkeypatch.setattr(eval_mod, "RESULTS", tmp_path / "results")
    summary = eval_mod.evaluate(eval_mod.parse_args([
        "--model", str(tmp_path / "merged"), "--split", "val", "--k", "2", "--limit", "4",
        "--max-new-tokens", "8", "--dataset", "gsm8k", "--local-data", str(data_dir), "--tag", "tiny",
    ]))
    assert summary["n"] == 4
    assert 0.0 <= summary["pass@1"] <= 1.0 and "maj@2" in summary
    assert (tmp_path / "results" / "samples" / "tiny_gsm8k_val.jsonl").exists()
