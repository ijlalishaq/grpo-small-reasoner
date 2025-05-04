# grpo-small-reasoner

A small learning project where I teach myself how GRPO (the reinforcement learning method behind DeepSeek-R1) works by using it on a tiny model, on my laptop.

The model is Qwen2.5-0.5B-Instruct. The task is short generated arithmetic word problems ("Sara has 12 apples and buys 3 packs of 4..."). The only reward is a Python function that checks whether the final number is right. It runs without a GPU, a cloud account or any paid API, and one run takes about an hour on a normal 4-core laptop CPU.

A 0.5B model trained for 100 steps won't be good at math, and I wasn't aiming for that. I wanted to see each part of RL fine-tuning up close: sampling a group of answers, scoring them, turning the scores into advantages and watching the model change.

New to GRPO? [CONCEPTS.md](CONCEPTS.md) is my notes on every idea used here, in the order the code uses them.

## How it works

```
word problem ──► model samples G=4 answers ──► rewards.py scores each (correct? well-formatted?)
                                                       │
         LoRA update ◄── advantage = reward − group mean (÷ std)
```

There's no reward model and no value network. The answer checker is plain Python with unit tests.

## What you need

| | |
|---|---|
| Hardware | Any laptop with 4+ CPU cores and 16 GB of RAM (training peaks around 6 GB) |
| Disk | About 3 GB for the CPU version of PyTorch and the 0.5B model |
| Time | Smoke test: about a minute and a half. Training run: about an hour. Eval: 8 to 40 minutes (sampling 4 answers per question is the slow part; `--k 1` skips it) |
| Accounts | None. The model downloads from Hugging Face without logging in, and the dataset is generated locally |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch, much smaller
pip install -r requirements.txt
pytest                 # fast unit tests, a couple of seconds
pytest -m slow         # end-to-end train → merge → eval on a tiny random model (about a minute)
```

## Running it

```bash
# 1. Baseline first, so there's something to compare against
python eval.py --model Qwen/Qwen2.5-0.5B-Instruct --split val --tag base

# 2. Smoke test: 3 steps, just to see that nothing crashes
python train_grpo.py configs/smoke.yaml

# 3. The real run (about an hour), then merge the LoRA adapter and evaluate
python train_grpo.py configs/main.yaml --seed 0
python merge_lora.py outputs/main-seed0 outputs/main-seed0-merged
python eval.py --model outputs/main-seed0-merged --split val --tag grpo-seed0

# 4. Look at the reward curve and at what the model actually wrote
python show_curve.py outputs/main-seed0
less results/samples/grpo-seed0_toy_val.jsonl
```

Want it even faster? Add `--limit 20 --k 1` to the eval commands while you're poking around.

## Layout

| File | What it does |
|---|---|
| `toy_math.py` | Generates the word problems (1 to 3 steps, small numbers) with fixed seeds, so train/val/test are the same everywhere |
| `rewards.py` | Answer extraction, number normalization, correctness (1.0) and format (0.2) rewards. Shared by training and eval |
| `grpo_math.py` | Group advantages, zero-variance fraction, pass@k and majority vote, written out so you can read and test them |
| `data.py` | The prompt template and dataset loading (toy by default, GSM8K as a harder option) |
| `train_grpo.py` | GRPO + LoRA training via TRL, driven by a YAML config |
| `sft.py` | Optional supervised warm start on 200 worked examples, to teach the answer format first |
| `merge_lora.py` | Folds a LoRA adapter into a standalone model |
| `eval.py` | pass@1, maj@k, pass@k, format rate and mean length; appends to `results/results.jsonl` |
| `show_curve.py` | Prints a training curve in the terminal from the saved logs |
| `configs/` | `main.yaml` (the 100-step CPU run), `smoke.yaml` (3 steps), `ablations/` (one-change experiments) |
| `tests/` | Reward/parser, GRPO math and dataset tests, plus a CPU end-to-end pipeline test |

## What to watch in the training logs

| Log key | Healthy | If not |
|---|---|---|
| `reward` | Creeps up over the run | Read the logged completions. Is the parser rejecting good answers? |
| `reward_std` | Clearly above 0 | It's the spread of rewards within a group. Near 0 means groups are all-right or all-wrong, so there's no signal. Raise the temperature or G, or make the problems easier |
| `rewards/correctness_reward`, `rewards/format_reward` | Both rise | If only the format reward moves, the model is learning tags, not math |
| `completion_length` | Stable or slowly rising | Sitting at `max_completion_length` means answers get cut off |
| `kl` | Small and smooth (when β > 0) | Spikes mean the policy is drifting fast. Lower the learning rate |

## What I found

One run of `configs/main.yaml` (seed 0), limited to 4 CPU threads (`OMP_NUM_THREADS=4`). Training took 56 minutes and peaked at about 6 GB of RAM. The evals took 8 to 40 minutes each; sampling 4 answers per val question is the slow part.

| Model | val pass@1 | val maj@4 | val pass@4 | test pass@1 | format rate (test) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct, no training | 54% | 64% | 69% | 50.5% | 30.5% |
| After 100 GRPO steps | 91% | 94% | 99% | 88% | 100% |

The reward curve, from `python show_curve.py outputs/main-seed0`:

```
reward (mean of every 10 steps)
steps    1-10      0.653 ######################
steps   11-20      0.668 ######################
steps   21-30      0.825 ############################
steps   31-40      0.875 #############################
steps   41-50      1.188 ########################################
steps   51-60      1.050 ###################################
steps   61-70      1.133 ######################################
steps   71-80      1.123 ######################################
steps   81-90      1.125 ######################################
steps   91-100     1.135 ######################################
```

And by problem type on the test set (the generator cycles 1-, 2- and 3-step problems):

| Problem type | No training | After GRPO |
|---|---|---|
| 1 step | 99% | 100% |
| 2 steps | 48% | 100% |
| 3 steps | 5% | 64% |

What I take from this:

Part of the jump is format. 78 of the base model's 200 test answers have no `<answer>` tag, and most of those still state a result, usually ending with a sentence like "So, Mina has **19** cards left". If you just take the last number in the text, the base model scores 66% on test, not 50.5%, so about 15 of the 37 points are the model learning the output format. The format reward hit its 0.2 maximum after about 20 steps.

The rest of the gain is mostly two-step problems, which went from 48% to 100%. One-step problems were already solved.

Three-step problems are where it still fails: all 24 wrong test answers are three-step problems. 14 of them squeeze the whole calculation into one line, like `39 + 30 - 20 = 59` (it's 49), while only 5 of the 42 correct three-step answers do that. Writing one calculation per line seems to be what keeps the arithmetic straight.

Learning stalled around step 40. After that, `reward_std` dropped to about 0.03 to 0.1: most groups were all right (the easy problems) or all wrong, so there was little left to learn from. A bigger group, a higher temperature or filtering out the easy problems are what I'd try next.

I expected answers to get longer, since that's the usual story about RL and reasoning. They got shorter: greedy answers on test went from 63 to 47 tokens on average, because the model dropped the prose around its calculations and kept the arithmetic. At this size, at least, there was no "thinking longer".

pass@4 went up too, from 69% to 99%, so on this task GRPO didn't only move probability from pass@4 into pass@1. That probably says more about how easy the toy problems are than about GRPO in general.

The worked example in the prompt matters a lot. Without it, the base model wrote its reasoning in `<思考>` tags (Chinese for "think") in 91 of 100 val answers, never once matched the format, and got 39% right instead of 54%.

All of this is one seed on an easy, made-up task. I wouldn't read anything into the decimals, and none of it says the model got better at real math. Trying the trained model on a slice of GSM8K (`--dataset gsm8k --limit 50`) is the obvious next check.

Raw numbers are in `results/results.jsonl`.

## Experiments to try next

Each file in `configs/ablations/` changes one thing from the main run, so each is another hour on the laptop:

- `group_2.yaml` / `group_8.yaml`: how much does the group size G matter?
- `kl_0.04.yaml`: does a KL penalty slow learning down, or keep the model saner?
- `no_format.yaml`: correctness reward only. Does the model still pick up the tags?
- `from_sft.yaml`: start from a quick SFT warm start (`python sft.py`, then `merge_lora.py outputs/sft outputs/sft-merged`)

## Things I left out on purpose

- Bigger models, GPUs and vLLM. They'd make the numbers better and the project harder to run and read.
- GSM8K as the main task. It's still supported (`dataset: gsm8k` in a config, `--dataset gsm8k` for eval), but a 0.5B model gets so few right that most GRPO groups have nothing to learn from on a CPU budget. CONCEPTS.md, section 4, explains why.
- Experiment trackers. TRL's console logs and `show_curve.py` are enough at this size.

Pinned versions are in `requirements.txt`. TRL's API changes often, so if you upgrade, run `pytest -m slow` first.

## License

MIT
