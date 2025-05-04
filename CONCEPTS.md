# Concepts behind this project

A short guide to the ideas this repo uses, in the order you meet them in the code. Each section points to the file where the idea lives.

## 1. RL with verifiable rewards

Normal fine-tuning (SFT) shows the model the right answer and trains it to copy it. Reinforcement learning works the other way round. The model writes its own answers, a reward says how good each one was, and training makes the high-reward answers more likely.

The usual RLHF setup gets the reward from a *learned reward model* trained on human preferences. Math is easier than that: we can just check the answer. That's "RL with verifiable rewards" (RLVR), the recipe behind DeepSeek-R1. The reward here is a few lines of Python. See `rewards.py`.

That makes the parser the most important code in the repo. If it gives credit for a wrong answer, or misses a right one, training optimizes the wrong thing and the eval reports the wrong number. That's why `rewards.py` is shared by training and eval and has its own tests.

## 2. Policy gradient in one paragraph

The model (the *policy*) assigns a probability to every answer it can write. The policy-gradient update is roughly:

```
loss = - advantage * log P(answer | question)
```

Gradient descent on this loss raises the probability of answers with positive advantage and lowers it for negative ones. The *advantage* is "how much better than expected was this answer". Everything below is about how to compute that "expected" part, called the *baseline*, cheaply and with low noise.

## 3. GRPO: the group is the baseline

PPO, the classic algorithm, trains a second network (a *value model* or critic) to predict the expected reward. For an LLM that's a second copy of a large model, which costs memory and is hard to train.

GRPO (Group Relative Policy Optimization, from the DeepSeekMath paper) drops the critic. For each question it samples a *group* of G answers and uses the group's own average as the baseline:

```
advantage_i = (reward_i - mean(group)) / std(group)
```

If 2 of 8 answers are right, the right ones get a large positive advantage and the wrong ones a small negative one. You can run this yourself in `grpo_math.py::group_advantages`, and `tests/test_grpo_math.py` has worked examples.

The group size G is a trade-off. A bigger group gives a better baseline and a better chance that at least one answer differs from the rest, but it also costs proportionally more generation. The `group_2` and `group_8` experiments test that.

How the loss is averaged matters too. The original GRPO averages the loss per answer, then over answers, so each token of a long answer counts for less. That favours long wrong answers, whose penalty gets spread thin. DAPO and Dr. GRPO fix this by averaging over all tokens in the batch, and that's what the TRL version pinned here (0.16) does: it divides the summed per-token loss by the total number of completion tokens.

## 4. Zero-variance groups: when there's nothing to learn

If all G answers are wrong (too hard), or all are right (too easy), every advantage is 0 and that question contributes no gradient at all. TRL logs the average within-group spread as `reward_std`. If it sinks toward 0, training has stalled. You can fix it by filtering the dataset to questions of medium difficulty, raising the temperature, or raising G. `grpo_math.py::zero_variance_fraction` computes the same thing.

This is also why the default dataset here is a generated toy one (`toy_math.py`) and not GSM8K. A 0.5B model gets most GSM8K questions wrong, so on a CPU budget of ~100 steps nearly every group would be all-wrong. The toy problems are pitched so the model gets some right and some wrong, which is where GRPO actually has something to learn from.

## 5. The KL penalty (β)

Optionally, GRPO adds a penalty for drifting away from the starting model:

```
loss = policy_loss + β * KL(policy || reference)
```

This keeps the model from forgetting how to write fluent text while chasing reward. It needs a frozen copy of the starting model in memory (with LoRA, TRL just switches the adapter off to get it, so the cost is compute rather than memory). Recent work (DAPO, Dr. GRPO) found that β = 0 often works fine for verifiable-reward tasks, so the main config uses 0 and the `kl_0.04` experiment tests the difference. The `kl` column in the logs shows the drift either way.

## 6. Reward hacking

The model optimizes whatever the reward actually measures, not what you meant. Some typical hacks, and the defence in this repo for each:

| Hack | Defence |
|---|---|
| Write "17 or 18" to get credit for either | `normalize_number` rejects answers holding more than one number |
| Perfect tags, wrong answer | The format reward is only 0.2, versus 1.0 for correctness |
| Several `<answer>` blocks | The format regex requires exactly one of each tag |

The real defence is reading samples. `log_completions: true` prints some every logging step, and eval writes every output to `results/samples/`.

## 7. LoRA

Full fine-tuning of even 0.5B parameters needs gradients and optimizer state for every weight, which is a lot for a laptop. LoRA freezes the base weights and learns a low-rank update for each weight matrix W:

```
W_effective = W + (alpha / r) * B @ A      # A is r×d, B is d×r, r = 16
```

Only the attention layers get adapters here, so only about 2M of the 494M weights (under 0.5%) are trained. That keeps each step fast enough on a CPU. `merge_lora.py` adds `B @ A` into W once, so the result is a normal model again.

## 8. Measuring improvement without fooling yourself

Three numbers come out of `eval.py`:

- pass@1 is one greedy (deterministic) answer per question. It's the headline number.
- maj@k samples k answers and takes the majority vote (also called "self-consistency"). It's usually above pass@1.
- pass@k counts a question as solved if any of the k samples is right, so it measures what the model *can* do. One open question in RL research is whether GRPO teaches new skills or just moves pass@k probability into pass@1. Comparing both before and after training lets you check that on your own model.

All tuning happens on the 100-question validation split. The test set (200 questions) is only touched for the final numbers, or it stops being an unbiased estimate. The toy generator uses a different seed per split and drops any repeated question, so the splits never overlap.

RL is noisy, so a +3 point gain from a single run can be pure luck; on a 100-question val set, one question is a whole point. If you have the patience, run a couple of seeds before believing a small difference.

Finally, evaluating the toy-trained model on a slice of GSM8K (`--dataset gsm8k --limit 50`) shows whether anything carries over to harder, real problems. Don't expect much at this scale.

## 9. SFT warm start

A 0.5B instruct model often ignores the tag format at first. Only about 30% of its answers on the toy test set had valid tags; the rest usually ended with a plain sentence like "So, Mina has **19** cards left". So early GRPO steps are spent learning tags rather than math (in the main run the format reward maxed out after about 20 steps). `sft.py` trains on a couple of hundred formatted worked solutions first. The `from_sft` experiment checks whether that is worth the extra step.

## Further reading

- DeepSeekMath (introduces GRPO): https://arxiv.org/abs/2402.03300
- DeepSeek-R1 (RLVR at scale): https://arxiv.org/abs/2501.12948
- Dr. GRPO, "Understanding R1-Zero-Like Training": https://arxiv.org/abs/2503.20783
- Dang and Ngo, RL for reasoning in small LLMs: https://arxiv.org/abs/2503.16219
- TRL GRPOTrainer docs: https://huggingface.co/docs/trl/main/en/grpo_trainer
