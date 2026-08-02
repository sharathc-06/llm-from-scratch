# Stage 1: GRPO from scratch (no trl) — cold-start RL on a base model

Trains a language model with reinforcement learning where the only reward
signal is "did it get the final math answer right" — no human-written
reasoning examples, no learned reward model. This is the actual algorithm
behind DeepSeek-R1's cold-start RL stage, implemented directly in PyTorch.

## Verified so far

All five pieces of the algorithm ran correctly end-to-end in a local smoke
test using a tiny random-init model (this sandbox can't reach the Hugging
Face Hub to download a real checkpoint):

- **Rollout sampling** — for each prompt, sample a *group* of completions
- **Reward scoring** — `reward.py`'s verifiable reward function, sanity-checked
  in isolation first (`python reward.py`)
- **Group-relative advantage** — normalizing rewards within each group
  (GRPO's replacement for PPO's critic network)
- **Token log-prob computation** — for policy, old-policy, and frozen reference
- **Clipped surrogate loss + KL penalty** — the actual GRPO update

With an untrained random model, rewards were correctly all 0 and advantages
correctly came out to 0 (no signal when every completion in a group gets the
same reward) — exactly the expected behavior, not a bug.

## Files

- `config.py` — `GRPOConfig`: group size, KL coefficient, clip epsilon, LR, etc.
- `reward.py` — the verifiable reward function. Run `python reward.py` on its
  own any time you change it — this is the single most important piece to get
  right, since a buggy reward function makes RL fail silently.
- `data.py` — loads GSM8K (falls back to a tiny synthetic set if offline).
- `grpo.py` — the algorithm itself: `sample_group`, `score_group`,
  `normalize_advantages`, `compute_token_logprobs`, `grpo_loss`.
- `train.py` — the training loop, using `transformers`' `AutoModelForCausalLM`
  so it works with any HF checkpoint.

## Running on Colab

```bash
pip install torch transformers datasets

python train.py --model_name HuggingFaceTB/SmolLM2-360M --max_steps 5 \
  --group_size 4 --max_new_tokens 64
```

Start with a handful of steps and a small group size to confirm it runs
against the real model and real GSM8K before committing to a full run —
same principle as every smoke test we've done so far. Watch three numbers
in the log line each step:

- **mean_reward** — should trend up over time, even if slowly and noisily.
  If it's stuck at 0 for a very long time, the model may need an easier
  problem subset to get started (curriculum), or your reward function may
  have a bug.
- **mean_kl** — should stay small and bounded. If it explodes, the policy is
  drifting too far from the reference model, usually a sign to lower the LR
  or raise `kl_coef`.
- **loss** — least informative of the three on its own in RL; trust reward
  and KL more.

Periodically read the actual printed sample completions (`log_samples_every`)
— this is the real signal for whether reasoning is improving, more than any
single number.

## What's not built yet (the rest of the R1 recipe)

This is stage 1 only (cold-start RL on the raw base model). Stages 2-3
(rejection-sample this model's own correct outputs into a fresh SFT dataset,
then run GRPO again on that improved model) come next, once this stage
produces a model that's actually getting some rewards.
