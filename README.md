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

# Stage 2: rejection sampling + SFT

Takes your stage-1 GRPO checkpoint, samples multiple attempts per training
question, keeps only the ones that got the right answer, and uses those to
fine-tune a **fresh copy** of the base model (not the GRPO checkpoint --
starting clean, matching the actual R1 recipe). The model teaches itself:
no human wrote these training examples, they're the model's own successful
reasoning, filtered for correctness.

## Verified locally

Both scripts ran end-to-end with a tiny random-init model (this sandbox has
no Hugging Face Hub access): rejection sampling correctly scores completions
and writes valid JSONL; the SFT dataset correctly masks prompt tokens (only
completion tokens get gradient signal); a real forward/backward/optimizer
step runs cleanly.

## Files

- `rejection_sample.py` — samples N attempts per question from your GRPO
  checkpoint, filters to reward >= 1.0 (genuinely correct, not just
  format-bonus), writes `{question, answer, completion}` JSONL.
- `sft_train.py` — standard SFT: cross-entropy loss on completion tokens
  only (prompt masked with -100), trained from a fresh base model copy.
- `data.py`, `reward.py` — same as stage 1, copied in so this folder is
  self-contained.

## Running on Colab

Step 1 — generate the dataset from your stage-1 checkpoint:

```bash
python rejection_sample.py --ckpt /path/to/your/grpo/checkpoints/final \
    --trained_tokenizer HuggingFaceTB/SmolLM2-360M-Instruct \
    --n_questions 500 --samples_per_question 4 --out sft_data.jsonl
```

Watch the accept rate it prints. Given your stage-1 eval showed ~2-4%
accuracy, expect a low accept rate here too -- that's normal, not a bug.
If you end up with fewer than ~50-100 examples, raise `--n_questions` or
`--samples_per_question` before moving on; too little data makes stage 2
unreliable.

Step 2 — fine-tune a fresh model on it:

```bash
python sft_train.py --data sft_data.jsonl \
    --base_model HuggingFaceTB/SmolLM2-360M-Instruct \
    --epochs 3 --out_dir sft_checkpoint
```

Note the LR here (`2e-5` default) is much higher than GRPO's -- this is
ordinary supervised learning, not RL, so it converges faster and doesn't
need the same caution.

## Evaluating

Reuse `eval_checkpoint.py` from stage 1 -- point `--ckpt` at `sft_checkpoint`
and `--trained_tokenizer` at the same base model name, and compare against
both the raw base model AND your stage-1 GRPO checkpoint. Three-way
comparison is the real test of whether stage 2 helped.

