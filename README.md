# llm-from-scratch: building a mini reasoning model (mini-R1)

A small language model trained end-to-end by hand: pretraining, then
reinforcement learning on verifiable rewards (GRPO, the algorithm behind
DeepSeek-R1), then rejection-sampled SFT, then another round of RL. No step
outsourced to a library that does the algorithm for you (aside from
`transformers` for the model itself and standard `torch` primitives).

## Results so far

Evaluated on 200 held-out GSM8K test problems (never seen during training),
using `eval_checkpoint.py`, base `SmolLM2-360M` as the reference point:

| Checkpoint | Accuracy | Avg reward |
|---|---|---|
| Base model (no training) | 0.5–2.0%* | 0.006–0.024* |
| Stage 1: GRPO cold-start RL | 9.5% | 0.125 |
| Stage 2: rejection-sampled SFT | 8.5% | 0.142 |

*Base model accuracy varies run to run due to residual sampling
non-determinism in `generate()` even with a fixed seed on GPU -- both runs
are shown to make that visible rather than hide it.

**Honest read of these numbers:** both trained checkpoints are a genuine,
large improvement over the untouched base model (roughly 5-15x on
accuracy). Stage 1 vs stage 2 accuracy (9.5% vs 8.5%) is within the ~2-point
margin of error at n=200 -- not a confident win either way. Avg reward,
which is continuous rather than binary and so less noisy at this sample
size, favors stage 2 (0.142 vs 0.125), and stage 2's completions were more
consistently *some* reward (rarely a flat 0.00) rather than stage 1's mix of
full misses and full hits. Read as: stage 2 produced more reliably
on-track, correctly-formatted reasoning, even where it didn't land the
final number -- a believable outcome for rejection-sampling+SFT, whose whole
mechanism is "imitate your own successful outputs more consistently,"
rather than a guaranteed accuracy jump.

## Stage 1: GRPO from scratch (no trl) — cold-start RL on a base model

Trains a language model with reinforcement learning where the only reward
signal is "did it get the final math answer right" — no human-written
reasoning examples, no learned reward model. This is the actual algorithm
behind DeepSeek-R1's cold-start RL stage, implemented directly in PyTorch.

**Files** (`grpo/`):
- `config.py` — `GRPOConfig`: group size, KL coefficient, clip epsilon, LR, etc.
- `reward.py` — the verifiable reward function, plus `has_explicit_marker`
  for stricter SFT-data filtering (see stage 2). Run `python reward.py` on
  its own any time you change it — a buggy reward function makes RL fail
  silently.
- `data.py` — loads GSM8K (falls back to a tiny synthetic set if offline).
- `grpo.py` — the algorithm: `sample_group`, `score_group`,
  `normalize_advantages`, `compute_token_logprobs`, `grpo_loss`.
- `train.py` — the training loop, using `transformers`' `AutoModelForCausalLM`
  so it works with any HF checkpoint *or local checkpoint directory* (see
  stage 3, which reuses this unchanged).
- `eval_checkpoint.py` — scores a checkpoint against held-out GSM8K test
  problems, side by side with the untouched base model.

**Running:**
```bash
pip install torch transformers datasets

python train.py --model_name HuggingFaceTB/SmolLM2-360M-Instruct \
  --max_steps 1000 --group_size 8 --max_new_tokens 128
```

Use an **instruct** model, not a plain base model — instruct models are
trained to recognize a chat template and actually attempt an answer instead
of rambling into a hallucinated follow-up question. `format_prompt()` in
`data.py` applies the tokenizer's chat template automatically when present.

Watch three numbers per step:
- **mean_reward** — should trend up over time, even if slowly and noisily.
- **mean_kl** — should move off 0.0000 as training progresses (proof the
  policy is actually diverging from the frozen reference), but stay bounded,
  not explode.
- **loss** — least informative on its own; trust reward and KL more.

## Stage 2: rejection sampling + SFT

Takes the stage-1 checkpoint, samples multiple attempts per training
question, keeps only the ones that got the right answer **and** used the
explicit `#### <answer>` format (stricter than the RL reward function, since
a "correct" answer via muddled/contradictory reasoning is fine to dilute
across a GRPO batch but bad to directly teach a model to imitate via SFT),
and fine-tunes a **fresh copy** of the base model on those — not the GRPO
checkpoint, starting clean, matching the actual R1 recipe.

**Files** (`sft/`):
- `rejection_sample.py` — samples N attempts per question, filters to
  reward ≥ 1.0 AND an explicit `####` marker, truncates completions right
  after the answer, writes `{question, answer, completion}` JSONL.
- `sft_train.py` — standard SFT: cross-entropy loss on completion tokens
  only (prompt masked with `-100`), trained from a fresh base model copy.
  Saves the tokenizer/chat template alongside the checkpoint this time.

**Running:**
```bash
python rejection_sample.py --ckpt /path/to/grpo/checkpoints/final \
    --trained_tokenizer HuggingFaceTB/SmolLM2-360M-Instruct \
    --n_questions 500 --samples_per_question 4 --out sft_data.jsonl

python sft_train.py --data sft_data.jsonl \
    --base_model HuggingFaceTB/SmolLM2-360M-Instruct \
    --epochs 3 --out_dir sft_checkpoint
```

## Stage 3: second round of RL on the improved model

The actual R1 recipe doesn't stop at one SFT pass — run GRPO again, this
time starting from the stage-2 checkpoint instead of the raw base model.
Since the starting point already reasons more consistently, RL should have
an easier time finding and reinforcing further improvement than it did from
a cold start.

**No new code needed.** `grpo/train.py` already accepts any model path via
`--model_name`, including a local checkpoint directory — and `sft_train.py`
already saved the tokenizer/chat template alongside `sft_checkpoint`, so it
loads exactly like an HF repo id would:

```bash
python train.py --model_name /path/to/sft_checkpoint \
  --max_steps 1000 --group_size 8 --max_new_tokens 128
```

Same monitoring approach as stage 1 — watch `mean_reward` and `mean_kl`,
smoke-test with a handful of steps before committing to a full run, and
evaluate afterward with `eval_checkpoint.py` (pointing `--trained_tokenizer`
at `sft_checkpoint` rather than the original HF repo, since that's where the
correct chat template now lives) for a real three-way comparison against
both stage 1 and stage 2.
