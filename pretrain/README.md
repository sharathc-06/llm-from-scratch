# Phase 1: Pretraining with real training infra

A decoder-only GPT (same lineage as your `transformer_from_scratch` repo) plus the
infrastructure real training runs actually need: mixed precision, gradient
accumulation, a cosine LR schedule with warmup, and checkpoint save/resume.

Verified working end-to-end on this machine at small scale (tokenizer training,
data caching, forward/backward, checkpointing, and **resume-from-checkpoint all
tested and confirmed correct**). Swap in a bigger dataset and model size and
it's the same code path on a real GPU.

## Milestone 1: stable training loop, 1 epoch on TinyStories

That's the target for this phase — nothing fancier yet. You'll know you've
hit it when: loss decreases smoothly, val loss doesn't diverge from train
loss, and a checkpoint reload produces identical loss curves (proof your
resume logic is correct, not just that it runs).

## Files

- `model.py` — GPT model: causal self-attention, pre-norm transformer blocks,
  weight-tied embedding/output head, GPT-2-style scaled residual init.
- `data.py` — trains a byte-level BPE tokenizer (via `tokenizers`), caches the
  tokenized corpus as a memmapped `uint16` array, and serves random chunks.
- `config.py` — two dataclasses: `GPTConfig` (model shape) and `TrainConfig`
  (optimization/logging/checkpointing).
- `train.py` — the training loop: cosine LR w/ warmup, gradient accumulation,
  `torch.autocast` + `GradScaler` for mixed precision, periodic eval, periodic
  checkpointing, and `--resume_from` to pick back up.

## Getting real data (TinyStories)

This sandbox can't reach Hugging Face's dataset host, so it was smoke-tested
on tiny_shakespeare instead. On Colab/your own machine, get TinyStories like
this:

```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories', split='train')
with open('data/corpus.txt', 'w') as f:
    for row in ds:
        f.write(row['text'] + '\n')
"
```

That gives you `data/corpus.txt`, which is all `train.py` needs.

## Running it

```bash
pip install torch tokenizers

# first run: trains the tokenizer, caches tokens, starts training
python train.py --data_path data/corpus.txt

# resume after a crash / to keep going
python train.py --data_path data/corpus.txt --resume_from checkpoints/final.pt
```

Default `GPTConfig` is ~6 layers / 384 dim / 6 heads (~15-20M params) —
reasonable for a single T4 on TinyStories. Tune `TrainConfig` (batch size,
grad accum, max_steps, LR) to your GPU's memory and the total token budget
you want to spend.

Quick CPU/small-hardware smoke test (confirms your setup works before
committing to a real run):

```bash
python train.py --data_path data/corpus.txt --max_steps 20 \
  --micro_batch_size 4 --grad_accum_steps 2 --block_size 64 \
  --n_layer 2 --d_model 128 --n_head 4
```

## Next up (phase 2)

Once you've got a converged base model and can generate coherent short
stories with `model.generate(...)`, that checkpoint becomes the starting
point for instruction fine-tuning (SFT) — same `GPT` class, same
checkpoint format, new dataset and a loss mask over the prompt tokens.
