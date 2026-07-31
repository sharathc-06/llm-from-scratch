import os
import math
import time
import argparse

import torch

from config import GPTConfig, TrainConfig
from model import GPT
from data import train_tokenizer, load_tokenizer, build_token_array, TokenDataset


def get_lr(step, cfg: TrainConfig):
    if step < cfg.warmup_steps:
        return cfg.lr_max * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.lr_min
    decay_ratio = (step - cfg.warmup_steps) / max(1, (cfg.max_steps - cfg.warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # cosine decay to lr_min
    return cfg.lr_min + coeff * (cfg.lr_max - cfg.lr_min)


@torch.no_grad()
def estimate_loss(model, dataset_train, dataset_val, tcfg: TrainConfig, device):
    model.eval()
    out = {}
    for name, ds in [("train", dataset_train), ("val", dataset_val)]:
        losses = torch.zeros(tcfg.eval_iters)
        for i in range(tcfg.eval_iters):
            x, y = ds.get_batch(tcfg.micro_batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path, model, optimizer, scaler, step, gcfg, tcfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "gpt_config": gcfg.__dict__,
    }, path)
    print(f"[checkpoint] saved to {path} at step {step}")


def load_checkpoint(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    print(f"[checkpoint] resumed from {path} at step {ckpt['step']}")
    return ckpt["step"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None, help="override GPTConfig.block_size (e.g. for quick CPU smoke tests)")
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    args = parser.parse_args()

    gcfg = GPTConfig()
    tcfg = TrainConfig()
    if args.data_path:
        tcfg.data_path = args.data_path
    if args.max_steps:
        tcfg.max_steps = args.max_steps
    if args.resume_from:
        tcfg.resume_from = args.resume_from
    if args.micro_batch_size:
        tcfg.micro_batch_size = args.micro_batch_size
    if args.grad_accum_steps:
        tcfg.grad_accum_steps = args.grad_accum_steps
    if args.block_size:
        gcfg.block_size = args.block_size
    if args.n_layer:
        gcfg.n_layer = args.n_layer
    if args.d_model:
        gcfg.d_model = args.d_model
        gcfg.d_ff = 4 * args.d_model
    if args.n_head:
        gcfg.n_head = args.n_head

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(tcfg.seed)

    # --- tokenizer ---
    if not os.path.exists(tcfg.tokenizer_path):
        print("Training BPE tokenizer...")
        tokenizer = train_tokenizer(tcfg.data_path, tcfg.tokenizer_path, vocab_size=gcfg.vocab_size)
    else:
        tokenizer = load_tokenizer(tcfg.tokenizer_path)
    gcfg.vocab_size = tokenizer.get_vocab_size()

    # --- data ---
    cache_path = tcfg.data_path + ".ids.npy"
    token_array = build_token_array(tcfg.data_path, tokenizer, cache_path)
    ds_train = TokenDataset(token_array, gcfg.block_size, split="train")
    ds_val = TokenDataset(token_array, gcfg.block_size, split="val")
    print(f"Corpus: {len(token_array):,} tokens | vocab: {gcfg.vocab_size}")

    # --- model / optimizer ---
    model = GPT(gcfg).to(device)
    print(f"Model params: {model.num_params() / 1e6:.2f}M")
    optimizer = model.configure_optimizer(tcfg.weight_decay, tcfg.lr_max, tcfg.betas)
    scaler = torch.amp.GradScaler(device, enabled=(tcfg.use_amp and device == "cuda"))
    amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    start_step = 0
    if tcfg.resume_from and os.path.exists(tcfg.resume_from):
        start_step = load_checkpoint(tcfg.resume_from, model, optimizer, scaler)

    t0 = time.time()
    for step in range(start_step, tcfg.max_steps):
        lr = get_lr(step, tcfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro_step in range(tcfg.grad_accum_steps):
            x, y = ds_train.get_batch(tcfg.micro_batch_size, device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(tcfg.use_amp and device == "cuda")):
                _, loss = model(x, y)
                loss = loss / tcfg.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % tcfg.log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_sec = (tcfg.micro_batch_size * tcfg.grad_accum_steps * gcfg.block_size * tcfg.log_every) / max(dt, 1e-6)
            print(f"step {step:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | {tok_per_sec:,.0f} tok/s")

        if step % tcfg.eval_every == 0 and step > 0:
            losses = estimate_loss(model, ds_train, ds_val, tcfg, device)
            print(f"  [eval] step {step} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        if step % tcfg.ckpt_every == 0 and step > 0:
            save_checkpoint(os.path.join(tcfg.ckpt_dir, f"step_{step}.pt"), model, optimizer, scaler, step, gcfg, tcfg)

    save_checkpoint(os.path.join(tcfg.ckpt_dir, "final.pt"), model, optimizer, scaler, tcfg.max_steps, gcfg, tcfg)
    print("Training complete.")


if __name__ == "__main__":
    main()
