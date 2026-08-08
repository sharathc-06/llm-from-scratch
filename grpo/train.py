import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import copy
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import GRPOConfig
from data import load_gsm8k, format_prompt, SYNTHETIC_FALLBACK
from reward import reward_fn
from grpo import sample_group, score_group, normalize_advantages, compute_token_logprobs, grpo_loss


def save_checkpoint(path, model, optimizer, step):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    torch.save({"optimizer_state": optimizer.state_dict(), "step": step}, os.path.join(path, "train_state.pt"))
    print(f"[checkpoint] saved to {path} at step {step}")


def load_checkpoint(path, model, optimizer, device):
    model_loaded = AutoModelForCausalLM.from_pretrained(path).to(device)
    model.load_state_dict(model_loaded.state_dict())
    state = torch.load(os.path.join(path, "train_state.pt"), map_location=device)
    optimizer.load_state_dict(state["optimizer_state"])
    print(f"[checkpoint] resumed from {path} at step {state['step']}")
    return state["step"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    cfg = GRPOConfig()
    if args.model_name:
        cfg.model_name = args.model_name
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.group_size:
        cfg.group_size = args.group_size
    if args.max_new_tokens:
        cfg.max_new_tokens = args.max_new_tokens

    device = cfg.device or args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    amp_enabled = cfg.use_amp and device == "cuda"
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)

    print(f"Loading policy model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    
    # Cast reference model to FP16 to save ~1.4 GB VRAM
    ref_model = copy.deepcopy(policy).half().to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(args.resume_from, policy, optimizer, device)

    data = load_gsm8k("train")
    if data is SYNTHETIC_FALLBACK:
        print("\n" + "!" * 70)
        print("! WARNING: real GSM8K failed to load -- training on the TINY")
        print("! SYNTHETIC fallback set (6 trivial arithmetic questions).")
        print("! This is fine for a smoke test, but results are meaningless")
        print("! for a real run. Fix the dataset loading error above first.")
        print("!" * 70 + "\n")
    print(f"Loaded {len(data)} training examples")

    step = start_step
    data_idx = start_step * cfg.batch_prompts
    while step < cfg.max_steps:
        batch_loss = 0.0
        batch_reward = 0.0
        batch_kl = 0.0
        n_prompts = 0

        optimizer.zero_grad(set_to_none=True)

        for _ in range(cfg.batch_prompts):
            example = data[data_idx % len(data)]
            data_idx += 1
            prompt = format_prompt(example["question"], tokenizer)

            # Rollout collection
            policy.eval()
            with autocast():
                gen_ids, attn_mask, completion_mask, completions, prompt_len = sample_group(
                    policy, tokenizer, prompt, cfg, device
                )
            policy.train()

            rewards = score_group(completions, example["answer"], reward_fn)
            advantages = normalize_advantages(rewards).to(device)

            # Reference logprobs pass
            with torch.no_grad(), autocast():
                ref_logprobs = compute_token_logprobs(ref_model, gen_ids, attn_mask)

            # Policy forward pass (using detach to eliminate redundant forward pass)
            with autocast():
                new_logprobs = compute_token_logprobs(policy, gen_ids, attn_mask)
                old_logprobs = new_logprobs.detach()
                loss, mean_kl = grpo_loss(
                    new_logprobs, old_logprobs, ref_logprobs, advantages, completion_mask, cfg
                )

            (loss / cfg.batch_prompts).backward()

            batch_loss += loss.item()
            batch_reward += rewards.mean().item()
            batch_kl += mean_kl
            n_prompts += 1

            # Cache logging values locally before deleting tensors
            last_prompt = prompt
            last_completion = completions[0]
            last_reward = rewards[0].item()

            # Clean tensor references immediately so native allocator reuses memory
            del gen_ids, attn_mask, completion_mask, completions
            del rewards, advantages, old_logprobs, ref_logprobs
            del new_logprobs, loss

        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_every == 0:
            print(
                f"step {step:5d} | loss {batch_loss/n_prompts:.4f} "
                f"| mean_reward {batch_reward/n_prompts:.3f} | mean_kl {batch_kl/n_prompts:.4f}"
            )

        if step % cfg.log_samples_every == 0 and step > 0:
            print(f"  [sample] prompt: {last_prompt[:80]}...")
            print(f"  [sample] completion: {last_completion[:200]}")
            print(f"  [sample] reward: {last_reward:.2f}")

        if step % cfg.ckpt_every == 0 and step > 0:
            save_checkpoint(os.path.join(cfg.ckpt_dir, f"step_{step}"), policy, optimizer, step)

        step += 1

    save_checkpoint(os.path.join(cfg.ckpt_dir, "final"), policy, optimizer, cfg.max_steps)
    print("GRPO training complete.")


if __name__ == "__main__":
    main()
