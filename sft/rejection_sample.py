"""
Stage 2, part 1: rejection sampling.

Take the GRPO-trained checkpoint, generate multiple completions per training
question, and keep only the ones that got the right answer. This is how the
model's own successful reasoning becomes a fresh SFT dataset -- no human
wrote these examples, the model wrote them itself and we filtered for
correctness.

Usage:
    python rejection_sample.py --ckpt checkpoints/final \
        --trained_tokenizer HuggingFaceTB/SmolLM2-360M-Instruct \
        --n_questions 500 --samples_per_question 4 \
        --out sft_data.jsonl
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_gsm8k, format_prompt
from reward import reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="the GRPO stage-1 checkpoint to sample from")
    parser.add_argument("--trained_tokenizer", default="HuggingFaceTB/SmolLM2-360M-Instruct",
                         help="must match cfg.model_name from the GRPO training run")
    parser.add_argument("--n_questions", type=int, default=500, help="how many training questions to sample from")
    parser.add_argument("--samples_per_question", type=int, default=4,
                         help="completions attempted per question -- higher = more chances to catch a correct one")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8, help="need diversity across samples to catch correct ones")
    parser.add_argument("--out", default="sft_data.jsonl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint from {args.ckpt} ...")
    model = AutoModelForCausalLM.from_pretrained(args.ckpt).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.trained_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_gsm8k("train")
    data = data[: args.n_questions]
    print(f"Sampling from {len(data)} training questions, {args.samples_per_question} attempts each")

    kept = 0
    attempted = 0
    with open(args.out, "w") as f:
        for i, ex in enumerate(data):
            prompt = format_prompt(ex["question"], tokenizer)
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            enc_repeated = {k: v.repeat(args.samples_per_question, 1) for k, v in enc.items()}

            with torch.no_grad():
                out = model.generate(
                    **enc_repeated, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=args.temperature, pad_token_id=tokenizer.pad_token_id,
                )
            completions = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

            for completion in completions:
                attempted += 1
                r = reward_fn(completion, ex["answer"])
                if r >= 1.0:  # only keep genuinely correct completions, not format-only partial credit
                    f.write(json.dumps({
                        "question": ex["question"],
                        "answer": ex["answer"],
                        "completion": completion.strip(),
                    }) + "\n")
                    kept += 1

            if i % 20 == 0:
                rate = kept / max(attempted, 1)
                print(f"  [{i}/{len(data)}] kept {kept}/{attempted} so far ({rate:.1%} accept rate)")

    print(f"\nDone. Kept {kept}/{attempted} completions ({kept/max(attempted,1):.1%} accept rate) -> {args.out}")
    if kept < 50:
        print("WARNING: fewer than 50 examples kept -- this is a thin SFT dataset. Consider raising "
              "--n_questions or --samples_per_question to gather more before training on it.")


if __name__ == "__main__":
    main()
