"""
Evaluate a GRPO checkpoint against held-out GSM8K problems.
Run this on Colab/Kaggle, right where the checkpoint already lives --
no need to move a large checkpoint anywhere.

Usage:
    python eval_checkpoint.py --ckpt checkpoints/final --n_examples 50
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_gsm8k, format_prompt
from reward import reward_fn, extract_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to a checkpoint folder, e.g. checkpoints/final")
    parser.add_argument("--base_model", default="HuggingFaceTB/SmolLM2-360M", help="only used to load the tokenizer")
    parser.add_argument("--n_examples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.3, help="low temp for eval -- we want its best attempt, not diverse sampling")
    parser.add_argument("--print_samples", type=int, default=5, help="how many full completions to print")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint from {args.ckpt} ...")
    model = AutoModelForCausalLM.from_pretrained(args.ckpt).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # also load the raw base model for a side-by-side comparison -- this is the
    # real test of "did GRPO help at all" rather than a number in isolation
    print(f"Loading base model {args.base_model} for comparison ...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    base_model.eval()

    data = load_gsm8k("test")  # held-out split -- NOT what it trained on
    data = data[:args.n_examples]
    print(f"Evaluating on {len(data)} held-out test examples\n")

    def run_eval(m, label):
        correct = 0
        total_reward = 0.0
        samples_printed = 0
        for ex in data:
            prompt = format_prompt(ex["question"], tokenizer)
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = m.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=args.temperature, pad_token_id=tokenizer.pad_token_id,
                )
            completion = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            r = reward_fn(completion, ex["answer"])
            total_reward += r
            if r >= 1.0:
                correct += 1
            if samples_printed < args.print_samples:
                print(f"  [{label}] Q: {ex['question'][:80]}...")
                print(f"  [{label}] A (gt={ex['answer']}): {completion[:200]}")
                print(f"  [{label}] reward: {r:.2f}\n")
                samples_printed += 1

        acc = correct / len(data)
        avg_reward = total_reward / len(data)
        print(f"=== {label}: accuracy {acc:.1%} ({correct}/{len(data)}) | avg reward {avg_reward:.3f} ===\n")
        return acc, avg_reward

    print("--- BASE MODEL (no RL training) ---")
    base_acc, base_reward = run_eval(base_model, "base")

    print("--- GRPO-TRAINED CHECKPOINT ---")
    trained_acc, trained_reward = run_eval(model, "trained")

    print("=" * 60)
    print(f"Base model:     {base_acc:.1%} accuracy, {base_reward:.3f} avg reward")
    print(f"Trained model:  {trained_acc:.1%} accuracy, {trained_reward:.3f} avg reward")
    delta = trained_acc - base_acc
    print(f"Delta:          {delta:+.1%} accuracy")
    print("=" * 60)


if __name__ == "__main__":
    main()
