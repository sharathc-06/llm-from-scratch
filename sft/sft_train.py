"""
Stage 2, part 2: supervised fine-tuning on rejection-sampled data.

Fine-tunes a FRESH copy of the base instruct model (not the GRPO checkpoint --
starting clean, per the actual R1 recipe) on the model's own correct
completions from rejection_sample.py. Standard SFT: cross-entropy loss on
completion tokens only, prompt tokens masked out (-100).

Usage:
    python sft_train.py --data sft_data.jsonl \
        --base_model HuggingFaceTB/SmolLM2-360M-Instruct \
        --epochs 3 --out_dir sft_checkpoint
"""
import argparse
import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import format_prompt


class RejectionSampledDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=384):
        self.examples = []
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = format_prompt(ex["question"], self.tokenizer)
        full_text = prompt + ex["completion"] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"][: self.max_length]

        # mask out prompt tokens (-100 is the standard "ignore this position" label in
        # PyTorch's cross-entropy loss) -- only the completion should contribute to loss
        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(full_ids))
        for i in range(prompt_len):
            labels[i] = -100

        return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_token_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention_mask = [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(b["input_ids"]) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="jsonl file from rejection_sample.py")
    parser.add_argument("--base_model", default="HuggingFaceTB/SmolLM2-360M-Instruct",
                         help="starts from a FRESH copy of this model, not the GRPO checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5, help="standard SFT LR -- much higher than GRPO's, since this is ordinary supervised learning")
    parser.add_argument("--out_dir", default="sft_checkpoint")
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading FRESH base model {args.base_model} (not the GRPO checkpoint)")
    model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    model.train()

    dataset = RejectionSampledDataset(args.data, tokenizer)
    print(f"Loaded {len(dataset)} rejection-sampled examples")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_enabled = device == "cuda"

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                out = model(**batch)
                loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step:5d} | loss {loss.item():.4f}")
            step += 1

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)  # save the tokenizer/chat template this time -- fixes last stage's gap
    print(f"Saved SFT checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
