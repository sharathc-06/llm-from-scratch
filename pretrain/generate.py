"""Load a phase-1 checkpoint and generate text. Usage:
    python generate.py --ckpt final.pt --tokenizer tokenizer.json --prompt "Once upon a time"
"""
import argparse
import torch
from model import GPT
from config import GPTConfig
from tokenizers import Tokenizer, decoders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    gcfg = GPTConfig(**ckpt["gpt_config"])
    model = GPT(gcfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint at step {ckpt['step']} ({model.num_params()/1e6:.2f}M params)")

    tok = Tokenizer.from_file(args.tokenizer)
    tok.decoder = decoders.ByteLevel()  # strips Ġ/Ċ byte-level BPE artifacts

    ids = tok.encode(args.prompt).ids
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
