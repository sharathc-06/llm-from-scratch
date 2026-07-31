import os
import numpy as np
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


def train_tokenizer(text_path: str, out_path: str, vocab_size: int = 8000):
    """Train a byte-level BPE tokenizer on a text file and save it to out_path."""
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    tok.train([text_path], trainer)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tok.save(out_path)
    return tok


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def build_token_array(text_path: str, tokenizer: Tokenizer, cache_path: str) -> np.ndarray:
    """Tokenize the whole corpus once and cache it as a uint16 numpy array."""
    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text).ids
    arr = np.array(ids, dtype=np.uint16)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, arr)
    return arr


class TokenDataset:
    """
    Serves random contiguous chunks of a large token array.
    Not a torch Dataset/DataLoader on purpose -- for language-model pretraining,
    random-offset sampling directly from a memmapped array is simpler and faster
    than the usual Dataset/__getitem__/collate machinery.
    """

    def __init__(self, token_array: np.ndarray, block_size: int, split: str = "train", val_fraction: float = 0.01):
        n = len(token_array)
        split_idx = int(n * (1 - val_fraction))
        if split == "train":
            self.data = token_array[:split_idx]
        else:
            self.data = token_array[split_idx:]
        self.block_size = block_size

    def get_batch(self, batch_size: int, device: str):
        ix = torch.randint(len(self.data) - self.block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if device == "cuda":
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
