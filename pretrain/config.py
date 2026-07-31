from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 8000
    block_size: int = 256        # max context length (tokens)
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 4 * 384
    dropout: float = 0.1


@dataclass
class TrainConfig:
    # data
    data_path: str = "data/corpus.txt"
    tokenizer_path: str = "data/tokenizer.json"

    # optimization
    max_steps: int = 20000
    micro_batch_size: int = 32      # per forward/backward pass
    grad_accum_steps: int = 4       # effective batch = micro_batch_size * grad_accum_steps
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)

    # mixed precision
    use_amp: bool = True

    # logging / checkpointing
    log_every: int = 20
    eval_every: int = 250
    eval_iters: int = 50
    ckpt_every: int = 500
    ckpt_dir: str = "checkpoints"
    resume_from: str | None = None   # path to a .pt checkpoint to resume from

    # misc
    seed: int = 1337
