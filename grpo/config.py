from dataclasses import dataclass
from typing import Optional


@dataclass
class GRPOConfig:
    # model -- instruct variant matters: base models don't know to stop after
    # answering, which pollutes both reward scoring and gradient credit
    # assignment (see grpo.py's _truncate_at_answer for the mitigation, but
    # starting from a model that already knows to terminate is the real fix)
    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"

    # rollout / sampling
    group_size: int = 8          # completions sampled per prompt (this replaces PPO's critic)
    max_prompt_tokens: int = 256
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50

    # GRPO loss
    kl_coef: float = 0.04        # penalty weight against the frozen reference model
    clip_eps: float = 0.2        # PPO-style clipping on the probability ratio
    inner_epochs: int = 1        # gradient updates per rollout batch (>1 = reuse rollouts, PPO-style)

    # optimization
    lr: float = 1e-6             # RL fine-tuning LR is much smaller than pretraining LR
    max_steps: int = 1000
    batch_prompts: int = 4       # distinct prompts sampled per training step
    grad_clip: float = 1.0

    # mixed precision -- meaningfully helps on Tensor-Core GPUs (T4, A10, etc.),
    # not on P100 (no Tensor Cores) or CPU
    use_amp: bool = True

    # logging / checkpointing
    log_every: int = 5
    ckpt_every: int = 100
    ckpt_dir: str = "checkpoints"
    log_samples_every: int = 20  # print a full generated sample this often

    seed: int = 1337
    device: Optional[str] = None  # auto-detected if None
