"""
GRPO from scratch -- no trl, no RL library. The whole point of building this
yourself is understanding these five pieces well enough to debug them:

  1. sample_group      -- for one prompt, sample `group_size` completions
  2. score_group        -- run the verifiable reward function on each
  3. normalize_advantages -- group-relative advantage (this replaces PPO's critic)
  4. compute_token_logprobs -- per-token log p(token | context) under a model
  5. grpo_loss          -- clipped surrogate objective + KL penalty vs reference

A group of completions to the SAME prompt is what "group-relative" means:
rewards are normalized within that group, not against a global baseline, which
is exactly what removes the need for a learned value/critic network that PPO
requires.
"""
import re
import torch
import torch.nn.functional as F


def _truncate_at_answer(text: str, tokenizer, prompt_len: int):
    """
    Base (non-instruct) models don't know to stop after answering -- they'll
    keep rambling into a hallucinated new question. If we don't cut that off,
    two things break: reward scoring grabs the wrong (later) number, and the
    model gets gradient signal for tokens it generated *after* already
    answering, polluting credit assignment.

    Truncates `text` right after the first '#### <number>' match. Returns the
    truncated text and how many generated tokens that corresponds to (by
    re-encoding the truncated text -- an approximation for BPE, but accurate
    enough for masking purposes). If no '####' is found, returns the text and
    length unchanged (that completion gets 0 reward anyway, from reward_fn).
    """
    m = re.search(r"####\s*-?\d[\d,]*\.?\d*", text)
    if not m:
        return text, None  # no truncation possible/needed
    truncated_text = text[: m.end()]
    truncated_ids = tokenizer(truncated_text, add_special_tokens=False)["input_ids"]
    return truncated_text, len(truncated_ids)


@torch.no_grad()
def sample_group(model, tokenizer, prompt: str, cfg, device):
    """Sample cfg.group_size completions for a single prompt. Returns padded
    input_ids/attention_mask for the full (prompt+completion) sequences, a
    completion_mask marking which positions are generated (not prompt/pad/
    post-answer rambling), and the decoded+truncated completion texts (for
    reward scoring)."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_prompt_tokens)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    # repeat the same prompt group_size times -- this is the "group" in GRPO
    input_ids = input_ids.repeat(cfg.group_size, 1)
    attention_mask = attention_mask.repeat(cfg.group_size, 1)

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        pad_token_id=tokenizer.pad_token_id,
    )

    full_attention_mask = (gen != tokenizer.pad_token_id).long()
    completion_mask = torch.zeros_like(gen)
    completion_mask[:, prompt_len:] = full_attention_mask[:, prompt_len:]

    raw_completions = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
    completions = []
    for i, text in enumerate(raw_completions):
        truncated_text, cutoff_len = _truncate_at_answer(text, tokenizer, prompt_len)
        completions.append(truncated_text)
        if cutoff_len is not None:
            # zero out mask positions after the answer -- no gradient credit/
            # blame for tokens generated after the model already answered
            completion_mask[i, prompt_len + cutoff_len:] = 0

    return gen, full_attention_mask, completion_mask, completions, prompt_len


def score_group(completions, ground_truth, reward_fn):
    """Run the verifiable reward function on each completion in the group."""
    return torch.tensor([reward_fn(c, ground_truth) for c in completions], dtype=torch.float32)


def normalize_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Group-relative advantage: (reward - group mean) / group std.
    This is GRPO's replacement for a learned critic -- the group itself
    provides the baseline. If every completion in a group gets the same
    reward (e.g. all wrong), advantage is ~0 for all of them: no signal,
    which is exactly correct -- there's nothing to prefer within the group."""
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)


def compute_token_logprobs(model, input_ids, attention_mask):
    """Per-token log p(token_t | tokens_<t}) for every position, shifted so
    logprobs[:, t] is the log-prob of predicting input_ids[:, t+1]."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logprobs  # (B, T-1)


def grpo_loss(new_logprobs, old_logprobs, ref_logprobs, advantages, completion_mask, cfg):
    """
    Clipped surrogate policy objective (same form as PPO) plus a KL penalty
    against a frozen reference model (keeps the policy from drifting into
    degenerate text that happens to fool the reward function).

    advantages: (B,) one scalar per sequence -> broadcast across its tokens
    completion_mask: (B, T-1) shifted to align with the logprob tensors
    """
    mask = completion_mask[:, 1:].float()  # align with the T-1 shift
    adv = advantages.unsqueeze(1)  # (B, 1) broadcasts across tokens

    ratio = torch.exp(new_logprobs - old_logprobs)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2)

    # k3 KL estimator (Schulman): unbiased, always >= 0, lower variance than
    # the naive (logp - logp_ref) difference
    log_ratio_ref = ref_logprobs - new_logprobs
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1

    per_token_loss = policy_loss + cfg.kl_coef * kl
    denom = mask.sum().clamp(min=1.0)
    loss = (per_token_loss * mask).sum() / denom
    mean_kl = (kl * mask).sum() / denom
    return loss, mean_kl.item()
