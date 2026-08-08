"""
Verifiable reward for math problems: no learned reward model, no human labels.
We parse the final numeric answer out of the model's generation and compare it
directly to ground truth. This is the entire "reward signal" GRPO trains against.
"""
import re
from typing import Optional


def extract_answer(text: str) -> Optional[float]:
    """
    Pull a final numeric answer out of free-form text. Priority order:
      1. explicit '#### <num>' marker (the format we prompt the model to use)
      2. 'answer is/: <num>' phrasing
      3. fall back to the last number that appears anywhere in the text
    Returns None if no number can be found at all -- that's a reward of 0.
    """
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if m:
        raw = m.group(1)
    else:
        m = re.search(r"(?:answer|Answer)\s*(?:is|:)?\s*(-?\d[\d,]*\.?\d*)", text)
        if m:
            raw = m.group(1)
        else:
            nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
            if not nums:
                return None
            raw = nums[-1]

    raw = raw.replace(",", "").strip().rstrip(".")
    try:
        return float(raw)
    except ValueError:
        return None


def has_explicit_marker(text: str) -> bool:
    """True only if the '#### <number>' format was actually used -- as opposed
    to extract_answer's looser fallback (grabbing the last number anywhere in
    the text), which can accidentally match a correct final number even when
    the reasoning leading up to it was muddled, contradictory, or hallucinated.
    Useful for filtering SFT training data, where you're teaching the model to
    literally reproduce the completion -- a stricter bar than scoring RL reward."""
    return re.search(r"####\s*-?\d[\d,]*\.?\d*", text) is not None


def reward_fn(response: str, ground_truth: float) -> float:
    """
    Core verifiable reward:
      +1.0  if the extracted final answer matches ground truth
      +0.1  bonus for using the '####' format we asked for (rewards following
            instructions, independent of correctness -- helps early on when
            correctness reward is almost always 0)
       0.0  otherwise
    Deliberately simple. Reward hacking risk lives right here: if this function
    has a loophole (e.g. matching too loosely), GRPO *will* find and exploit it.
    """
    reward = 0.0
    pred = extract_answer(response)
    if pred is not None and ground_truth is not None:
        if abs(pred - ground_truth) < 1e-4:
            reward += 1.0
    if "####" in response:
        reward += 0.1
    return reward


if __name__ == "__main__":
    # quick manual sanity checks -- run this file directly to sanity-check the reward fn
    cases = [
        ("The answer is 5. #### 5", 5.0, 1.1),
        ("I think it's 42", 42.0, 1.0),
        ("Let's see... 3 + 4 = 7. #### 7", 5.0, 0.1),
        ("I have no idea", 5.0, 0.0),
    ]
    for text, gt, expected in cases:
        r = reward_fn(text, gt)
        status = "OK" if abs(r - expected) < 1e-6 else "MISMATCH"
        print(f"[{status}] reward={r:.2f} expected={expected:.2f} | {text!r} vs gt={gt}")