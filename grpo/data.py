"""
GSM8K loading, with a tiny synthetic fallback so the pipeline can be smoke-tested
without network access to the real dataset.
"""

SYNTHETIC_FALLBACK = [
    {"question": "What is 2 + 3?", "answer": 5.0},
    {"question": "What is 10 - 4?", "answer": 6.0},
    {"question": "What is 6 * 7?", "answer": 42.0},
    {"question": "What is 15 / 3?", "answer": 5.0},
    {"question": "What is 8 + 9?", "answer": 17.0},
    {"question": "What is 20 - 12?", "answer": 8.0},
]


def load_gsm8k(split: str = "train"):
    # the datasets library's naming for this repo has shifted before --
    # try the current canonical id first, then the older bare name, then fall
    # back to the synthetic set if neither loads (e.g. offline)
    last_error = None
    for repo_id in ["openai/gsm8k", "gsm8k"]:
        try:
            from datasets import load_dataset
            ds = load_dataset(repo_id, "main", split=split)
            examples = []
            for row in ds:
                gt = float(row["answer"].split("####")[-1].strip().replace(",", ""))
                examples.append({"question": row["question"], "answer": gt})
            print(f"[data] loaded GSM8K from '{repo_id}': {len(examples)} examples")
            return examples
        except Exception as e:
            last_error = e
            continue
    print(f"[data] could not load GSM8K from any known repo id ({last_error}); using tiny synthetic fallback")
    return SYNTHETIC_FALLBACK


def format_prompt(question: str, tokenizer=None) -> str:
    """
    Format a GSM8K question into a prompt. If a tokenizer with a chat template
    is passed (true for instruct models), use it -- instruct models are
    trained to expect the specific turn-formatting tokens their chat template
    inserts, and without it they often don't recognize they're being asked to
    respond at all (frequently emitting an end-of-turn token almost
    immediately instead of attempting an answer).
    """
    instruction = (
        "Solve the following math problem step by step. "
        "End your response with '#### <final numeric answer>'.\n\n"
        f"Question: {question}"
    )
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{instruction}\nAnswer:"
