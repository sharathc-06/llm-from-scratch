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
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=split)
        examples = []
        for row in ds:
            gt = float(row["answer"].split("####")[-1].strip().replace(",", ""))
            examples.append({"question": row["question"], "answer": gt})
        return examples
    except Exception as e:
        print(f"[data] could not load GSM8K ({e}); using tiny synthetic fallback for local testing")
        return SYNTHETIC_FALLBACK


def format_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "End your response with '#### <final numeric answer>'.\n\n"
        f"Question: {question}\nAnswer:"
    )
