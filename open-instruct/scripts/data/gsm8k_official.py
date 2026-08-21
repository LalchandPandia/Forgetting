"""
Build local GSM8K train/test jsonl files straight from the official dataset,
in the same schema gsm8k_eval.py and the SFT pipeline already consume
(messages + a "Question: ...\\nAnswer: " wrapper on the user turn -- see
get_question()/get_gold_answer() in open_instruct/gsm8k_eval.py).

Unlike scripts/data/gsm8k.py (which pushes a differently-shaped dataset to the
HF Hub), this writes local files and keeps the exact row shape already in use
operationally (e.g. data/processed/100_gsm8k_test.jsonl), just sourced from
the full, unmodified splits instead of an ad hoc subsample -- as close to
official GSM8K as this pipeline's format allows.

Usage:
    python scripts/data/gsm8k_official.py \
        --train_out data/sft/gsm8k_official_train.jsonl \
        --test_out data/eval/gsm8k_official_test.jsonl
"""
import argparse
import json
import os

import datasets


def to_row(idx, example):
    return {
        "dataset": "gsm8k",
        "id": idx,
        "messages": [
            {"role": "user", "content": f"Question: {example['question']}\nAnswer: "},
            {"role": "assistant", "content": example["answer"]},
        ],
    }


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_out", default="data/sft/gsm8k_official_train.jsonl")
    ap.add_argument("--test_out", default="data/eval/gsm8k_official_test.jsonl")
    args = ap.parse_args()

    dataset = datasets.load_dataset("gsm8k", "main")

    write_jsonl(args.train_out, [to_row(i, ex) for i, ex in enumerate(dataset["train"])])
    write_jsonl(args.test_out, [to_row(i, ex) for i, ex in enumerate(dataset["test"])])


if __name__ == "__main__":
    main()
