"""Download and format nsk7153/MedCalc-Bench-Verified for SFT training.

Downloads the dataset's train/test splits (the tiny "one_shot" split is a
few-shot exemplar set from the original benchmark, not training data, and is
skipped) and reshapes each row into a three-turn system/user/assistant chat
example using the "think step by step, then answer in JSON" template:

    system:    fixed instruction to reason privately, then answer
    user:      the patient note + calculation question, with formatting
               instructions for a single <think>...</think> block and a
               final <answer>{"answer": ...}</answer> JSON block
    assistant: "Let me solve this step by step.\\n<think>" followed by the
               dataset's own Ground Truth Explanation (closing </think>) and
               the Ground Truth Answer wrapped as the required answer JSON

This makes each row a complete supervised input/target pair (not a bare
generation prompt), in the same {"dataset", "id", "messages"} schema used
elsewhere in this pipeline (see utils/pull_hf_splits.py).

Usage:
    python utils/prepare_medcalc.py
    python utils/prepare_medcalc.py --output_dir /some/other/path
"""

import argparse
import json
import os

from datasets import load_dataset

SOURCE_DATASET = "nsk7153/MedCalc-Bench-Verified"

SYSTEM_PROMPT = (
    "You are a helpful assistant. You first think about the reasoning "
    "process in the mind and then provide the user with the answer."
)

USER_TEMPLATE = """You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score.
Here is the patient note:
{note}

Here is the task:
{question}

Please show your entire reasoning process in a single <think></think> block (do not open or close the tag more than once). Your final response must be in JSON format within <answer></answer> tags. For example,
<think>
[entire reasoning process here]
</think>

<answer>
{{"answer": str(short_and_direct_answer_of_the_question)}}
</answer>"""

ASSISTANT_PREFIX = "Let me solve this step by step.\n<think>"


def build_messages(example):
    user_content = USER_TEMPLATE.format(note=example["Patient Note"], question=example["Question"])

    explanation = (example["Ground Truth Explanation"] or "").strip()
    answer_json = json.dumps({"answer": example["Ground Truth Answer"]})
    assistant_content = f"{ASSISTANT_PREFIX}\n{explanation}\n</think>\n\n<answer>\n{answer_json}\n</answer>"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def to_sft_schema(dataset):
    def _map(example, idx):
        return {
            "dataset": "medcalc",
            "id": idx,
            "messages": build_messages(example),
            "calculator_name": example["Calculator Name"],
            "category": example["Category"],
            "output_type": example["Output Type"],
            "ground_truth_answer": example["Ground Truth Answer"],
            "lower_limit": example["Lower Limit"],
            "upper_limit": example["Upper Limit"],
        }

    return dataset.map(_map, with_indices=True, remove_columns=dataset.column_names)


def write_jsonl(dataset, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.to_json(path, orient="records", lines=True)


def main():
    parser = argparse.ArgumentParser(description="Download and format MedCalc-Bench-Verified for SFT")
    parser.add_argument("--output_dir", type=str, default="/net/spaces/scratch/lcpandia/data/processed")
    args = parser.parse_args()

    ds = load_dataset(SOURCE_DATASET)
    out_dir = os.path.join(args.output_dir, "MedCalc")

    for split_name in ("train", "test"):
        if split_name not in ds:
            print(f"[WARN] split '{split_name}' not found in {SOURCE_DATASET}, skipping")
            continue

        sft_ds = to_sft_schema(ds[split_name])
        out_path = os.path.join(out_dir, f"{split_name}.jsonl")
        write_jsonl(sft_ds, out_path)
        print(f"[OK] wrote {out_path} ({sft_ds.num_rows} rows)")


if __name__ == "__main__":
    main()
