"""Convert the GSM8K / MedCalc files used by RL_exp into open-instruct's RLVR format for grpo_fast.py.

Each output row has:
    messages           prompt turns only ([user] for GSM8K, [system, user] for MedCalc)
    ground_truth       GSM8K: final answer string; MedCalc: JSON with the fields MedCalcVerifier grades on
    dataset            verifier name ("gsm8k" -> GSM8KVerifier, "medcalc" -> MedCalcVerifier)
    assistant_prefill  MedCalc only; appended after the generation prompt by rlvr_tokenize_prefill_v1

Usage (defaults are the paths used by RL_exp/core/data.py):
    python scripts/data/convert_rl_exp_to_rlvr.py --output_dir /net/spaces/scratch/lcpandia/data/processed/rlvr
"""

import argparse
import json
import os

# Must equal ASSISTANT_PREFIX in Forgetting/utils/prepare_medcalc.py; checked against every MedCalc row below.
# The trailing "\n" matches eval_medcalc_qwen.build_prompt (prefix_text + ASSISTANT_PREFIX + "\n").
MEDCALC_ASSISTANT_PREFIX = "Let me solve this step by step.\n<think>"

DEFAULTS = {
    "gsm8k_train": "/net/spaces/scratch/lcpandia/data/processed/llama3/gsm8k/perplexity_p0_olmo2_7B/100_perplexity_p0_olmo2_7B_gsm8k.jsonl",
    "gsm8k_test": "/home/lcpandia/Forgetting/open-instruct/data/eval/gsm8k_official_test.jsonl",
    "medcalc_train": "/net/spaces/scratch/lcpandia/data/processed/medcalc/MedCalc/train.jsonl",
    "medcalc_test": "/net/spaces/scratch/lcpandia/data/processed/medcalc/MedCalc/test.jsonl",
}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[OK] wrote {path} ({len(rows)} rows)")


def gsm8k_answer(solution):
    # Same target as RL_exp's GSM8KDataset: the text after "####", without thousands separators.
    if "####" not in solution:
        raise ValueError(f"GSM8K solution has no '####' final answer: {solution[-200:]!r}")
    return solution.split("####")[-1].strip().replace(",", "")


def convert_gsm8k(rows):
    return [
        {
            "messages": [row["messages"][0]],
            "ground_truth": gsm8k_answer(row["messages"][1]["content"]),
            "dataset": "gsm8k",
            "id": row.get("id", i),
        }
        for i, row in enumerate(rows)
    ]


def convert_medcalc(rows):
    converted = []
    for i, row in enumerate(rows):
        if not row["messages"][-1]["content"].startswith(MEDCALC_ASSISTANT_PREFIX):
            raise ValueError(
                f"Row {i}: assistant turn does not start with MEDCALC_ASSISTANT_PREFIX; re-sync it with prepare_medcalc.py."
            )
        target = {
            "ground_truth_answer": row["ground_truth_answer"],
            "category": row.get("category"),
            "output_type": row.get("output_type"),
            "lower_limit": row.get("lower_limit"),
            "upper_limit": row.get("upper_limit"),
        }
        converted.append(
            {
                "messages": row["messages"][:-1],
                "ground_truth": json.dumps(target),
                "dataset": "medcalc",
                "assistant_prefill": MEDCALC_ASSISTANT_PREFIX + "\n",
                "id": row.get("id", i),
                "calculator_name": row.get("calculator_name"),
            }
        )
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name}", default=default)
    parser.add_argument("--output_dir", default="/net/spaces/scratch/lcpandia/data/processed/rlvr")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "medcalc"], choices=["gsm8k", "medcalc"])
    args = parser.parse_args()

    converters = {"gsm8k": convert_gsm8k, "medcalc": convert_medcalc}
    for dataset in args.datasets:
        for split in ("train", "test"):
            rows = converters[dataset](read_jsonl(getattr(args, f"{dataset}_{split}")))
            write_jsonl(rows, os.path.join(args.output_dir, dataset, f"{split}.jsonl"))


if __name__ == "__main__":
    main()
