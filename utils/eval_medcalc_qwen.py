"""
MedCalc-Bench zero-shot evaluation (no fine-tuning).

Protocol:
  - Prompt: same system/user/assistant-prefix template as prepare_medcalc.py
    (imported from there so this evaluator and the SFT-data builder can't
    drift onto different prompt wording) -- <think>/<answer> contract per
    Lin et al. 2025 (arXiv:2509.20758) Table 6.
  - Grading: medcalc_grading.grade() -- Khandekar et al. 2024
    (arXiv:2406.12036) Sec 3.1: rule-based (risk/severity/diagnosis) and date
    calculators use exact match; equation-based lab/physical/dosage use
    within-5%-of-ground-truth (or the dataset's own bounds when present).

Usage:
  pip install vllm datasets python-dateutil

  # against the gated ncbi/MedCalc-Bench-v1.0 hub dataset
  python eval_medcalc_qwen.py --model Qwen/Qwen2.5-7B-Instruct --out init.json

  # against a local {dataset,id,messages,...} jsonl file, e.g. the
  # test.jsonl produced by prepare_medcalc.py
  python eval_medcalc_qwen.py --model Qwen/Qwen2.5-7B-Instruct \\
      --test_file /path/to/MedCalc/test.jsonl --out init.json
"""

import argparse
import json
from collections import defaultdict

from datasets import load_dataset

from medcalc_grading import extract_answer, grade
from prepare_medcalc import ASSISTANT_PREFIX, SYSTEM_PROMPT, USER_TEMPLATE


def build_prompt(tokenizer, row):
    """Builds the zero-shot prompt for one row.

    Rows loaded from a local jsonl (prepare_medcalc.py's output) already carry
    a full "messages" list -- system+user+assistant -- so only messages[:-1]
    (system+user) is rendered, matching eval_medcalc.py's approach. Rows
    loaded from the raw ncbi/MedCalc-Bench-v1.0 hub dataset have no such
    turn-list, only "note"/"question" strings, so the same system/user
    template is rendered from scratch in that case.
    """

    if row.get("messages") is not None:
        prefix_text = tokenizer.apply_chat_template(
            row["messages"][:-1], add_generation_prompt=True, tokenize=False
        )
    else:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(note=row["note"], question=row["question"])},
        ]
        prefix_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    return prefix_text + ASSISTANT_PREFIX + "\n"


# ----------------------------------------------------------------------------
# Dataset plumbing
#   Column names in ncbi/MedCalc-Bench-v1.0 have varied across revisions, so
#   this is checked against ds.column_names right after loading (see main())
#   rather than failing silently mid-run.
# ----------------------------------------------------------------------------

COLS = {
    "note": "Patient Note",
    "question": "Question",
    "ground_truth": "Ground Truth Answer",
    "category": "Category",
    "calculator": "Calculator Name",
    "output_type": "Output Type",
    "lower_limit": "Lower Limit",
    "upper_limit": "Upper Limit",
}


def check_columns(column_names):
    missing = [col for col in COLS.values() if col not in column_names]
    if missing:
        raise ValueError(
            f"Dataset is missing expected column(s) {missing}. "
            f"Available columns: {column_names}. Update COLS in this script to match."
        )


def normalize_row(ex):
    return {key: ex[col] for key, col in COLS.items()}


def load_local_jsonl(path):
    """Loads a {dataset,id,messages,...} jsonl file, e.g. prepare_medcalc.py's
    test.jsonl, into the same internal row shape normalize_row() produces
    (minus "note"/"question", which build_prompt() doesn't need here since
    "messages" already has the rendered user turn).
    """

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(
                {
                    "messages": r["messages"],
                    "ground_truth": r.get("ground_truth_answer"),
                    "category": r.get("category"),
                    "calculator": r.get("calculator_name"),
                    "output_type": r.get("output_type"),
                    "lower_limit": r.get("lower_limit"),
                    "upper_limit": r.get("upper_limit"),
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--test_file", type=str, default=None,
                     help="local {dataset,id,messages,...} jsonl file (e.g. prepare_medcalc.py's "
                          "test.jsonl); if omitted, falls back to the gated ncbi/MedCalc-Bench-v1.0 "
                          "hub dataset via --split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N instances")
    ap.add_argument("--out", default="medcalc_init.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    if args.test_file:
        rows = load_local_jsonl(args.test_file)
    else:
        ds = load_dataset("ncbi/MedCalc-Bench-v1.0", split=args.split)
        print("columns:", ds.column_names)
        check_columns(ds.column_names)
        rows = [normalize_row(ex) for ex in ds]

    if args.limit:
        rows = rows[: args.limit]

    if not rows:
        print("[WARN] no rows to evaluate")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [build_prompt(tokenizer, r) for r in rows]

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192)
    sp = SamplingParams(
        temperature=0.0,                      # greedy: reproducible Init point
        max_tokens=args.max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sp)

    # Grade.
    n_correct = n_parsed = 0
    by_cat = defaultdict(lambda: {"n": 0, "correct": 0, "parsed": 0})
    records = []

    for row, out in zip(rows, outputs):
        completion = out.outputs[0].text
        pred = extract_answer(completion)
        ok = grade(
            pred,
            ground_truth=row["ground_truth"],
            category=row.get("category"),
            output_type=row.get("output_type"),
            lower_limit=row.get("lower_limit"),
            upper_limit=row.get("upper_limit"),
        )

        cat = str(row["category"])
        by_cat[cat]["n"] += 1
        by_cat[cat]["correct"] += int(ok)
        by_cat[cat]["parsed"] += int(pred is not None)
        n_correct += int(ok)
        n_parsed += int(pred is not None)

        records.append({
            "calculator": row["calculator"],
            "category": cat,
            "gt": row["ground_truth"],
            "pred": pred,
            "correct": ok,
            "completion": completion,      # keep: needed for error-type analysis
        })

    n = len(rows)
    print(f"\nmodel: {args.model}   n={n}")
    print(f"accuracy   {n_correct / n:.4f}")
    print(f"parse rate {n_parsed / n:.4f}")
    print(f"\n{'category':<24}{'n':>6}{'acc':>9}{'parse':>9}")
    for cat in sorted(by_cat):
        s = by_cat[cat]
        print(f"{cat:<24}{s['n']:>6}{s['correct']/s['n']:>9.4f}{s['parsed']/s['n']:>9.4f}")

    with open(args.out, "w") as f:
        json.dump({
            "model": args.model,
            "n": n,
            "accuracy": n_correct / n,
            "parse_rate": n_parsed / n,
            "by_category": {k: dict(v) for k, v in by_cat.items()},
            "records": records,
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()