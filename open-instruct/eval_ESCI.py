"""
ESCI (Amazon Shopping Queries Dataset) zero-shot evaluation.

Reads the same jsonl format create_ESCI.py produces:
  {"dataset": "esci", "id": N, "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "<answer>{label}</answer>"}
  ]}
where label is 0=Exact, 1=Substitute, 2=Complement, 3=Irrelevant.

Builds the zero-shot prompt from messages[:-1] (system+user), applies the
checkpoint's own chat template, and appends "<answer>" -- the exact fixed
opener create_ESCI.py's assistant turn begins with -- so the model continues
right where the training target would start. Scoring is plain exact match on
the predicted digit vs. the row's true label -- no tolerance/leniency needed
since this is a 4-way classification task, not free-form generation.

Usage:
    python eval_ESCI.py --model Qwen/Qwen2.5-7B-Instruct \\
        --test_file data/processed/esci/test_esci.jsonl --out esci_eval.json
"""

import argparse
import json
import re
from collections import defaultdict

ANSWER_PREFIX = "<answer>"
ANSWER_RE = re.compile(r"([0-3])")

LABEL_NAMES = {0: "Exact", 1: "Substitute", 2: "Complement", 3: "Irrelevant"}


def load_rows(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(tokenizer, row):
    """messages[:-1] is system+user; add_generation_prompt only inserts the
    role marker, so the fixed "<answer>" opener -- baked into the content of
    the (excluded) assistant turn during data creation -- is appended
    manually to reproduce the exact same starting point."""

    prefix_text = tokenizer.apply_chat_template(row["messages"][:-1], add_generation_prompt=True, tokenize=False)
    return prefix_text + ANSWER_PREFIX


def gold_label(row):
    match = ANSWER_RE.search(row["messages"][-1]["content"])
    return int(match.group(1)) if match else None


def extract_answer(completion):
    """First digit 0-3 in the completion. The prompt already ends in
    "<answer>", so a well-behaved model's completion is just "{d}</answer>";
    searching (rather than requiring it as the very first character) also
    tolerates a stray leading space or re-opened "<answer>" tag."""

    match = ANSWER_RE.search(completion)
    return int(match.group(1)) if match else None


def print_samples(prompts, outputs, n):
    """Print the exact rendered prompt + raw generation for the first n
    examples. Mirrors gsm8k_eval.py's / eval_anli.py's helper of the same name."""
    for i in range(min(n, len(prompts))):
        print(f"\n{'=' * 80}\n[sample {i}] PROMPT:\n{prompts[i]}")
        print(f"\n[sample {i}] GENERATION:\n{outputs[i].outputs[0].text}")
    print(f"{'=' * 80}\n" if n > 0 else "", end="")


def main():
    ap = argparse.ArgumentParser(description="Zero-shot ESCI evaluation")
    ap.add_argument("--model", required=True)
    ap.add_argument("--test_file", required=True, help="e.g. data/processed/esci/test_esci.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N instances")
    ap.add_argument("--print_n", type=int, default=0,
                     help="print the exact rendered prompt + raw generation for the first N examples")
    ap.add_argument("--out", default="esci_eval.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    rows = load_rows(args.test_file)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("[WARN] no rows to evaluate")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [build_prompt(tokenizer, row) for row in rows]

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192)
    sp = SamplingParams(
        temperature=0.0,  # greedy: reproducible
        max_tokens=args.max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sp)
    print_samples(prompts, outputs, args.print_n)

    n_correct = n_parsed = 0
    by_label = defaultdict(lambda: {"n": 0, "correct": 0})
    confusion = defaultdict(int)
    records = []

    for row, out in zip(rows, outputs):
        completion = out.outputs[0].text
        pred = extract_answer(completion)
        gold = gold_label(row)
        ok = pred is not None and pred == gold

        gold_name = LABEL_NAMES.get(gold, str(gold))
        pred_name = LABEL_NAMES.get(pred, "INVALID")
        by_label[gold_name]["n"] += 1
        by_label[gold_name]["correct"] += int(ok)
        confusion[(gold_name, pred_name)] += 1
        n_correct += int(ok)
        n_parsed += int(pred is not None)

        records.append({
            "id": row.get("id"),
            "gold": gold,
            "pred": pred,
            "correct": ok,
            "completion": completion,
        })

    n = len(rows)
    print(f"\nmodel: {args.model}   n={n}")
    print(f"accuracy   {n_correct / n:.4f}")
    print(f"parse rate {n_parsed / n:.4f}")

    print(f"\n{'gold':<12}{'n':>6}{'recall':>9}")
    for label_name in LABEL_NAMES.values():
        s = by_label[label_name]
        if s["n"]:
            print(f"{label_name:<12}{s['n']:>6}{s['correct'] / s['n']:>9.4f}")

    print(f"\n{'gold -> pred':<28}{'count':>7}")
    for (g, p), c in sorted(confusion.items(), key=lambda x: -x[1]):
        print(f"{g + ' -> ' + p:<28}{c:>7}")

    with open(args.out, "w") as f:
        json.dump(
            {
                "model": args.model,
                "n": n,
                "accuracy": n_correct / n,
                "parse_rate": n_parsed / n,
                "by_label": {k: dict(v) for k, v in by_label.items()},
                "confusion": {f"{g}->{p}": c for (g, p), c in confusion.items()},
                "records": records,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
