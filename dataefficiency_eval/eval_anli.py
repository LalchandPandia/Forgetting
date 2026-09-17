"""
ANLI evaluation by exact match on the verbalized label.

Reads the same jsonl format used for training:
  {"dataset": "anli", "id": 1, "messages": [{"role":"user", ...},
                                            {"role":"assistant","content":"Unclear"}]}

The assistant turn is the gold label and is never shown to the model.

Usage:
  python eval_anli.py --model allenai/OLMo-2-1124-7B --data test_anli.jsonl --raw
  python eval_anli.py --model Qwen/Qwen2.5-1.5B-Instruct --data test_anli.jsonl --out r1.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict

# create_anli.py's prompt never tells the model to be terse, and strict
# exact-match scoring (the default here) has no tolerance for preamble --
# without this, a correctly-reasoned "Based on the premise, ... so Unclear."
# scores as invalid, not just wrong. Prepended as a system turn (chat-template
# mode) or a leading line (--raw mode).
ANSWER_INSTRUCTION = "Answer with only one word: Yes, No, or Unclear."

# Canonical label set. Keys are what we compare against after normalization.
LABELS = ["yes", "no", "unclear"]

# Surface forms that map onto a canonical label. Keep this deliberately tight:
# anything broader starts rescuing predictions the model never really committed
# to, which inflates accuracy relative to a strict exact-match protocol.
ALIASES = {
    "yes": "yes",
    "no": "no",
    "unclear": "unclear",
    "a": "yes",
    "b": "no",
    "c": "unclear",
    "a. yes": "yes",
    "b. no": "no",
    "c. unclear": "unclear",
    "entailment": "yes",
    "contradiction": "no",
    "neutral": "unclear",
    "true": "yes",
    "false": "no",
}


def normalize(text):
    """Strip whitespace/punctuation/casing. Returns '' if nothing is left."""
    if text is None:
        return ""
    t = text.strip().lower()
    t = t.split("\n")[0].strip()          # first line only
    t = re.sub(r"^[\s\"'`*]+|[\s\"'`*.,!;:]+$", "", t)
    return t.strip()


def to_label(text, strict):
    """Map a completion to a canonical label, or None if it does not match one.

    strict=True  -> the whole normalized string must be a known surface form.
    strict=False -> also accept a label appearing as the first token, which
                    catches 'Yes, the premise states...' style continuations.
    """
    t = normalize(text)
    if not t:
        return None
    if t in ALIASES:
        return ALIASES[t]
    if not strict:
        first = t.split()[0].rstrip(".,:;")
        if first in ALIASES:
            return ALIASES[first]
    return None


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            msgs = ex["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            gold = next(m["content"] for m in msgs if m["role"] == "assistant")
            gold_norm = to_label(gold, strict=True)
            if gold_norm is None:
                raise ValueError(f"id={ex.get('id')}: gold {gold!r} is not a known label")
            rows.append({"id": ex.get("id"), "prompt": user, "gold": gold_norm})
    return rows


def print_samples(prompts, outputs, n):
    """Print the exact rendered prompt + raw generation for the first n
    examples -- what the model actually saw and produced, not a
    reconstruction. Mirrors gsm8k_eval.py's helper of the same name."""
    for i in range(min(n, len(prompts))):
        print(f"\n{'=' * 80}\n[sample {i}] PROMPT:\n{prompts[i]}")
        print(f"\n[sample {i}] GENERATION:\n{outputs[i].outputs[0].text}")
    print(f"{'=' * 80}\n" if n > 0 else "", end="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="test_anli.jsonl")
    ap.add_argument("--raw", action="store_true",
                    help="feed the prompt as a plain completion (base models). "
                         "Default applies the tokenizer's chat template.")
    ap.add_argument("--lenient", action="store_true",
                    help="accept a label as the first token of a longer completion")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--print_n", type=int, default=0,
                    help="print the exact rendered prompt + raw generation for the first N examples")
    ap.add_argument("--out", default="anli_eval.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.raw:
        prompts = [f"{ANSWER_INSTRUCTION}\n\n{r['prompt']}" for r in rows]
    else:
        prompts = [
            tok.apply_chat_template(
                [
                    {"role": "system", "content": ANSWER_INSTRUCTION},
                    {"role": "user", "content": r["prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in rows
        ]

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192)
    sp = SamplingParams(
        temperature=0.0,                  # greedy: the eval must be reproducible
        max_tokens=args.max_new_tokens,
        stop=["\n", "Question:", "Premise:"],
    )
    outs = llm.generate(prompts, sp)
    print_samples(prompts, outs, args.print_n)

    strict = not args.lenient
    n_correct = n_valid = 0
    per_label = defaultdict(lambda: {"n": 0, "correct": 0})
    confusion = Counter()
    records = []

    for r, o in zip(rows, outs):
        completion = o.outputs[0].text
        pred = to_label(completion, strict=strict)
        ok = pred is not None and pred == r["gold"]

        n_correct += int(ok)
        n_valid += int(pred is not None)
        per_label[r["gold"]]["n"] += 1
        per_label[r["gold"]]["correct"] += int(ok)
        confusion[(r["gold"], pred or "INVALID")] += 1

        records.append({
            "id": r["id"], "gold": r["gold"], "pred": pred,
            "correct": ok, "raw": completion,
        })

    n = len(rows)
    print(f"\nmodel      {args.model}")
    print(f"data       {args.data}   n={n}")
    print(f"accuracy   {n_correct / n:.4f}")
    print(f"valid rate {n_valid / n:.4f}   (predictions that parsed to a label)")
    if n_valid:
        print(f"acc|valid  {n_correct / n_valid:.4f}")

    print(f"\n{'gold':<10}{'n':>6}{'recall':>9}")
    for lab in LABELS:
        s = per_label[lab]
        if s["n"]:
            print(f"{lab:<10}{s['n']:>6}{s['correct'] / s['n']:>9.4f}")

    # Prediction distribution: a model that collapses onto one label shows up
    # here long before it shows up in the headline accuracy.
    dist = Counter(r["pred"] or "INVALID" for r in records)
    print("\npredicted:", dict(dist))

    print(f"\n{'gold -> pred':<24}{'count':>7}")
    for (g, p), c in sorted(confusion.items(), key=lambda x: -x[1]):
        print(f"{g + ' -> ' + p:<24}{c:>7}")

    with open(args.out, "w") as f:
        json.dump({
            "model": args.model,
            "data": args.data,
            "n": n,
            "accuracy": n_correct / n,
            "valid_rate": n_valid / n,
            "per_label": {k: dict(v) for k, v in per_label.items()},
            "prediction_distribution": dict(dist),
            "confusion": {f"{g}->{p}": c for (g, p), c in confusion.items()},
            "records": records,
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()