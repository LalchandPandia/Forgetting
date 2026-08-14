"""
Dump raw, unparsed generations from multiple checkpoints on the same prompts.

The point is to look at strings BEFORE any answer extraction, so you can tell
"the model got worse" apart from "the scorer stopped working".

Usage:
    python inspect_generations.py \
        --ckpt base=/path/base \
        --ckpt ifeval=/path/ifeval \
        --ckpt gsm8k=/path/gsm8k \
        --n 20

Writes generations.jsonl (machine-readable) and generations.txt (for reading).
"""

import argparse, json, difflib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def load_prompts(n):
    """IFEval prompts carry formatting constraints; GSM8K carry arithmetic.
    A format-collapsed model fails the first and parses badly on the second."""
    ife = load_dataset("google/IFEval", split="train").select(range(n))
    gsm = load_dataset("gsm8k", "main", split="test").select(range(n))
    prompts = [{"task": "ifeval", "idx": i, "prompt": r["prompt"], "ref": None}
               for i, r in enumerate(ife)]
    prompts += [{"task": "gsm8k", "idx": i, "prompt": r["question"],
                 "ref": r["answer"].split("####")[-1].strip()}
                for i, r in enumerate(gsm)]
    return prompts


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=512):
    outs = []
    for p in prompts:
        msgs = [{"role": "user", "content": p["prompt"]}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = gen[0][enc.input_ids.shape[1]:]
        outs.append({
            **p,
            "rendered_prompt": text,
            "completion": tok.decode(new, skip_special_tokens=False),
            "n_new_tokens": len(new),
            # hit the cap => answer may be truncated off the end, which the
            # extractor then can't find. This alone can manufacture a score drop.
            "hit_cap": len(new) >= max_new_tokens,
        })
    return outs


def config_diff(paths):
    """Tokenizer/EOS drift between checkpoints is a common silent scorer-killer."""
    print("\n=== config drift ===")
    for fname in ["generation_config.json", "tokenizer_config.json"]:
        texts = {}
        for name, p in paths.items():
            f = Path(p) / fname
            texts[name] = f.read_text().splitlines() if f.exists() else ["<missing>"]
        names = list(texts)
        for a, b in zip(names, names[1:]):
            d = list(difflib.unified_diff(texts[a], texts[b], a, b, lineterm="", n=0))
            print(f"\n-- {fname}: {a} vs {b} --")
            print("\n".join(d[2:]) if len(d) > 2 else "  (identical)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="name=/path, repeatable")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    paths = dict(c.split("=", 1) for c in args.ckpt)
    config_diff(paths)

    prompts = load_prompts(args.n)
    results = {}
    for name, path in paths.items():
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto").eval()
        print(f"[{name}] generating {len(prompts)} completions...")
        results[name] = generate(model, tok, prompts, args.max_new_tokens)
        del model
        torch.cuda.empty_cache()

    with open("generations.jsonl", "w") as f:
        for name, outs in results.items():
            for o in outs:
                f.write(json.dumps({"ckpt": name, **o}) + "\n")

    # side-by-side, grouped by prompt -- this is the file you actually read
    names = list(results)
    with open("generations.txt", "w") as f:
        for i, p in enumerate(prompts):
            f.write(f"\n{'='*100}\n[{p['task']} #{p['idx']}]  ref={p['ref']}\n")
            f.write(f"PROMPT: {p['prompt'][:400]}\n")
            for name in names:
                o = results[name][i]
                flag = "  <-- HIT TOKEN CAP" if o["hit_cap"] else ""
                f.write(f"\n--- {name} ({o['n_new_tokens']} tok){flag} ---\n")
                f.write(o["completion"] + "\n")

    # quick automated flags; no substitute for reading, but orients you
    print("\n=== flags ===")
    for name, outs in results.items():
        gsm = [o for o in outs if o["task"] == "gsm8k"]
        ife = [o for o in outs if o["task"] == "ifeval"]
        caps = sum(o["hit_cap"] for o in outs)
        # format collapse: GSM8K-style terminator bleeding into IFEval answers
        bleed = sum("####" in o["completion"] for o in ife)
        empty = sum(o["n_new_tokens"] < 5 for o in outs)
        med_len = sorted(o["n_new_tokens"] for o in ife)[len(ife) // 2]
        print(f"{name:>10}: hit_cap={caps:>3}  empty={empty:>3}  "
              f"'####'-in-ifeval={bleed:>3}  median_ifeval_len={med_len}")
    print("\nNow open generations.txt and grade the 20 GSM8K answers by hand.")
    print("Manual accuracy near base + harness reporting -8  =>  parser failure.")
