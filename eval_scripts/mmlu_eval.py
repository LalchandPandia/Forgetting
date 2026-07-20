"""
Evaluate a chat model on MMLU via vLLM, reproducing two different eval
methodologies from two different repos, both reading questions from a local
jsonl file. Both repos actually use vLLM for local-model eval (retaining-by-doing's
scripts/eval.sh sets model_module_path=core.vllm_utils.vLLMCausalLM; oe_eval's
default local backend is also vLLM), so this script does the same.

Input jsonl format (one row per question), matching the fields used by both
source implementations:
    {
        "question": "...",
        "choices": ["choice A text", "choice B text", "choice C text", "choice D text"],
        "answer": 2,            # int index into `choices` (0=A, 1=B, ...)
        "subject": "abstract_algebra"   # only needed for --style tulu_mc
    }

Styles
------
--style retaining_by_doing
    Reproduces retaining-by-doing's MMLUDataset + core/evaluation/run.py:
    https://github.com/.../retaining-by-doing/blob/main/core/data.py (MMLUDataset)
    Zero-shot, free-form generation ("Reason about it and answer with
    'The answer is: <option>'"), answer extracted via regex, exact match.

--style tulu_mc
    Reproduces oe_eval's `mmlu_<subject>:mc::tulu` task:
    https://github.com/.../olmes/oe_eval/configs/tasks.py
    https://github.com/.../olmes/oe_eval/tasks/oe_eval_tasks/mmlu.py (GenericMMLU_MC)
    5-shot (fewshot_as_multiturn), ranked classification: for each answer
    choice, scores the model's log-likelihood of that choice as a
    continuation and picks the argmax. No free-form generation -- scored via
    vLLM's `prompt_logprobs` (echo) feature, the standard way to get
    loglikelihoods for a fixed continuation out of vLLM.

    This is a faithful-in-spirit but simplified reimplementation of oe_eval's
    fewshot_context / doc_to_text / loglikelihood-request machinery -- it does
    not depend on the oe_eval framework itself, only on a local jsonl of
    questions (and, for fewshot, either a --fewshot_jsonl file or examples
    borrowed from the input file itself).

Usage
-----
python mmlu_eval.py --input_jsonl data/eval/mmlu.jsonl --style retaining_by_doing \
    --model_name_or_path /path/to/model --output_jsonl preds.jsonl

python mmlu_eval.py --input_jsonl data/eval/mmlu.jsonl --style tulu_mc \
    --model_name_or_path /path/to/model --output_jsonl preds.jsonl \
    --fewshot_jsonl data/mmlu_dev_fewshot.jsonl
"""
import re
import json
import argparse
from collections import defaultdict

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

LABELS = list("ABCDE")


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_vllm(model_name_or_path, tensor_parallel_size=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.chat_template is None:
        raise ValueError(
            f"{model_name_or_path} has no chat template; both eval styles require an "
            "instruct/chat model."
        )
    if tensor_parallel_size is None:
        tensor_parallel_size = max(torch.cuda.device_count(), 1)
    llm = LLM(model=model_name_or_path, tensor_parallel_size=tensor_parallel_size)
    return llm, tokenizer


# ---------------------------------------------------------------------------
# Style 1: retaining-by-doing (zero-shot generation + regex extraction)
# ---------------------------------------------------------------------------

def build_retaining_by_doing_prompt(question, choices):
    choices_text = "\n".join(f"{c}. {s}" for c, s in zip(LABELS, choices))
    return (
        f"{question}\n\nAnswer options:\n{choices_text}\n\n"
        'Reason about it and answer with "The answer is: <option>"'
    )


def parse_retaining_by_doing_answer(output_text):
    output_text = output_text.replace("*", "")
    match = re.search(r"\bThe(.*)answer is(?: option)?:?\s*(\w+)", output_text, re.IGNORECASE)
    if match:
        return match.group(2).upper().strip()
    return None


def run_retaining_by_doing(rows, llm, tokenizer, batch_size, max_new_tokens):
    predictions = []
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": build_retaining_by_doing_prompt(row["question"], row["choices"])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for row, output in zip(batch, outputs):
            output_text = output.outputs[0].text
            pred_label = parse_retaining_by_doing_answer(output_text)
            gold_label = LABELS[row["answer"]]
            predictions.append(dict(
                subject=row.get("subject"),
                question=row["question"],
                output_text=output_text,
                pred_label=pred_label,
                gold_label=gold_label,
                correct=pred_label == gold_label,
            ))
    return predictions


# ---------------------------------------------------------------------------
# Style 2: oe_eval's mmlu:mc::tulu (5-shot, fewshot_as_multiturn, ranked choice)
# ---------------------------------------------------------------------------

def make_mc_turn(question, choices, answer_idx=None):
    labels = LABELS[:len(choices)]
    lines = "\n".join(f" {l}. {c}" for l, c in zip(labels, choices))
    user_content = f"{question}\n{lines}\nAnswer:"
    assistant_content = f" {labels[answer_idx]}" if answer_idx is not None else None
    return user_content, assistant_content


def build_fewshot_map(input_rows, fewshot_jsonl, num_shots):
    if fewshot_jsonl is not None:
        shot_rows = load_jsonl(fewshot_jsonl)
        by_subject = defaultdict(list)
        for row in shot_rows:
            by_subject[row["subject"]].append(row)
        return {subj: rows[:num_shots] for subj, rows in by_subject.items()}, set()

    print(
        "[warn] --fewshot_jsonl not given for --style tulu_mc: borrowing the first "
        f"{num_shots} example(s) per subject from --input_jsonl as few-shot context "
        "(oe_eval normally uses MMLU's fixed 'dev' split for this, which is NOT what "
        "this script is doing -- results will differ from a real oe_eval run)."
    )
    by_subject = defaultdict(list)
    for idx, row in enumerate(input_rows):
        by_subject[row["subject"]].append((idx, row))

    fewshot_map = {}
    excluded_indices = set()
    for subj, indexed_rows in by_subject.items():
        shots = indexed_rows[:num_shots]
        fewshot_map[subj] = [row for _, row in shots]
        excluded_indices.update(idx for idx, _ in shots)
    return fewshot_map, excluded_indices


def score_choices_vllm(llm, tokenizer, prefix_choice_pairs):
    """
    prefix_choice_pairs: list of (prefix_text, choice_str) pairs.
    Returns the summed log-probability of choice_str's tokens under the model,
    via vLLM's prompt_logprobs ("echo") feature -- the standard way to score a
    fixed continuation without actually sampling it. As with any BPE-based
    loglikelihood scoring (same caveat lm-eval-harness has), the prefix/choice
    token boundary is determined by tokenizing the prefix alone, which can
    occasionally differ by a token or two from how the concatenated string
    would be tokenized.
    """
    full_texts = []
    n_prefix_tokens_list = []
    for prefix_text, choice_str in prefix_choice_pairs:
        n_prefix_tokens = len(tokenizer(prefix_text, add_special_tokens=False).input_ids)
        full_texts.append(prefix_text + choice_str)
        n_prefix_tokens_list.append(n_prefix_tokens)

    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    outputs = llm.generate(full_texts, sampling_params, use_tqdm=False)

    scores = []
    for output, n_prefix_tokens in zip(outputs, n_prefix_tokens_list):
        prompt_token_ids = output.prompt_token_ids
        prompt_logprobs = output.prompt_logprobs
        total = 0.0
        for pos in range(n_prefix_tokens, len(prompt_token_ids)):
            token_id = prompt_token_ids[pos]
            total += prompt_logprobs[pos][token_id].logprob
        scores.append(total)
    return scores


def run_tulu_mc(rows, llm, tokenizer, num_shots, fewshot_jsonl):
    fewshot_map, excluded_indices = build_fewshot_map(rows, fewshot_jsonl, num_shots)

    eval_rows = []
    prefix_texts = []
    for idx, row in enumerate(rows):
        if idx in excluded_indices:
            continue
        subject = row["subject"]
        description = (
            f"The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.\n\n"
        )

        messages = []
        shots = fewshot_map.get(subject, [])
        for i, shot in enumerate(shots):
            user_content, assistant_content = make_mc_turn(shot["question"], shot["choices"], shot["answer"])
            if i == 0:
                user_content = description + user_content
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})

        eval_user_content, _ = make_mc_turn(row["question"], row["choices"])
        if not shots:
            eval_user_content = description + eval_user_content
        messages.append({"role": "user", "content": eval_user_content})

        prefix_texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        eval_rows.append(row)

    pairs = []
    choice_counts = []
    for row, prefix_text in zip(eval_rows, prefix_texts):
        labels = LABELS[:len(row["choices"])]
        choice_counts.append(len(labels))
        for label in labels:
            pairs.append((prefix_text, f" {label}"))

    scores_flat = score_choices_vllm(llm, tokenizer, pairs)

    predictions = []
    ptr = 0
    for row, n_choices in zip(eval_rows, choice_counts):
        labels = LABELS[:n_choices]
        row_scores = scores_flat[ptr:ptr + n_choices]
        ptr += n_choices
        pred_idx = max(range(len(row_scores)), key=lambda i: row_scores[i])
        predictions.append(dict(
            subject=row["subject"],
            question=row["question"],
            choice_logprobs=dict(zip(labels, row_scores)),
            pred_label=labels[pred_idx],
            gold_label=labels[row["answer"]],
            correct=pred_idx == row["answer"],
        ))
    return predictions


# ---------------------------------------------------------------------------

def summarize(predictions):
    by_subject = defaultdict(list)
    for p in predictions:
        by_subject[p.get("subject") or "all"].append(p["correct"])

    metrics = {"accuracy": sum(p["correct"] for p in predictions) / len(predictions)}
    metrics["per_subject"] = {
        subj: sum(corrects) / len(corrects) for subj, corrects in by_subject.items()
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True, help="MMLU questions as jsonl.")
    parser.add_argument("--style", required=True, choices=["retaining_by_doing", "tulu_mc"])
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Defaults to all visible GPUs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Used by --style retaining_by_doing only.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Used by --style retaining_by_doing only.")
    parser.add_argument("--num_shots", type=int, default=5, help="Used by --style tulu_mc only.")
    parser.add_argument(
        "--fewshot_jsonl", default=None,
        help="Used by --style tulu_mc only. Jsonl of {question,choices,answer,subject} to use as "
             "5-shot context per subject. If omitted, shots are borrowed from --input_jsonl itself.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    llm, tokenizer = load_vllm(args.model_name_or_path, args.tensor_parallel_size)

    if args.style == "retaining_by_doing":
        predictions = run_retaining_by_doing(rows, llm, tokenizer, args.batch_size, args.max_new_tokens)
    else:
        predictions = run_tulu_mc(rows, llm, tokenizer, args.num_shots, args.fewshot_jsonl)

    write_jsonl(args.output_jsonl, predictions)
    metrics = summarize(predictions)
    metrics_path = args.output_jsonl.rsplit(".jsonl", 1)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"style={args.style}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"predictions written to {args.output_jsonl}")
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
