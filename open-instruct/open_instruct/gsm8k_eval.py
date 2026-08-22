"""
Evaluate a chat model on GSM8K via vLLM, reproducing two different eval
methodologies from two different repos (plus a third, local style matching
this project's own SFT prompt format), all reading questions from a local
jsonl file.

Input jsonl format (one row per question) is either the standard GSM8K schema:
    {
        "question": "Natalia sold clips to 48 of her friends in April...",
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May. ... #### 72"
    }
or open-instruct's chat schema, as produced by data/processed/*.jsonl:
    {
        "messages": [
            {"role": "user", "content": "Question: Natalia sold...\nAnswer: "},
            {"role": "assistant", "content": "Natalia sold 48/2 = <<48/2=24>>24 clips in May. ... #### 72"}
        ]
    }
(the Question:/Answer: wrapper on the user turn is stripped back out before use --
see get_question()/get_gold_answer()).

Styles
------
--style retaining_by_doing
    Reproduces retaining-by-doing's GSM8KDataset: https://github.com/.../retaining-by-doing/blob/main/core/data.py
    NOTE: this class exists in core/data.py but is *not* actually wired into
    DATASET_NAME_TO_MODULE in core/evaluation/run.py or scripts/eval.sh's
    dataset_name_shorts list -- it's present in the repo but effectively dead
    code for eval purposes. We reproduce it as written anyway.
    Zero-shot, free-form generation, answer extracted via a cascading regex
    (currency/boxed/inline-math/last-number fallbacks), then exact string
    match against the gold answer extracted the same way.

--style train_format
    Zero-shot, same free-form generation + cascading-regex scoring as
    retaining_by_doing, but the user turn is passed through VERBATIM as
    "Question: {q}\nAnswer: " -- the exact string this project's SFT data
    uses (see scripts/data/gsm8k_official.py) -- instead of
    retaining_by_doing's bare, unwrapped question text. Isolates whether a
    train/eval prompt-format mismatch (retaining_by_doing's prompt does not
    end in the "\nAnswer: " cue the model was fine-tuned to expect right
    before generating) explains a gap between retaining_by_doing and tulu,
    as opposed to an actual capability/reasoning difference.

--style tulu
    Reproduces oe_eval's `gsm8k::tulu` task:
    https://github.com/.../olmes/oe_eval/configs/tasks.py
    https://github.com/.../olmes/oe_eval/tasks/oe_eval_tasks/gsm8k.py (GSM8K)
    8-shot CoT (fewshot_as_multiturn), greedy decoding, stopped at
    "Question:"/eos-like strings, answer extracted as the last number in the
    response, exact match (case-insensitive, commas/currency/trailing-period
    stripped) against the last number in the gold "#### N" answer.

    This is a faithful-in-spirit but simplified reimplementation -- it does
    not depend on the oe_eval framework itself. The 8 few-shot examples are
    the standard GSM8K CoT shots (fewshot_source: "STD:GSM8k" in oe_eval);
    since that fixed set isn't vendored here, use --fewshot_jsonl to supply
    them explicitly, or omit it to borrow the first --num_shots rows from
    --input_jsonl itself (with a warning -- this will not match oe_eval's
    actual fixed shots).

Usage
-----
python gsm8k_eval.py --input_jsonl data/eval/gsm8k.jsonl --style retaining_by_doing \
    --model_name_or_path /path/to/model --output_jsonl preds.jsonl

python gsm8k_eval.py --input_jsonl data/eval/gsm8k.jsonl --style tulu \
    --model_name_or_path /path/to/model --output_jsonl preds.jsonl \
    --fewshot_jsonl data/gsm8k_std_8shot.jsonl
"""
import re
import json
import argparse

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


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


def get_question(row):
    """Rows are either the flat {question, answer} GSM8K schema, or open-instruct's
    {messages: [{role: user, content: "Question: ...\\nAnswer: "}, ...]} schema. For the
    latter, strip the Question:/Answer: wrapper so callers get the bare question text
    (retaining_by_doing feeds it straight into a chat template; tulu's make_query()
    re-adds the same wrapper itself)."""
    if "messages" in row:
        content = row["messages"][0]["content"]
        content = re.sub(r"^Question:\s*", "", content)
        content = re.sub(r"\s*Answer:\s*$", "", content)
        return content
    return row["question"]


def get_gold_answer(row):
    if "messages" in row:
        return row["messages"][1]["content"]
    return row["answer"]


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
# Style 1: retaining-by-doing (zero-shot generation + cascading regex)
# ---------------------------------------------------------------------------

def parse_retaining_by_doing_answer(output_text):
    parsed = re.findall(r"\$?[\d,]+\.?\d*", output_text)
    if len(parsed) >= 1:
        parsed_output_text = parsed[-1]
        parsed_output_text = re.sub(r"[,$]", "", parsed_output_text)
        parsed_output_text = re.sub(r"\.$", "", parsed_output_text)
        return parsed_output_text

    matches = re.findall(r"\\boxed{((?:[^{}]|{[^{}]*})*)}", output_text)
    if matches:
        return matches[-1].strip()

    matches = re.findall(r"\$([^$]+)\$", output_text)
    if matches:
        return matches[-1].strip()

    matches = re.findall(r"(?:^|[^\d])(\d+(?:\.\d+)?|\.\d+)(?:[^\d]|$)", output_text)
    if matches:
        return matches[-1].strip()

    return None


def run_retaining_by_doing(rows, llm, tokenizer, batch_size, max_new_tokens):
    predictions = []
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": get_question(row)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for row, output in zip(batch, outputs):
            output_text = output.outputs[0].text
            pred = parse_retaining_by_doing_answer(output_text)
            gold = parse_retaining_by_doing_answer(get_gold_answer(row))
            predictions.append(dict(
                question=get_question(row),
                output_text=output_text,
                pred=pred,
                gold=gold,
                correct=pred is not None and pred == gold,
            ))
    return predictions


# ---------------------------------------------------------------------------
# Style 1b: train_format -- zero-shot, but the user turn is passed through
# VERBATIM as "Question: {q}\nAnswer: " (the exact string open-instruct's SFT
# data uses -- see scripts/data/gsm8k_official.py), instead of get_question()'s
# stripped/bare question text. retaining_by_doing's prompt does NOT match what
# the model was fine-tuned to see right before generating (no "Question:"/
# "Answer:" framing, no trailing space); this style isolates that one variable
# to test whether the format mismatch itself explains an accuracy gap, using
# the same answer extraction/scoring as retaining_by_doing so it's otherwise
# an apples-to-apples comparison.
# ---------------------------------------------------------------------------

def run_train_format(rows, llm, tokenizer, batch_size, max_new_tokens):
    predictions = []
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [row["messages"][0]] if "messages" in row
                else [{"role": "user", "content": f"Question: {row['question']}\nAnswer: "}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for row, output in zip(batch, outputs):
            output_text = output.outputs[0].text
            pred = parse_retaining_by_doing_answer(output_text)
            gold = parse_retaining_by_doing_answer(get_gold_answer(row))
            predictions.append(dict(
                question=get_question(row),
                output_text=output_text,
                pred=pred,
                gold=gold,
                correct=pred is not None and pred == gold,
            ))
    return predictions


# ---------------------------------------------------------------------------
# Style 2: oe_eval's gsm8k::tulu (8-shot CoT, fewshot_as_multiturn, exact match)
# ---------------------------------------------------------------------------

def extract_last_number(text):
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return numbers[-1] if numbers else text.strip()


def normalize_cot_answer(answer, short_answer):
    """Port of oe_eval's GSM8K.normalize_answer_str: turns the raw GSM8K
    '<<calc>> ... #### N' rationale into a natural-sounding CoT ending in
    'So the answer is N.'"""
    answer = re.sub(r"<<.*?>>", "", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = re.split(r"####", answer)[0].strip()
    if answer:
        answer = answer[0].capitalize() + answer[1:]
    if not answer.endswith("."):
        answer += "."
    return f"{answer} So the answer is {short_answer}."


def make_query(question):
    return f"Question: {question}\nAnswer:"


def build_fewshot_shots(fewshot_jsonl, input_rows, num_shots):
    if fewshot_jsonl is not None:
        shot_rows = load_jsonl(fewshot_jsonl)[:num_shots]
        return shot_rows, set()

    print(
        f"[warn] --fewshot_jsonl not given for --style tulu: borrowing the first {num_shots} "
        "rows from --input_jsonl as few-shot context (oe_eval normally uses a fixed standard "
        "GSM8K 8-shot CoT prompt for this, which is NOT what this script is doing -- results "
        "will differ from a real oe_eval run)."
    )
    shots = input_rows[:num_shots]
    return shots, set(range(len(shots)))


def normalize_for_exact_match(text):
    for pattern in [",", r"\$", r"(?s).*#### ", r"\.$"]:
        text = re.sub(pattern, "", text)
    return text.strip().lower()


def run_tulu(rows, llm, tokenizer, num_shots, fewshot_jsonl, max_new_tokens):
    shots, excluded_indices = build_fewshot_shots(fewshot_jsonl, rows, num_shots)

    messages = []
    for shot in shots:
        shot_answer = get_gold_answer(shot)
        short_answer = shot_answer.split("####")[-1].strip()
        messages.append({"role": "user", "content": make_query(get_question(shot))})
        messages.append({"role": "assistant", "content": normalize_cot_answer(shot_answer, short_answer)})

    eval_rows = [row for idx, row in enumerate(rows) if idx not in excluded_indices]
    prompts = []
    for row in eval_rows:
        row_messages = messages + [{"role": "user", "content": make_query(get_question(row))}]
        prompts.append(tokenizer.apply_chat_template(row_messages, tokenize=False, add_generation_prompt=True))

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        stop=["Question:", "</s>", "<|im_end|>"],
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    predictions = []
    for row, output in zip(eval_rows, outputs):
        output_text = output.outputs[0].text
        pred_raw = extract_last_number(output_text)
        gold_raw = extract_last_number(get_gold_answer(row).split("####")[-1].strip())
        pred = normalize_for_exact_match(pred_raw)
        gold = normalize_for_exact_match(gold_raw)
        predictions.append(dict(
            question=get_question(row),
            output_text=output_text,
            pred=pred,
            gold=gold,
            correct=pred == gold,
        ))
    return predictions


# ---------------------------------------------------------------------------

def summarize(predictions):
    return {"accuracy": sum(p["correct"] for p in predictions) / len(predictions)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True, help="GSM8K questions as jsonl.")
    parser.add_argument("--style", required=True, choices=["retaining_by_doing", "train_format", "tulu"])
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Defaults to all visible GPUs.")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Used by --style retaining_by_doing/train_format only."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=None,
        help="Defaults to 4096 for --style retaining_by_doing/train_format, 512 for --style tulu "
             "(matching each repo's own eval config).",
    )
    parser.add_argument("--num_shots", type=int, default=8, help="Used by --style tulu only.")
    parser.add_argument(
        "--fewshot_jsonl", default=None,
        help="Used by --style tulu only. Jsonl of {question,answer} for the 8-shot CoT context. "
             "If omitted, shots are borrowed from --input_jsonl itself.",
    )
    args = parser.parse_args()

    max_new_tokens = args.max_new_tokens or (512 if args.style == "tulu" else 4096)

    rows = load_jsonl(args.input_jsonl)
    llm, tokenizer = load_vllm(args.model_name_or_path, args.tensor_parallel_size)

    if args.style == "retaining_by_doing":
        predictions = run_retaining_by_doing(rows, llm, tokenizer, args.batch_size, max_new_tokens)
    elif args.style == "train_format":
        predictions = run_train_format(rows, llm, tokenizer, args.batch_size, max_new_tokens)
    else:
        predictions = run_tulu(rows, llm, tokenizer, args.num_shots, args.fewshot_jsonl, max_new_tokens)

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
