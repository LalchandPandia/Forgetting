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
--style base
    Reproduces oe_eval's `gsm8k::olmes` task -- the DEFAULT (non-chat)
    TASK_CONFIG_DEFAULTS in oe_eval's gsm8k.py, for evaluating a raw BASE
    model that has no chat template. Flat 8-shot CoT via plain string
    concatenation ("Question: ...\nAnswer: ..." x8, no roles, no chat
    template), stopped at "Question:"/"\n\n"/eos-like strings, answer
    extracted as the last number in the response, exact match against the
    last number in the gold "#### N" answer. Uses the real fixed "STD:GSM8k"
    8-shot examples (vendored from olmes/eval_gsm8k_base.py), not borrowed
    rows. This is the only style that does NOT require an instruct/chat
    model -- --style retaining_by_doing/train_format/tulu all do.

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

--style train_format_fewshot
    The 8-shot sibling of train_format: identical machinery to --style tulu
    (real vendored STD:GSM8k shots by default, fewshot_as_multiturn, same
    stop sequences/extraction/scoring) but with "Answer:" kept on the USER
    side ("Question: {q}\nAnswer: ") at every turn, shots included, instead
    of tulu's assistant-side assistant_prefix convention. Isolates whether a
    gap between train_format and tulu comes from shot count (0 vs 8) or from
    "Answer:" placement, by holding everything else about tulu's methodology
    fixed and only swapping that one convention for this project's own.

--style tulu
    Reproduces oe_eval's `gsm8k::tulu` task -- the chat_overrides layered on
    top of the same GSM8K task `--style base` reproduces the default of:
    https://github.com/.../olmes/oe_eval/tasks/oe_eval_tasks/gsm8k.py (GSM8K)
    8-shot CoT (fewshot_as_multiturn), greedy decoding, stopped at
    "Question:"/eos-like strings, answer extracted as the last number in the
    response, exact match (case-insensitive, commas/currency/trailing-period
    stripped) against the last number in the gold "#### N" answer.

    Uses the real fixed "STD:GSM8k" 8-shot examples by default (the same
    FEWSHOT_EXAMPLES --style base uses), each shot's user turn as bare
    "Question: {q}" and assistant turn as "Answer: {a}" (assistant_prefix),
    with "Answer:" also appended after the chat template's own generation
    prompt for the row actually being scored -- matching oe_eval's
    convert_chat_instance (context = apply_chat_template(...) + assistant_prefix).
    Pass --fewshot_jsonl to substitute custom shots in the raw GSM8K schema
    instead (those get normalized from "<<calc>> ... #### N" into the same
    "So the answer is N." style first).

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


def print_samples(prompts, outputs, n):
    """Print the exact rendered prompt + raw generation for the first n
    examples -- what the model actually saw and produced, not a
    reconstruction. Call once, on the first batch only."""
    for i in range(min(n, len(prompts))):
        print(f"\n{'=' * 80}\n[sample {i}] PROMPT:\n{prompts[i]}")
        print(f"\n[sample {i}] GENERATION:\n{outputs[i].outputs[0].text}")
    print(f"{'=' * 80}\n" if n > 0 else "", end="")


def get_question(row):
    """Rows are either the flat {question, answer} GSM8K schema, or open-instruct's
    {messages: [{role: user, content: "Question: ...\\nAnswer: "}, ...]} schema. For the
    latter, strip the Question:/Answer: wrapper so callers get the bare question text
    (retaining_by_doing feeds it straight into a chat template; tulu/base re-add
    their own "Question: ..." framing on top of the bare text)."""
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


def load_vllm(model_name_or_path, tensor_parallel_size=None, require_chat_template=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if require_chat_template and tokenizer.chat_template is None:
        raise ValueError(
            f"{model_name_or_path} has no chat template; --style retaining_by_doing/train_format/tulu "
            "all require an instruct/chat model (use --style base for a raw base model)."
        )
    if tensor_parallel_size is None:
        tensor_parallel_size = max(torch.cuda.device_count(), 1)
    llm = LLM(model=model_name_or_path, tensor_parallel_size=tensor_parallel_size)
    return llm, tokenizer


# ---------------------------------------------------------------------------
# Style 0: base -- oe_eval's gsm8k::olmes methodology for BASE (non-chat)
# models: flat 8-shot CoT via plain string concatenation, no chat template at
# all. This is the DEFAULT TASK_CONFIG_DEFAULTS in oe_eval's gsm8k.py; the
# "tulu" style below is what that file's chat_overrides layer on top of this
# once a chat-templated model is being evaluated. Uses the real fixed
# "STD:GSM8k" 8-shot examples (vendored from oe_eval's fewshot_sources.py via
# olmes/eval_gsm8k_base.py), not borrowed rows from --input_jsonl.
# ---------------------------------------------------------------------------

FEWSHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.",
    },
]

BASE_STOP_SEQUENCES = ["Question:", "</s>", "<|im_end|>", "\n\n"]


def build_base_fewshot_prefix():
    parts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in FEWSHOT_EXAMPLES]
    return "\n\n".join(parts) + "\n\n"


def run_base(rows, llm, tokenizer, batch_size, max_new_tokens, print_n=0):
    predictions = []
    fewshot_prefix = build_base_fewshot_prefix()
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, stop=BASE_STOP_SEQUENCES)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [fewshot_prefix + f"Question: {get_question(row)}\nAnswer:" for row in batch]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        if start == 0:
            print_samples(prompts, outputs, print_n)
        for row, output in zip(batch, outputs):
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


def run_retaining_by_doing(rows, llm, tokenizer, batch_size, max_new_tokens, print_n=0):
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
        if start == 0:
            print_samples(prompts, outputs, print_n)
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

def run_train_format(rows, llm, tokenizer, batch_size, max_new_tokens, print_n=0):
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
        if start == 0:
            print_samples(prompts, outputs, print_n)
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


def build_fewshot_messages(fewshot_jsonl, num_shots, answer_side="assistant_prefix"):
    """Builds the fewshot_as_multiturn message list, in one of two conventions
    for where "Answer:" goes:

      answer_side="assistant_prefix" (oe_eval's real gsm8k::tulu convention --
        see olmes/eval_gsm8k_instruct.py's build_messages): user turn is bare
        "Question: {q}"; "Answer:" is an ASSISTANT-side prefix instead
        ("Answer: {a}"), also applied to the final generation prompt by
        run_tulu (appended after apply_chat_template, matching
        oe_eval's convert_chat_instance).

      answer_side="user_suffix" (this project's own SFT format -- see
        get_question()'s docstring / scripts/data/gsm8k_official.py): user
        turn ends in "Question: {q}\\nAnswer: "; the assistant turn/generation
        starts fresh right after it, with no "Answer:" prefix at all -- the
        same shape as --style train_format, just with 8 shots instead of zero.

    Defaults to the real vendored STD:GSM8k shots (FEWSHOT_EXAMPLES, already
    pre-written ending in "So the answer is N." -- used verbatim, no
    normalization). Pass --fewshot_jsonl for custom shots in the raw GSM8K
    schema instead; those get run through normalize_cot_answer first since
    they carry the raw '<<calc>> ... #### N' format.
    """
    assert answer_side in ("assistant_prefix", "user_suffix")

    if fewshot_jsonl is not None:
        shot_rows = load_jsonl(fewshot_jsonl)[:num_shots]
        shots = []
        for shot in shot_rows:
            shot_answer = get_gold_answer(shot)
            short_answer = shot_answer.split("####")[-1].strip()
            shots.append((get_question(shot), normalize_cot_answer(shot_answer, short_answer)))
    else:
        shots = [(ex["question"], ex["answer"]) for ex in FEWSHOT_EXAMPLES[:num_shots]]

    messages = []
    for question, answer in shots:
        if answer_side == "assistant_prefix":
            messages.append({"role": "user", "content": f"Question: {question}"})
            messages.append({"role": "assistant", "content": f"Answer: {answer}"})
        else:
            messages.append({"role": "user", "content": f"Question: {question}\nAnswer: "})
            messages.append({"role": "assistant", "content": answer})
    return messages


def normalize_for_exact_match(text):
    for pattern in [",", r"\$", r"(?s).*#### ", r"\.$"]:
        text = re.sub(pattern, "", text)
    return text.strip().lower()


def run_fewshot_chat(rows, llm, tokenizer, num_shots, fewshot_jsonl, max_new_tokens, answer_side, print_n=0):
    """Shared by --style tulu (answer_side="assistant_prefix") and --style
    train_format_fewshot (answer_side="user_suffix") -- identical 8-shot
    machinery, differing only in where "Answer:" sits relative to the turn
    boundary. See build_fewshot_messages() for what that changes."""
    shot_messages = build_fewshot_messages(fewshot_jsonl, num_shots, answer_side)

    prompts = []
    for row in rows:
        question = get_question(row)
        if answer_side == "assistant_prefix":
            row_messages = shot_messages + [{"role": "user", "content": f"Question: {question}"}]
            prompt = tokenizer.apply_chat_template(row_messages, tokenize=False, add_generation_prompt=True)
            # assistant_prefix: appended as a raw string after the chat template's
            # own generation-prompt header, matching olmes convert_chat_instance.
            prompts.append(prompt + "Answer:")
        else:
            row_messages = shot_messages + [{"role": "user", "content": f"Question: {question}\nAnswer: "}]
            prompts.append(tokenizer.apply_chat_template(row_messages, tokenize=False, add_generation_prompt=True))

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        stop=["Question:", "</s>", "<|im_end|>"],
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    print_samples(prompts, outputs, print_n)

    predictions = []
    for row, output in zip(rows, outputs):
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


def run_tulu(rows, llm, tokenizer, num_shots, fewshot_jsonl, max_new_tokens, print_n=0):
    return run_fewshot_chat(
        rows, llm, tokenizer, num_shots, fewshot_jsonl, max_new_tokens, "assistant_prefix", print_n
    )


def run_train_format_fewshot(rows, llm, tokenizer, num_shots, fewshot_jsonl, max_new_tokens, print_n=0):
    return run_fewshot_chat(rows, llm, tokenizer, num_shots, fewshot_jsonl, max_new_tokens, "user_suffix", print_n)


# ---------------------------------------------------------------------------

def summarize(predictions):
    return {"accuracy": sum(p["correct"] for p in predictions) / len(predictions)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True, help="GSM8K questions as jsonl.")
    parser.add_argument(
        "--style", required=True,
        choices=["base", "retaining_by_doing", "train_format", "train_format_fewshot", "tulu"],
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Defaults to all visible GPUs.")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Used by --style base/retaining_by_doing/train_format only."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=None,
        help="Defaults to 4096 for --style retaining_by_doing/train_format, 512 for --style "
             "base/tulu/train_format_fewshot (matching each repo's own eval config).",
    )
    parser.add_argument(
        "--num_shots", type=int, default=8, help="Used by --style tulu/train_format_fewshot only."
    )
    parser.add_argument(
        "--fewshot_jsonl", default=None,
        help="Used by --style tulu/train_format_fewshot only. Jsonl of {question,answer} for custom 8-shot "
             "CoT context. If omitted (the default), uses the real fixed STD:GSM8k shots (same as --style base).",
    )
    parser.add_argument(
        "--print_n", type=int, default=0,
        help="Print the exact rendered prompt + raw generation for the first N examples to stdout.",
    )
    args = parser.parse_args()

    max_new_tokens = args.max_new_tokens or (
        512 if args.style in ("base", "tulu", "train_format_fewshot") else 4096
    )

    rows = load_jsonl(args.input_jsonl)
    llm, tokenizer = load_vllm(
        args.model_name_or_path, args.tensor_parallel_size, require_chat_template=(args.style != "base")
    )

    if args.style == "base":
        predictions = run_base(rows, llm, tokenizer, args.batch_size, max_new_tokens, args.print_n)
    elif args.style == "retaining_by_doing":
        predictions = run_retaining_by_doing(rows, llm, tokenizer, args.batch_size, max_new_tokens, args.print_n)
    elif args.style == "train_format":
        predictions = run_train_format(rows, llm, tokenizer, args.batch_size, max_new_tokens, args.print_n)
    elif args.style == "train_format_fewshot":
        predictions = run_train_format_fewshot(
            rows, llm, tokenizer, args.num_shots, args.fewshot_jsonl, max_new_tokens, args.print_n
        )
    else:
        predictions = run_tulu(
            rows, llm, tokenizer, args.num_shots, args.fewshot_jsonl, max_new_tokens, args.print_n
        )

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
