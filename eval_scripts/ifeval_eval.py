"""
Evaluate a chat model on IFEval via vLLM, reproducing two different eval
methodologies from two different repos, both reading prompts from a local
jsonl file. Both repos actually use vLLM for local-model eval
(retaining-by-doing's scripts/eval.sh sets
model_module_path=core.vllm_utils.vLLMCausalLM; oe_eval's default local
backend is also vLLM), so this script does the same.

Styles
------
--style retaining_by_doing
    Reproduces retaining-by-doing's IFEvalDataset + core/evaluation/run.py
    ("ifeval_verify" task): https://github.com/.../retaining-by-doing/blob/main/core/data.py
    Verifier: core/evaluation/ifeval_utils.py's IF_FUNCTIONS_MAP, itself adapted
    from allenai/open-instruct's if_functions.py -- a custom reimplementation of
    the 25 IFEval constraint types, one constraint per example.

    Input jsonl rows need:
        {"messages": [{"role": "user", "content": "..."}],   # or "prompt": "..."
         "ground_truth": "{\\"func_name\\": \\"...\\", ...other kwargs...}"}
    ("ground_truth" may also already be given pre-parsed as "targets": {...}.)

    Requires --retaining_by_doing_path pointing at a checkout of that repo, so
    core/evaluation/ifeval_utils.py can be imported.

--style tulu
    Reproduces oe_eval's `ifeval::tulu` task: configs/tasks.py + tasks/oe_eval_tasks/ifeval.py.
    Verifier: Google's original instructions_registry.py (vendored at
    oe_eval/dependencies/ifeval/), scored both strict and loose, at both
    prompt level (all instructions must pass) and instruction level, exactly
    following oe_eval/dependencies/ifeval/utils.py's
    test_instruction_following_strict/loose (vendored below to avoid pulling
    in the full oe_eval + lm_eval dependency stack).

    Input jsonl rows follow the official HuggingFaceH4/ifeval schema:
        {"key": 1000, "prompt": "...", "instruction_id_list": ["..."], "kwargs": [{...}]}

    Requires --olmes_path pointing at a checkout of the olmes repo, so
    oe_eval.dependencies.ifeval.instructions_registry can be imported (needs
    nltk/spacy/langdetect/emoji/syllapy/immutabledict installed -- the real
    IFEval verifier's own dependencies).

Usage
-----
python ifeval_eval.py --input_jsonl data/eval/ifeval.jsonl --style retaining_by_doing \
    --model_name_or_path /path/to/model --output_jsonl preds.jsonl \
    --retaining_by_doing_path /path/to/retaining-by-doing

python ifeval_eval.py --input_jsonl ifeval_official.jsonl --style tulu \
    --model_name_or_path /path/to/model --output_jsonl preds.jsonl \
    --olmes_path /path/to/olmes
"""
import sys
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


def load_vllm(model_name_or_path, tensor_parallel_size=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.chat_template is None:
        raise ValueError(f"{model_name_or_path} has no chat template; IFEval requires an instruct/chat model.")
    if tensor_parallel_size is None:
        tensor_parallel_size = max(torch.cuda.device_count(), 1)
    llm = LLM(model=model_name_or_path, tensor_parallel_size=tensor_parallel_size)
    return llm, tokenizer


def generate_all(llm, tokenizer, prompt_messages, max_new_tokens):
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in prompt_messages
    ]
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    return [output.outputs[0].text for output in outputs]


# ---------------------------------------------------------------------------
# Style 1: retaining-by-doing (single custom constraint per example)
# ---------------------------------------------------------------------------

def run_retaining_by_doing(rows, llm, tokenizer, max_new_tokens, repo_path):
    sys.path.insert(0, repo_path)
    from core.evaluation.ifeval_utils import IF_FUNCTIONS_MAP  # noqa: E402

    prompt_messages = []
    targets_list = []
    for row in rows:
        if "messages" in row:
            messages = [m for m in row["messages"] if m["role"] != "assistant"]
        else:
            messages = [{"role": "user", "content": row["prompt"]}]
        prompt_messages.append(messages)
        targets_list.append(row["targets"] if "targets" in row else json.loads(row["ground_truth"]))

    output_texts = generate_all(llm, tokenizer, prompt_messages, max_new_tokens)

    predictions = []
    for messages, targets, output_text in zip(prompt_messages, targets_list, output_texts):
        func = IF_FUNCTIONS_MAP[targets["func_name"]]
        try:
            correct = bool(func(output_text, **targets))
        except Exception as e:
            print(f"[warn] verifier error for func_name={targets['func_name']}: {e}")
            correct = False

        predictions.append(dict(
            messages=messages,
            targets=targets,
            output_text=output_text,
            correct=correct,
        ))
    return predictions


def summarize_retaining_by_doing(predictions):
    return {"accuracy": sum(p["correct"] for p in predictions) / len(predictions)}


# ---------------------------------------------------------------------------
# Style 2: oe_eval's ifeval::tulu (official Google verifier, strict + loose,
# prompt-level + instruction-level). Vendored from
# oe_eval/dependencies/ifeval/utils.py to avoid the lm_eval dependency.
# ---------------------------------------------------------------------------

def _score_following(instructions_registry, instruction_id_list, kwargs_list, prompt, response, loose):
    is_following_list = []
    if loose:
        r = response.split("\n")
        candidates = [
            response,
            response.replace("*", ""),
            "\n".join(r[1:]).strip(),
            "\n".join(r[:-1]).strip(),
            "\n".join(r[1:-1]).strip(),
            "\n".join(r[1:]).strip().replace("*", ""),
            "\n".join(r[:-1]).strip().replace("*", ""),
            "\n".join(r[1:-1]).strip().replace("*", ""),
        ]
    else:
        candidates = [response]

    for index, instruction_id in enumerate(instruction_id_list):
        try:
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            kwargs = {k: v for k, v in kwargs_list[index].items() if v}
            instruction.build_description(**kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)

            is_following = False
            for candidate in candidates:
                if candidate.strip() and instruction.check_following(candidate):
                    is_following = True
                    break
            is_following_list.append(is_following)
        except Exception as e:
            print(f"[warn] verifier error for instruction_id={instruction_id}: {e}")
            is_following_list.append(False)
    return is_following_list


def run_tulu(rows, llm, tokenizer, max_new_tokens, olmes_path):
    sys.path.insert(0, olmes_path)
    from oe_eval.dependencies.ifeval import instructions_registry  # noqa: E402

    prompt_messages = [[{"role": "user", "content": row["prompt"]}] for row in rows]
    output_texts = generate_all(llm, tokenizer, prompt_messages, max_new_tokens)

    predictions = []
    for row, output_text in zip(rows, output_texts):
        strict_list = _score_following(
            instructions_registry, row["instruction_id_list"], row["kwargs"], row["prompt"], output_text, loose=False
        )
        loose_list = _score_following(
            instructions_registry, row["instruction_id_list"], row["kwargs"], row["prompt"], output_text, loose=True
        )

        predictions.append(dict(
            key=row.get("key"),
            prompt=row["prompt"],
            instruction_id_list=row["instruction_id_list"],
            output_text=output_text,
            inst_level_strict=strict_list,
            inst_level_loose=loose_list,
            prompt_level_strict=all(strict_list),
            prompt_level_loose=all(loose_list),
        ))
    return predictions


def summarize_tulu(predictions):
    all_strict_inst = [v for p in predictions for v in p["inst_level_strict"]]
    all_loose_inst = [v for p in predictions for v in p["inst_level_loose"]]
    return {
        "prompt_level_strict_acc": sum(p["prompt_level_strict"] for p in predictions) / len(predictions),
        "prompt_level_loose_acc": sum(p["prompt_level_loose"] for p in predictions) / len(predictions),
        "inst_level_strict_acc": sum(all_strict_inst) / len(all_strict_inst),
        "inst_level_loose_acc": sum(all_loose_inst) / len(all_loose_inst),
    }


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--style", required=True, choices=["retaining_by_doing", "tulu"])
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Defaults to all visible GPUs.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=None,
        help="Defaults to 4096 for --style retaining_by_doing, 2048 for --style tulu "
             "(matching each repo's own eval config).",
    )
    parser.add_argument(
        "--retaining_by_doing_path", default=None,
        help="Path to a retaining-by-doing checkout. Required for --style retaining_by_doing.",
    )
    parser.add_argument(
        "--olmes_path", default=None,
        help="Path to an olmes checkout. Required for --style tulu.",
    )
    args = parser.parse_args()

    if args.style == "retaining_by_doing" and not args.retaining_by_doing_path:
        parser.error("--retaining_by_doing_path is required for --style retaining_by_doing")
    if args.style == "tulu" and not args.olmes_path:
        parser.error("--olmes_path is required for --style tulu")

    max_new_tokens = args.max_new_tokens or (4096 if args.style == "retaining_by_doing" else 2048)

    rows = load_jsonl(args.input_jsonl)
    llm, tokenizer = load_vllm(args.model_name_or_path, args.tensor_parallel_size)

    if args.style == "retaining_by_doing":
        predictions = run_retaining_by_doing(rows, llm, tokenizer, max_new_tokens, args.retaining_by_doing_path)
        metrics = summarize_retaining_by_doing(predictions)
    else:
        predictions = run_tulu(rows, llm, tokenizer, max_new_tokens, args.olmes_path)
        metrics = summarize_tulu(predictions)

    write_jsonl(args.output_jsonl, predictions)
    metrics_path = args.output_jsonl.rsplit(".jsonl", 1)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"style={args.style}")
    for k, v in metrics.items():
        print(f"{k}={v:.4f}")
    print(f"predictions written to {args.output_jsonl}")
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
