"""Evaluate a single model across every dataset used in the dataefficiency project.

This is a standalone counterpart to dataefficiency/scripts/eval.py: instead of
evaluating one (model, dataset) pair per invocation, it takes just a model and
loops over the full 40-task catalog (the 30 tasks common to Llama/Mistral/Qwen
plus the 10 extra tasks that were previously only run on Llama), reusing the
dataefficiency package's model-family classes, prompt templates, and metrics
so behavior stays identical to the original per-task pipeline.

It intentionally does NOT modify anything under dataefficiency/ - it imports
from there (scripts.base_model, scripts.llama_child, etc.) and works around
two hardcoded assumptions in that code instead of editing it:

  1. BaseModel.load_task_prompt() reads '../prompts/prompts_by_task_modified.yaml'
     relative to the CWD (it assumes you `cd scripts/` first). We replicate
     that by chdir-ing into dataefficiency/scripts/ before touching any of it.
  2. BaseModel.get_data() resolves local `data/*.json` datasets against
     f"{Path.home()}/ft-intrinsic-dim/{dataset_name}" - a path specific to the
     original authors' machine. We monkeypatch Path.home() (only for this
     process) to point at a small fake-home directory inside this folder that
     symlinks `ft-intrinsic-dim/data` to the data/ folder extracted from
     dataefficiency/data.tar.gz, so the untouched upstream code resolves the
     right files.

Usage:
    cd dataefficiency_eval/
    python eval_all_datasets.py --model_name meta-llama/Llama-3.1-8B-Instruct

Run `python eval_all_datasets.py --list_tasks` to see the task catalog without
loading a model.
"""

import argparse
import json
import os
import pathlib
import sys
import tarfile
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GenerationConfig

THIS_DIR = Path(__file__).resolve().parent
DATAEFFICIENCY_DIR = THIS_DIR.parent / "dataefficiency"
DATAEFFICIENCY_SCRIPTS_DIR = DATAEFFICIENCY_DIR / "scripts"
CACHE_DIR = THIS_DIR / ".cache"
FAKE_HOME_DIR = CACHE_DIR / "fake_home"
DATA_TAR = DATAEFFICIENCY_DIR / "data.tar.gz"
EXTRACTED_DATA_DIR = DATAEFFICIENCY_DIR / "data"

if not DATAEFFICIENCY_DIR.is_dir():
    raise SystemExit(
        f"Expected to find the dataefficiency project at {DATAEFFICIENCY_DIR}, "
        "but it does not exist. This script expects to live as a sibling of "
        "dataefficiency/ inside the Forgetting repo."
    )

# dataefficiency's scripts/utils import each other as `scripts.X` / `utils.X`,
# i.e. relative to the dataefficiency/ root - so that's what needs to be on
# sys.path (not dataefficiency/scripts/ itself).
sys.path.insert(0, str(DATAEFFICIENCY_DIR))

from scripts.llama_child import Llama  # noqa: E402
from scripts.mistral_child import Mistral  # noqa: E402
from scripts.qwen_child import Qwen  # noqa: E402
from utils.calculate_eval_metrics import (  # noqa: E402
    calculate_bleu,
    calculate_f1,
    calculate_rouge,
    calculate_sequence_accuracy,
    calculate_str_contains,
)

MODEL_CLASSES = {"llama": Llama, "mistral": Mistral, "qwen": Qwen}


# ---------------------------------------------------------------------------
# The 40-dataset catalog: 30 tasks evaluated on Llama/Mistral/Qwen, plus the
# 10 extra tasks that were only run on Llama in the committed results. Each
# entry's (dataset_name, task) pair is the exact key used to look the prompt
# template up in prompts/prompts_by_task_modified.yaml.
# ---------------------------------------------------------------------------
CORE_TASKS = [
    {"name": "ade_corpus_v2_classification", "dataset_name": "ade-benchmark-corpus/ade_corpus_v2", "task": "Ade_corpus_v2_classification"},
    {"name": "anli", "dataset_name": "facebook/anli", "task": None},
    {"name": "banking77", "dataset_name": "legacy-datasets/banking77", "task": None},
    {"name": "boolean_expressions", "dataset_name": "data/boolean_expressions.json", "task": "boolean_expressions"},
    {"name": "boolq", "dataset_name": "super_glue", "task": "boolq"},
    {"name": "circa", "dataset_name": "google-research-datasets/circa", "task": None},
    {"name": "commonsense_qa", "dataset_name": "tau/commonsense_qa", "task": None},
    {"name": "fig_qa", "dataset_name": "nightingal3/fig-qa", "task": None},
    {"name": "formal_fallacies_syllogisms_negation", "dataset_name": "data/formal_fallacies_syllogisms_negation.json", "task": "formal_fallacies_syllogisms_negation"},
    {"name": "high", "dataset_name": "ehovy/race", "task": "high"},
    {"name": "hyperbaton", "dataset_name": "data/hyperbaton.json", "task": "hyperbaton"},
    {"name": "medmcqa", "dataset_name": "openlifescienceai/medmcqa", "task": None},
    {"name": "mmlu", "dataset_name": "cais/mmlu", "task": None},
    {"name": "mnist_ascii", "dataset_name": "data/mnist_ascii.json", "task": "mnist_ascii"},
    {"name": "mnli", "dataset_name": "nyu-mll/glue", "task": "mnli"},
    {"name": "mrpc", "dataset_name": "nyu-mll/glue", "task": "mrpc"},
    {"name": "object_counting", "dataset_name": "data/object_counting.json", "task": "object_counting"},
    {"name": "overruling", "dataset_name": "LawInformedAI/overruling", "task": None},
    {"name": "qnli", "dataset_name": "nyu-mll/glue", "task": "qnli"},
    {"name": "qqp", "dataset_name": "nyu-mll/glue", "task": "qqp"},
    {"name": "quail", "dataset_name": "textmachinelab/quail", "task": None},
    {"name": "reasoning_about_colored_objects", "dataset_name": "data/reasoning_about_colored_objects.json", "task": "reasoning_about_colored_objects"},
    {"name": "rte", "dataset_name": "super_glue", "task": "rte"},
    {"name": "sports_understanding", "dataset_name": "data/sports_understanding.json", "task": "sports_understanding"},
    {"name": "sst2", "dataset_name": "nyu-mll/glue", "task": "sst2"},
    {"name": "temporal_sequences", "dataset_name": "data/temporal_sequences.json", "task": "temporal_sequences"},
    {"name": "toxicchat0124", "dataset_name": "lmsys/toxic-chat", "task": "toxicchat0124"},
    {"name": "tracking_shuffled_objects", "dataset_name": "data/tracking_shuffled_objects.json", "task": "tracking_shuffled_objects"},
    {"name": "web_of_lies", "dataset_name": "data/web_of_lies.json", "task": "web_of_lies"},
    {"name": "wic", "dataset_name": "super_glue", "task": "wic"},
]

LLAMA_ONLY_EXTRA_TASKS = [
    {"name": "coqa", "dataset_name": "data/coqa.json", "task": "coqa"},
    {"name": "disaster_response_messages", "dataset_name": "community-datasets/disaster_response_messages", "task": None},
    {"name": "disfl_qa", "dataset_name": "data/disfl_qa.json", "task": "disfl_qa"},
    {"name": "imbalanced", "dataset_name": "clinc/clinc_oos", "task": "imbalanced"},
    {"name": "mmlu_pro", "dataset_name": "TIGER-Lab/MMLU-Pro", "task": None},
    {"name": "multitask_data_5", "dataset_name": "data/multitask_data_5.json", "task": "multitask_data_5"},
    {"name": "multitask_data_5_2", "dataset_name": "data/multitask_data_5_2.json", "task": "multitask_data_5_2"},
    {"name": "qa_wikidata", "dataset_name": "data/qa_wikidata.json", "task": "qa_wikidata"},
    {"name": "squad_v2", "dataset_name": "data/squad_v2.json", "task": "squad_v2"},
    {"name": "twitter_financial_news_sentiment", "dataset_name": "zeroshot/twitter-financial-news-sentiment", "task": None},
]

ALL_TASKS = CORE_TASKS + LLAMA_ONLY_EXTRA_TASKS

# Tasks whose local JSON file is not actually present in dataefficiency/data.tar.gz
# as shipped in this repo. They're kept in the catalog (matching the dataset
# inventory) but will fail at data-loading time with a clear error rather than
# being silently dropped.
KNOWN_MISSING_LOCAL_FILES = {"coqa", "squad_v2", "multitask_data_5", "multitask_data_5_2"}


# ---------------------------------------------------------------------------
# Environment setup: fix the two hardcoded-path assumptions in dataefficiency's
# BaseModel without editing that file.
# ---------------------------------------------------------------------------

def ensure_local_json_data_available():
    """Extract data.tar.gz if needed and wire up the fake-home path that
    BaseModel.get_data()'s hardcoded `{home}/ft-intrinsic-dim/{dataset_name}`
    lookup expects, without touching the real $HOME."""

    if not EXTRACTED_DATA_DIR.is_dir():
        if not DATA_TAR.is_file():
            raise SystemExit(
                f"Expected dataefficiency/data.tar.gz at {DATA_TAR} but it is "
                "missing, and the extracted data/ folder doesn't exist either. "
                "Local-JSON tasks (BBH-style tasks, mnist_ascii, etc.) cannot run."
            )
        print(f"Extracting {DATA_TAR} -> {EXTRACTED_DATA_DIR} ...")
        with tarfile.open(DATA_TAR) as tf:
            tf.extractall(DATAEFFICIENCY_DIR)

    fake_ft_intrinsic_dim = FAKE_HOME_DIR / "ft-intrinsic-dim"
    fake_ft_intrinsic_dim.mkdir(parents=True, exist_ok=True)
    data_link = fake_ft_intrinsic_dim / "data"
    if not data_link.exists():
        data_link.symlink_to(EXTRACTED_DATA_DIR)

    # BaseModel.get_data() does `home_dir = str(Path.home())` and then reads
    # f"{home_dir}/ft-intrinsic-dim/{self.dataset_name}" (dataset_name already
    # includes the "data/" prefix, e.g. "data/object_counting.json").
    pathlib.Path.home = classmethod(lambda cls: FAKE_HOME_DIR)


def chdir_into_dataefficiency_scripts():
    """BaseModel.load_task_prompt() opens '../prompts/...' relative to CWD,
    i.e. it assumes you're running from dataefficiency/scripts/."""
    os.chdir(DATAEFFICIENCY_SCRIPTS_DIR)


# ---------------------------------------------------------------------------
# Eval helpers (adapted from dataefficiency/scripts/eval.py)
# ---------------------------------------------------------------------------

def get_dataloader(data, split, tokenizer, batch_size, mycollator, max_seq_len, assistant_start_token):
    model_inputs = []
    for i in range(len(data[split])):
        prompt = data[split][i]['text'].split(assistant_start_token)[0] + f"\n {assistant_start_token}"
        tok_dict = tokenizer(prompt, add_special_tokens=False)
        if len(tok_dict['input_ids']) < max_seq_len:
            tok_dict['data_idx'] = i
            model_inputs.append(tok_dict)

    return DataLoader(model_inputs, collate_fn=mycollator, batch_size=batch_size, shuffle=False)


def clean_model_output(out, assistant_start_token, eos_token):
    out = out.split(assistant_start_token)[-1]
    out = out.split('####')[-1]
    out = out.strip('\n').split('\n')[0]
    out = out.split(eos_token)[0].strip().lower()
    out = out.strip('.').strip()
    return out


def clean_target_output(tar):
    tar = str(tar)
    tar = tar.split('####')[-1]
    tar = tar.strip('.').strip().lower()
    return tar


def eval_on_heldout(model, data, split, tokenizer, dataloader, answer_len, target_key,
                     assistant_start_token, eos_token, device):
    d_res = {'target': [], 'pred_clean': [], 'pred_org': []}

    max_new_toks = 30 if answer_len == "short" else 500

    generation_config = GenerationConfig(
        max_new_tokens=max_new_toks,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beam=4,  # kept as-is: matches upstream eval.py (should be num_beams; see README caveat)
        do_sample=False,
    )

    with torch.no_grad():
        for batch in dataloader:
            batch_in = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            outputs = model.generate(**batch_in, generation_config=generation_config, tokenizer=tokenizer)
            response = tokenizer.batch_decode(outputs)
            response_clean = [clean_model_output(out=r, assistant_start_token=assistant_start_token, eos_token=eos_token) for r in response]
            target_clean = [clean_target_output(data[split][int(idx)][target_key]) for idx in batch['data_idx']]
            d_res['target'] += target_clean
            d_res['pred_clean'] += response_clean
            d_res['pred_org'] += response

    result = pd.DataFrame(d_res)
    result['target'] = result['target'].astype(str)
    result['pred_clean'] = result['pred_clean'].astype(str)
    return result


def calculate_exact_string_match(preds, refs):
    accuracy = (preds == refs).sum() / len(preds)
    return {'exact_string_match_accuracy': accuracy}


def evaluate_one_task(model, tokenizer, model_prefix, model_name, task, args, device):
    """Run the full get_data -> generate -> score pipeline for a single task."""

    model_cls = MODEL_CLASSES[model_prefix](
        model_name=model_name,
        dataset_name=task["dataset_name"],
        data_size=None,
        task=task["task"],
        use_quantized=args.use_quantized,
        use_peft=False,
        peft_method=None,
        r=None,
        density=None,
        seed=args.seed,
    )

    prompt = model_cls.load_task_prompt()
    target_key = prompt['assistant_output']
    answer_len = prompt['answer_len']
    assistant_start_token = model_cls.assistant_start_token
    eos_token = model_cls.eos_token

    data = model_cls.get_data(tokenizer=tokenizer, max_seq_len=args.max_seq_len, filter_long_seq=False)

    if data[args.split].num_rows > args.max_test_examples:
        data[args.split] = data[args.split].shuffle(seed=args.seed).select(range(args.max_test_examples))

    mycollator = model_cls.get_datacollator(tokenizer=tokenizer, completion_only=True)
    loader = get_dataloader(
        data=data, split=args.split, tokenizer=tokenizer, batch_size=args.batch_size,
        mycollator=mycollator, max_seq_len=args.max_seq_len, assistant_start_token=assistant_start_token,
    )

    eval_result = eval_on_heldout(
        model=model, data=data, split=args.split, tokenizer=tokenizer, answer_len=answer_len,
        dataloader=loader, target_key=target_key, assistant_start_token=assistant_start_token,
        eos_token=eos_token, device=device,
    )

    if args.save_predictions:
        pred_dir = Path(args.result_dir) / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        eval_result.to_json(pred_dir / f"{model_prefix}_{task['name']}_predictions.jsonl", orient='records', lines=True)

    metrics = calculate_exact_string_match(
        preds=eval_result['pred_clean'].str.replace(' ', ''),
        refs=eval_result['target'].str.replace(' ', ''),
    )

    if args.calculate_all_metrics:
        metrics.update(calculate_bleu(preds=eval_result['pred_clean'], refs=eval_result['target']))
        metrics.update(calculate_rouge(preds=eval_result['pred_clean'], refs=eval_result['target']))
        metrics.update(calculate_sequence_accuracy(preds=eval_result['pred_clean'], refs=eval_result['target']))
        metrics.update(calculate_f1(preds=eval_result['pred_clean'], refs=eval_result['target']))
        metrics.update(calculate_str_contains(preds=eval_result['pred_clean'], refs=eval_result['target']))

    metrics['num_examples'] = int(len(eval_result))
    return metrics


def get_model_family(model_name, override=None):
    if override:
        return override
    lname = model_name.lower()
    for family in ("llama", "mistral", "qwen"):
        if family in lname:
            return family
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one model on every dataset in the dataefficiency task catalog."
    )
    parser.add_argument('--model_name', type=str, help='HF model id or local checkpoint path to evaluate')
    parser.add_argument('--model_family', type=str, choices=list(MODEL_CLASSES), default=None,
                         help='Override family detection (needed if the checkpoint path does not contain "llama"/"mistral"/"qwen")')
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--tasks', type=str, default=None,
                         help='Comma-separated task names to run (default: all 40). See --list_tasks.')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument('--max_test_examples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_quantized', action='store_true')
    parser.add_argument('--use_flash_attention', action='store_true')
    parser.add_argument('--use_safetensor', action='store_true')
    parser.add_argument('--calculate_all_metrics', action='store_true',
                         help='Also compute BLEU/ROUGE/sequence-accuracy/F1/str-contains (slower)')
    parser.add_argument('--save_predictions', action='store_true',
                         help='Write per-example predictions to result_dir/predictions/')
    parser.add_argument('--result_dir', type=str, default=str(THIS_DIR / "results"))
    parser.add_argument('--output_name', type=str, default=None,
                         help='Filename (without dir) for the aggregate results JSON; default: <model_prefix>_<sanitized model name>_all_tasks.json')
    parser.add_argument('--list_tasks', action='store_true', help='Print the task catalog and exit (no model needed)')
    args = parser.parse_args()

    if args.list_tasks:
        for t in ALL_TASKS:
            flag = " (local file missing from data.tar.gz)" if t["name"] in KNOWN_MISSING_LOCAL_FILES else ""
            print(f"{t['name']:40s} dataset_name={t['dataset_name']!r:45s} task={t['task']!r}{flag}")
        return

    if not args.model_name:
        parser.error("--model_name is required unless --list_tasks is passed")

    model_prefix = get_model_family(args.model_name, args.model_family)
    if model_prefix is None:
        parser.error(
            f"Could not infer model family from --model_name={args.model_name!r}; "
            "pass --model_family {llama,mistral,qwen} explicitly."
        )

    selected_tasks = ALL_TASKS
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(',') if t.strip()}
        unknown = wanted - {t["name"] for t in ALL_TASKS}
        if unknown:
            parser.error(f"Unknown task name(s): {sorted(unknown)}. See --list_tasks.")
        selected_tasks = [t for t in ALL_TASKS if t["name"] in wanted]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        output_path = result_dir / args.output_name
    else:
        sanitized = args.model_name.replace('/', '__')
        output_path = result_dir / f"{model_prefix}_{sanitized}_all_tasks.json"

    needs_local_data = any(t["dataset_name"].endswith('.json') for t in selected_tasks)
    if needs_local_data:
        ensure_local_json_data_available()
    chdir_into_dataefficiency_scripts()

    # Load the model + tokenizer once and reuse across all tasks.
    loader_cls = MODEL_CLASSES[model_prefix](
        model_name=args.model_name,
        dataset_name=selected_tasks[0]["dataset_name"],
        data_size=None,
        task=selected_tasks[0]["task"],
        use_quantized=args.use_quantized,
        use_peft=False,
        peft_method=None,
        r=None,
        density=None,
        seed=args.seed,
    )
    print(f"Loading model {args.model_name} (family={model_prefix}) ...")
    model, tokenizer = loader_cls.get_model_and_tokenizer(
        tokenizer_path=args.tokenizer_path,
        use_flash_attention=args.use_flash_attention,
        use_safetensors=args.use_safetensor,
    )
    tokenizer.padding_side = 'left'
    model.eval()
    model.to(device)
    print(f"Model loaded on device(s): {set(p.device for p in model.parameters())}")

    results = {}
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resuming into existing results file {output_path} ({len(results)} tasks already recorded)")

    for i, task in enumerate(selected_tasks):
        name = task["name"]
        print(f"\n[{i + 1}/{len(selected_tasks)}] {name} (dataset_name={task['dataset_name']}, task={task['task']})")
        t0 = time.time()
        try:
            metrics = evaluate_one_task(model, tokenizer, model_prefix, args.model_name, task, args, device)
            metrics['elapsed_seconds'] = round(time.time() - t0, 1)
            results[name] = metrics
            print(f"  -> {metrics}")
        except Exception as e:  # noqa: BLE001 - a bad task shouldn't kill the whole 40-task sweep
            print(f"  !! FAILED: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e)}

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Results written to {output_path}")
    ok = {k: v for k, v in results.items() if 'error' not in v}
    failed = {k: v for k, v in results.items() if 'error' in v}
    print(f"{len(ok)} tasks succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed tasks: {sorted(failed)}")


if __name__ == '__main__':
    main()
