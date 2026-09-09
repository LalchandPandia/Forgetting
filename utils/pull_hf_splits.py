"""Export official train/test splits for HuggingFace-hosted datasets to JSONL.

Reads `prompts/prompts_by_task_modified.yaml`, and for every entry backed by a
HuggingFace dataset (i.e. not a local `data/*.json` file), downloads the
dataset's own train split and its own test split, writing each to a separate
JSONL file. If a dataset has no official test split, its validation split is
used as the test split instead. Locally-curated `data/*.json` entries have no
official split and are skipped entirely.

Files are written under {output_dir}/{dataset_name}/, one directory per HF
dataset (nested by org/repo when the dataset name contains a "/", e.g.
".../openai/gsm8k/train.jsonl"). When a dataset has multiple tasks/configs
(e.g. super_glue: boolq, copa, ...), the task name prefixes the filename
within that directory (e.g. "boolq_train.jsonl", "copa_train.jsonl").

Each output row is written in the open-instruct SFT schema, e.g.:
    {"dataset": "gsm8k", "id": 0, "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}
The user turn is built from the task's `instruction` + `user_input` fields
(as declared in the prompts yaml) so that the row is self-contained even for
tasks (e.g. classification with a category legend) whose labels are
meaningless without the instruction text. Pass --no_instruction_prefix to
omit the instruction and keep only the raw question.

Usage:
    cd utils/
    python pull_hf_splits.py
    python pull_hf_splits.py --only cais/mmlu nyu-mll/glue
    python pull_hf_splits.py --output_dir /some/other/path
"""

import argparse
import json
import os
import traceback

import yaml
from datasets import concatenate_datasets, load_dataset


def is_local_entry(dataset_name):
    """Local, author-curated datasets are stored under data/*.json and have no official split."""
    return dataset_name.startswith("data/") or dataset_name.endswith(".json")


def resolve_split_names(entry, available_splits):
    """Figure out which of the dataset's actual splits correspond to train/test.

    `entry` may declare a `split` mapping (e.g. {"train": "train_r1", "test": "test_r1"})
    for datasets whose hub split names don't follow the train/validation/test convention.
    Returns (train_name, test_name, used_validation_fallback).
    """

    split_map = entry.get("split") or {}

    train_name = split_map.get("train", "train")
    test_name = split_map.get("test")

    if test_name is None and "test" in available_splits and "test" != train_name:
        test_name = "test"

    used_fallback = False
    if test_name is None or test_name not in available_splits:
        valid_name = split_map.get("valid")
        if valid_name is None:
            if "validation" in available_splits:
                valid_name = "validation"
            elif "valid" in available_splits:
                valid_name = "valid"

        if valid_name and valid_name in available_splits and valid_name != train_name:
            test_name = valid_name
            used_fallback = True
        else:
            test_name = None

    return train_name, test_name, used_fallback


def load_split_for_config(dataset_name, config_name, entry, trust_remote_code):
    """Load a dataset (optionally for one config/subset) and resolve its train/test splits."""

    ds = load_dataset(dataset_name, config_name, trust_remote_code=trust_remote_code)
    available_splits = list(ds.keys())

    train_name, test_name, used_fallback = resolve_split_names(entry, available_splits)

    train_split = ds[train_name] if train_name in available_splits else None
    test_split = ds[test_name] if test_name and test_name in available_splits else None

    return train_split, test_split, used_fallback, available_splits


def sanitize(name):
    return str(name).replace("/", "__").replace(" ", "_")


def short_dataset_id(dataset_name):
    """Short display name for the "dataset" field, e.g. "openai/gsm8k" -> "gsm8k"."""
    return dataset_name.split("/")[-1].lower()


def output_paths(output_dir, dataset_name, task_name):
    """Files land under {output_dir}/{dataset_name}/, one dataset per directory.

    `dataset_name` may itself contain "/" (e.g. "openai/gsm8k"), which nests
    the directory by org/repo, matching the HF Hub layout. When a dataset has
    multiple tasks/configs (e.g. super_glue: boolq, copa, ...) the task name
    prefixes the filename so they don't collide within that directory.
    """

    dataset_dir = os.path.join(output_dir, dataset_name)
    prefix = "" if task_name == "default" else f"{sanitize(task_name)}_"
    return (
        os.path.join(dataset_dir, f"{prefix}train.jsonl"),
        os.path.join(dataset_dir, f"{prefix}test.jsonl"),
    )


def build_user_query(user_input_spec, example):
    """Port of base_model.BaseModel.create_prompt's user-query construction.

    `user_input_spec` is either a single column name, or (for composite inputs
    like "PREMISE"/"HYPOTHESIS" pairs) a dict mapping a label to either a
    column name or a {key_name, fields, concat_symbol} spec for nested fields
    (e.g. ARC's answer choices).
    """

    if not isinstance(user_input_spec, dict):
        return str(example[user_input_spec])

    parts = ""
    for label, spec in user_input_spec.items():
        if isinstance(spec, dict):
            parts += "\n" + label + " - "
            data_field = example[spec["key_name"]]
            field_a, field_b = spec["fields"]
            concat_symbol = spec.get("concat_symbol")
            if concat_symbol:
                parts += "\n"
                for a, b in zip(data_field[field_a], data_field[field_b]):
                    parts += f"{a} {concat_symbol} {b}\n"
            else:
                parts += str(data_field[field_a])
        else:
            parts += f"{label} - {example[spec]} [SEP] \n"

    query = parts.strip().strip("\n")
    if query.endswith("[SEP]"):
        query = query[: -len("[SEP]")].strip()
    return query


def build_messages(entry, example, include_instruction):
    instruction = entry.get("instruction")
    user_query = build_user_query(entry["user_input"], example)
    assistant_content = str(example[entry["assistant_output"]])

    if include_instruction and instruction:
        user_content = f"{instruction.strip()}\n\n{user_query}"
    else:
        user_content = user_query

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def to_sft_schema(dataset, entry, dataset_field, include_instruction):
    """Reshape a raw HF split into {"dataset", "id", "messages"} rows."""

    def _map(example, idx):
        return {
            "dataset": dataset_field,
            "id": idx,
            "messages": build_messages(entry, example, include_instruction),
        }

    return dataset.map(_map, with_indices=True, remove_columns=dataset.column_names)


def write_jsonl(dataset, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.to_json(path, orient="records", lines=True)


def process_entry(dataset_name, task_name, entry, output_dir, trust_remote_code, include_instruction, log):
    subset = entry.get("subset")
    subsets = subset.split() if subset else [task_name]

    train_parts, test_parts = [], []
    used_fallback = False
    any_split_seen = False

    for sub in subsets:
        try:
            train_split, test_split, fallback, available = load_split_for_config(
                dataset_name, sub, entry, trust_remote_code
            )
        except Exception:
            log(f"[ERROR] {dataset_name} / {sub}: failed to load\n{traceback.format_exc()}")
            continue

        any_split_seen = True
        used_fallback = used_fallback or fallback

        if train_split is not None:
            if subset:
                train_split = train_split.add_column("category", [sub] * len(train_split))
            train_parts.append(train_split)
        else:
            log(f"[WARN] {dataset_name} / {sub}: no train split found (available: {available})")

        if test_split is not None:
            if subset:
                test_split = test_split.add_column("category", [sub] * len(test_split))
            test_parts.append(test_split)
        else:
            log(f"[WARN] {dataset_name} / {sub}: no test or validation split found (available: {available})")

    if not any_split_seen:
        return

    train_path, test_path = output_paths(output_dir, dataset_name, task_name)
    dataset_field = short_dataset_id(dataset_name)

    if train_parts:
        train_ds = train_parts[0] if len(train_parts) == 1 else concatenate_datasets(train_parts)
        train_ds = to_sft_schema(train_ds, entry, dataset_field, include_instruction)
        write_jsonl(train_ds, train_path)
        log(f"[OK] {dataset_name} / {task_name}: wrote {train_path} ({train_ds.num_rows} rows)")

    if test_parts:
        test_ds = test_parts[0] if len(test_parts) == 1 else concatenate_datasets(test_parts)
        test_ds = to_sft_schema(test_ds, entry, dataset_field, include_instruction)
        write_jsonl(test_ds, test_path)
        tag = " [from validation]" if used_fallback else ""
        log(f"[OK] {dataset_name} / {task_name}: wrote {test_path}{tag} ({test_ds.num_rows} rows)")


def main():
    parser = argparse.ArgumentParser(description="Export official HF train/test splits to JSONL")
    parser.add_argument("--yaml_path", type=str,
                         default="../dataefficiency/prompts/prompts_by_task_modified.yaml")
    parser.add_argument("--output_dir", type=str, default="/net/spaces/scratch/lcpandia/data/processed")
    parser.add_argument("--only", type=str, nargs="*", default=None,
                         help="restrict to these top-level dataset names")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    parser.add_argument("--no_instruction_prefix", action="store_true",
                         help="omit the task instruction from the user turn, keeping only the raw question")
    args = parser.parse_args()

    with open(args.yaml_path, "r") as f:
        config = yaml.safe_load(f)

    def log(msg):
        print(msg)

    for dataset_name, tasks in config.items():
        if is_local_entry(dataset_name):
            log(f"[SKIP] {dataset_name}: locally-curated dataset, no official split")
            continue

        if args.only and dataset_name not in args.only:
            continue

        for task_name, entry in tasks.items():
            process_entry(
                dataset_name=dataset_name,
                task_name=task_name if task_name is not None else "default",
                entry=entry,
                output_dir=args.output_dir,
                trust_remote_code=not args.no_trust_remote_code,
                include_instruction=not args.no_instruction_prefix,
                log=log,
            )


if __name__ == "__main__":
    main()
