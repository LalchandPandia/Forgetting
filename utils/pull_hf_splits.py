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


def safe_load_dataset(dataset_name, config_name, entry, allow_remote_code):
    """Load a dataset, working around two version-dependent `datasets` quirks.

    1. Newer `datasets` versions reject `trust_remote_code=True` outright for
       plain data-file datasets (no loading script), with an error telling
       you to remove the argument. So we try without it first, and only
       retry with it if the failure looks like it actually needs remote code.
    2. Some hub datasets still ship a legacy Python loading script. Recent
       `datasets` versions refuse to run *any* loading script at all (even
       with trust_remote_code) and raise "Dataset scripts are no longer
       supported". The Hub auto-converts such datasets to a script-free
       Parquet mirror on the `refs/convert/parquet` ref, so we retry there
       -- but ONLY if that mirror actually has the split name(s) this entry
       declares. The auto-conversion sometimes collapses many custom splits
       (e.g. "train_coling2022", "train_2020", "train_random", ...) down to
       a single generic "train"/"validation"/"test", which would silently
       hand back the wrong data for datasets whose entry.split names a
       specific variant. If the declared splits aren't there, we raise
       loudly instead of guessing.
    """

    try:
        return load_dataset(dataset_name, config_name)
    except Exception as e:
        msg = str(e)

        if "Dataset scripts are no longer supported" in msg:
            fallback_ds = load_dataset(dataset_name, config_name, revision="refs/convert/parquet")
            declared_splits = list((entry.get("split") or {}).values())
            missing = [s for s in declared_splits if s not in fallback_ds]
            if missing:
                raise RuntimeError(
                    f"{dataset_name} (config={config_name}) needs a loading script that `datasets` "
                    f"no longer runs, and its auto-converted Parquet mirror (refs/convert/parquet) "
                    f"doesn't have the declared split(s) {missing} -- it only has "
                    f"{list(fallback_ds.keys())}. This dataset needs a manual, dataset-specific fix "
                    f"(e.g. reading the right raw files directly), not a generic fallback."
                ) from e
            return fallback_ds

        needs_remote_code = "trust_remote_code" in msg or "custom code" in msg
        if allow_remote_code and needs_remote_code:
            return load_dataset(dataset_name, config_name, trust_remote_code=True)

        raise


def load_split_for_config(dataset_name, config_name, entry, allow_remote_code):
    """Load a dataset (optionally for one config/subset) and resolve its train/test splits."""

    ds = safe_load_dataset(dataset_name, config_name, entry, allow_remote_code)
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


def process_entry(dataset_name, task_name, entry, output_dir, allow_remote_code, include_instruction, log):
    """`task_name` is the raw yaml key, which is None when the dataset has a single
    (default) config -- that None must reach `load_dataset()` as-is (omitting the
    config argument), so it is kept separate from `display_task`, the "default"
    placeholder used only for log messages and output file naming.
    """

    display_task = task_name if task_name is not None else "default"
    subset = entry.get("subset")
    subsets = subset.split() if subset else [task_name]

    train_parts, test_parts = [], []
    used_fallback = False
    any_split_seen = False

    for sub in subsets:
        try:
            train_split, test_split, fallback, available = load_split_for_config(
                dataset_name, sub, entry, allow_remote_code
            )
        except Exception:
            log(f"[ERROR] {dataset_name} / {display_task}: failed to load\n{traceback.format_exc()}")
            continue

        any_split_seen = True
        used_fallback = used_fallback or fallback

        if train_split is not None:
            if subset:
                train_split = train_split.add_column("category", [sub] * len(train_split))
            train_parts.append(train_split)
        else:
            log(f"[WARN] {dataset_name} / {display_task}: no train split found (available: {available})")

        if test_split is not None:
            if subset:
                test_split = test_split.add_column("category", [sub] * len(test_split))
            test_parts.append(test_split)
        else:
            log(f"[WARN] {dataset_name} / {display_task}: no test or validation split found (available: {available})")

    if not any_split_seen:
        return

    train_path, test_path = output_paths(output_dir, dataset_name, display_task)
    dataset_field = short_dataset_id(dataset_name)

    try:
        if train_parts:
            train_ds = train_parts[0] if len(train_parts) == 1 else concatenate_datasets(train_parts)
            train_ds = to_sft_schema(train_ds, entry, dataset_field, include_instruction)
            write_jsonl(train_ds, train_path)
            log(f"[OK] {dataset_name} / {display_task}: wrote {train_path} ({train_ds.num_rows} rows)")

        if test_parts:
            test_ds = test_parts[0] if len(test_parts) == 1 else concatenate_datasets(test_parts)
            test_ds = to_sft_schema(test_ds, entry, dataset_field, include_instruction)
            write_jsonl(test_ds, test_path)
            tag = " [from validation]" if used_fallback else ""
            log(f"[OK] {dataset_name} / {display_task}: wrote {test_path}{tag} ({test_ds.num_rows} rows)")
    except Exception:
        log(f"[ERROR] {dataset_name} / {display_task}: failed to build/write SFT rows\n{traceback.format_exc()}")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML_PATH = os.path.join(
    SCRIPT_DIR, "..", "dataefficiency", "prompts", "prompts_by_task_modified.yaml"
)


def main():
    parser = argparse.ArgumentParser(description="Export official HF train/test splits to JSONL")
    parser.add_argument("--yaml_path", type=str, default=DEFAULT_YAML_PATH)
    parser.add_argument("--output_dir", type=str, default="/net/spaces/scratch/lcpandia/data/processed")
    parser.add_argument("--only", type=str, nargs="*", default=None,
                         help="restrict to these top-level dataset names")
    parser.add_argument("--no_trust_remote_code", action="store_true",
                         help="never fall back to trust_remote_code=True, even if a dataset seems to need it")
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
                task_name=task_name,
                entry=entry,
                output_dir=args.output_dir,
                allow_remote_code=not args.no_trust_remote_code,
                include_instruction=not args.no_instruction_prefix,
                log=log,
            )


if __name__ == "__main__":
    main()
