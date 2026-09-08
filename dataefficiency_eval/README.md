# dataefficiency_eval

Standalone script that evaluates a single model across the full dataset
catalog used in `../dataefficiency`, instead of one (model, dataset) pair per
run like `dataefficiency/scripts/eval.py`.

It reuses `dataefficiency`'s model-family classes, prompt templates
(`prompts/prompts_by_task_modified.yaml`), and metric functions directly (as a
library, via `sys.path`) rather than duplicating or modifying them.

Supported model families: `llama`, `mistral`, `qwen` (from `dataefficiency/scripts/`),
and `olmo` (`olmo_child.py`, next to this script - OLMo-2-Instruct models
trained with the Tulu chat template). Family is auto-detected from
`--model_name` (matches "llama"/"mistral"/"qwen"/"olmo" as a substring) or set
explicitly via `--model_family`.

## Task catalog (40 tasks)

- 30 "core" tasks previously evaluated on Llama, Mistral, and Qwen.
- 10 additional tasks (`coqa`, `disaster_response_messages`, `disfl_qa`,
  `imbalanced`, `mmlu_pro`, `multitask_data_5`, `multitask_data_5_2`,
  `qa_wikidata`, `squad_v2`, `twitter_financial_news_sentiment`) that were
  previously only evaluated on Llama.

Run `python eval_all_datasets.py --list_tasks` to print the full list without
loading a model.

**Known gap:** `coqa.json`, `squad_v2.json`, `multitask_data_5.json`, and
`multitask_data_5_2.json` are referenced by the prompt YAML but are not
present in `dataefficiency/data.tar.gz` as shipped in this repo. Those 4
tasks will fail with a clear file-not-found error and are skipped rather than
crashing the run; every other task is unaffected.

## Usage

```bash
cd dataefficiency_eval/
python eval_all_datasets.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --use_flash_attention --use_safetensor
```

Useful flags:

- `--tasks anli,mmlu,boolq` - restrict to a subset of tasks (comma-separated
  names from `--list_tasks`).
- `--model_family {llama,mistral,qwen,olmo}` - override family auto-detection,
  needed if your checkpoint path/name doesn't contain the family name.
- `--calculate_all_metrics` - also compute BLEU/ROUGE/sequence-accuracy/F1 in
  addition to exact-string-match (slower).
- `--save_predictions` - dump per-example predictions to
  `results/predictions/`.
- Re-running with the same `--output_name` resumes: already-recorded tasks
  are kept and results are written incrementally after each task, so a run
  that dies partway through doesn't lose earlier progress (delete the
  relevant key from the JSON, or the whole file, to force a task to re-run).

Results are written to `results/<model_prefix>_<sanitized model name>_all_tasks.json`
as `{task_name: {exact_string_match_accuracy, num_examples, elapsed_seconds, ...}}`.

## Setup this script does automatically

Two of `BaseModel`'s path assumptions (in `dataefficiency/scripts/base_model.py`)
are worked around at runtime rather than edited in place:

1. `load_task_prompt()` reads `../prompts/prompts_by_task_modified.yaml`
   relative to the CWD. This script `chdir`s into `dataefficiency/scripts/`
   before doing anything else.
2. `get_data()` resolves local `data/*.json` datasets against
   `{Path.home()}/ft-intrinsic-dim/{dataset_name}`. This script extracts
   `dataefficiency/data.tar.gz` (if not already extracted) and monkeypatches
   `Path.home()` for its own process to point at `.cache/fake_home/`, which
   symlinks `ft-intrinsic-dim/data` to the extracted `dataefficiency/data/`
   folder. Nothing under your real `$HOME` is touched.
