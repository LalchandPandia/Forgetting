#!/bin/bash
#SBATCH --job-name=eval_job
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#SBATCH --partition=general
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --constraint='a100|h100|h200'

source ~/.bashrc
conda activate /net/projects/clab/lcpandia/forgetting

# Needed on this cluster -- NCCL segfaults on this node's IB setup otherwise.
# Harmless (and a no-op) for TENSOR_PARALLEL_SIZE=1.
export NCCL_IB_DISABLE=1

echo "hostname $(hostname)"
which python
nvidia-smi

cd /home/lcpandia/Forgetting/open-instruct

# ---- configure the job here ------------------------------------------------
# Runs all three tasks, both styles each (6 evals total, sequentially).
MODEL_NAME_OR_PATH="/path/to/your/model"
TENSOR_PARALLEL_SIZE=1
OUTPUT_DIR="/net/scratch/lcpandia/eval_outputs"

declare -A INPUT_JSONL=(
    [mmlu]="/home/lcpandia/Forgetting/open-instruct/data/eval/mmlu.jsonl"
    [ifeval]="/home/lcpandia/Forgetting/open-instruct/data/eval/ifeval.jsonl"
    [gsm8k]="/net/scratch/lcpandia/data/processed/100_gsm8k_test.jsonl"
)
# ifeval --style tulu needs the official HuggingFaceH4/ifeval schema
# (prompt/instruction_id_list/kwargs), not the messages/ground_truth schema
# used by --style retaining_by_doing above.
IFEVAL_TULU_INPUT_JSONL="/home/lcpandia/Forgetting/open-instruct/data/eval/ifeval_official.jsonl"

declare -A SCRIPT=(
    [mmlu]="open_instruct/mmlu_eval.py"
    [ifeval]="open_instruct/ifeval_eval.py"
    [gsm8k]="open_instruct/gsm8k_eval.py"
)

declare -A STYLES=(
    [mmlu]="retaining_by_doing tulu_mc"
    [ifeval]="retaining_by_doing tulu"
    [gsm8k]="retaining_by_doing tulu"
)

RETAINING_BY_DOING_PATH="/home/lcpandia/Forgetting/retaining-by-doing"   # for ifeval --style retaining_by_doing
OLMES_PATH="/home/lcpandia/olmes"                                       # for ifeval --style tulu
# -----------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"
model_short=$(basename "${MODEL_NAME_OR_PATH}")

for task in mmlu ifeval gsm8k; do
    for style in ${STYLES[$task]}; do
        output_jsonl="${OUTPUT_DIR}/${task}_${style}_${model_short}_${SLURM_JOB_ID}.jsonl"

        input_jsonl="${INPUT_JSONL[$task]}"
        extra_args=()
        if [[ "${task}" == "ifeval" ]]; then
            if [[ "${style}" == "retaining_by_doing" ]]; then
                extra_args+=(--retaining_by_doing_path "${RETAINING_BY_DOING_PATH}")
            else
                extra_args+=(--olmes_path "${OLMES_PATH}")
                input_jsonl="${IFEVAL_TULU_INPUT_JSONL}"
            fi
        fi

        echo "=========================================="
        echo "TASK=${task} STYLE=${style}"
        echo "SCRIPT=${SCRIPT[$task]}"
        echo "INPUT_JSONL=${input_jsonl}"
        echo "OUTPUT_JSONL=${output_jsonl}"
        echo "=========================================="

        python "${SCRIPT[$task]}" \
            --input_jsonl "${input_jsonl}" \
            --style "${style}" \
            --model_name_or_path "${MODEL_NAME_OR_PATH}" \
            --output_jsonl "${output_jsonl}" \
            --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
            "${extra_args[@]}"
    done
done
