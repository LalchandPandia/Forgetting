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

cd /home/lalchand/Forgetting/eval_scripts

# ---- configure the job here ------------------------------------------------
TASK="mmlu"                        # mmlu | ifeval
STYLE="retaining_by_doing"         # mmlu:   retaining_by_doing | tulu_mc
                                    # ifeval: retaining_by_doing | tulu
MODEL_NAME_OR_PATH="/path/to/your/model"
INPUT_JSONL="/path/to/your/input.jsonl"
TENSOR_PARALLEL_SIZE=1

RETAINING_BY_DOING_PATH="/home/lalchand/Forgetting/retaining-by-doing"   # for ifeval --style retaining_by_doing
OLMES_PATH="/home/lalchand/olmes"                                       # for ifeval --style tulu

OUTPUT_DIR="/net/scratch/lcpandia/eval_outputs"
# -----------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"
model_short=$(basename "${MODEL_NAME_OR_PATH}")
OUTPUT_JSONL="${OUTPUT_DIR}/${TASK}_${STYLE}_${model_short}_${SLURM_JOB_ID}.jsonl"

echo "TASK=${TASK}"
echo "STYLE=${STYLE}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "INPUT_JSONL=${INPUT_JSONL}"
echo "OUTPUT_JSONL=${OUTPUT_JSONL}"

if [[ "${TASK}" == "mmlu" ]]; then
    python mmlu_eval.py \
        --input_jsonl "${INPUT_JSONL}" \
        --style "${STYLE}" \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --output_jsonl "${OUTPUT_JSONL}" \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"

elif [[ "${TASK}" == "ifeval" ]]; then
    extra_args=()
    if [[ "${STYLE}" == "retaining_by_doing" ]]; then
        extra_args+=(--retaining_by_doing_path "${RETAINING_BY_DOING_PATH}")
    elif [[ "${STYLE}" == "tulu" ]]; then
        extra_args+=(--olmes_path "${OLMES_PATH}")
    fi
    python ifeval_eval.py \
        --input_jsonl "${INPUT_JSONL}" \
        --style "${STYLE}" \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --output_jsonl "${OUTPUT_JSONL}" \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
        "${extra_args[@]}"

else
    echo "Unknown TASK=${TASK} (expected mmlu or ifeval)"
    exit 1
fi
