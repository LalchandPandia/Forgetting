#!/bin/bash
#SBATCH --job-name=eval_all_datasets
#SBATCH --output=eval_all_datasets_%j.out
#SBATCH --error=eval_all_datasets_%j.err
#SBATCH --partition=general
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --constraint='a100|h100|h200'

source ~/.bashrc
conda activate /net/projects/clab/lcpandia/forgetting
# NOTE: this env must also have dataefficiency's deps installed
# (transformers, datasets, peft, trl, accelerate, pynvml) -- see
# dataefficiency/pyproject.toml. If it doesn't, either install them here or
# point this at a different env (e.g. one built from dataefficiency/uv.lock).

export NCCL_IB_DISABLE=1

echo "hostname $(hostname)"
which python
nvidia-smi

cd /home/lcpandia/Forgetting/dataefficiency_eval

# ---- configure the job here ------------------------------------------------
MODEL_NAME="/path/to/your/model"
MODEL_FAMILY=""          # override if the checkpoint path doesn't contain
                          # "llama"/"mistral"/"qwen", e.g. MODEL_FAMILY="llama"
TASKS=""                 # comma-separated task names to restrict to, e.g.
                          # TASKS="anli,mmlu,boolq" -- empty runs all 40
BATCH_SIZE=8
MAX_SEQ_LEN=2048
RESULT_DIR="/net/scratch/lcpandia/eval_outputs/dataefficiency"
# -----------------------------------------------------------------------------

mkdir -p "${RESULT_DIR}"

extra_args=()
if [[ -n "${MODEL_FAMILY}" ]]; then
    extra_args+=(--model_family "${MODEL_FAMILY}")
fi
if [[ -n "${TASKS}" ]]; then
    extra_args+=(--tasks "${TASKS}")
fi

echo "=========================================="
echo "MODEL_NAME=${MODEL_NAME}"
echo "TASKS=${TASKS:-all 40}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "=========================================="

python eval_all_datasets.py \
    --model_name "${MODEL_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --result_dir "${RESULT_DIR}" \
    --use_flash_attention \
    --use_safetensor \
    "${extra_args[@]}"

echo "done -- see ${RESULT_DIR}/*_all_tasks.json"
