#!/bin/bash
#SBATCH --job-name=eval_esci
#SBATCH --output=eval_esci_%j.out
#SBATCH --error=eval_esci_%j.err
#SBATCH --partition=general
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --constraint='a100|h100|h200'


source ~/.bashrc
conda activate /net/projects/clab/lcpandia/vllm_22
export NCCL_IB_DISABLE=1

# Works around a conda numpy/mkl-service vs. libgomp conflict that otherwise
# crashes vLLM's model-registry inspection subprocess with:
#   "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library"
export MKL_THREADING_LAYER=GNU

echo "hostname $(hostname)"
which python
nvidia-smi

cd /home/lcpandia/Forgetting/open-instruct

# ---- configure the job here ------------------------------------------------
# Zero-shot eval (no fine-tuning): swap MODEL for a fine-tuned checkpoint path
# to instead measure a trained model's ESCI performance.
MODEL="Qwen/Qwen2.5-7B-Instruct"
TEST_FILE="/net/spaces/scratch/lcpandia/data/processed/esci/test_esci.jsonl"
OUTPUT_DIR="/net/scratch/lcpandia/eval_outputs"
MAX_NEW_TOKENS=8
PRINT_N=2   # print the exact rendered prompt + raw generation for the first N examples
LIMIT=""    # e.g. LIMIT=50 for a quick debug run; leave empty to evaluate all rows
# -----------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"
model_short=$(basename "${MODEL}")
output_json="${OUTPUT_DIR}/esci_eval_${model_short}_${SLURM_JOB_ID}.json"

extra_args=()
if [[ -n "${LIMIT}" ]]; then
    extra_args+=(--limit "${LIMIT}")
fi

echo "=========================================="
echo "MODEL=${MODEL}"
echo "TEST_FILE=${TEST_FILE}"
echo "OUTPUT_JSON=${output_json}"
echo "=========================================="

python eval_ESCI.py \
    --model "${MODEL}" \
    --test_file "${TEST_FILE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --print_n "${PRINT_N}" \
    --out "${output_json}" \
    "${extra_args[@]}"

echo "done -- see ${output_json}"
