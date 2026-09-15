#!/bin/bash
#SBATCH --job-name=medcalc_qwen_eval
#SBATCH --output=medcalc_qwen_eval_%j.out
#SBATCH --error=medcalc_qwen_eval_%j.err
#SBATCH --partition=general
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --constraint='a100|h100|h200'

source ~/.bashrc
conda activate /net/projects/clab/lcpandia/forgetting
export NCCL_IB_DISABLE=1

echo "hostname $(hostname)"
which python
nvidia-smi

cd /home/lcpandia/Forgetting/utils

# ---- configure the job here ------------------------------------------------
# Zero-shot eval (no fine-tuning): swap MODEL for a fine-tuned checkpoint path
# to instead measure a trained model's MedCalc performance.
MODEL="Qwen/Qwen2.5-7B-Instruct"
TEST_FILE="/net/spaces/scratch/lcpandia/data/processed/medcalc/MedCalc/test.jsonl"
OUTPUT_DIR="/net/scratch/lcpandia/eval_outputs"
MAX_NEW_TOKENS=2048
LIMIT=""   # e.g. LIMIT=50 for a quick debug run; leave empty to evaluate all rows
# -----------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"
model_short=$(basename "${MODEL}")
output_json="${OUTPUT_DIR}/medcalc_zero_shot_${model_short}_${SLURM_JOB_ID}.json"

extra_args=()
if [[ -n "${LIMIT}" ]]; then
    extra_args+=(--limit "${LIMIT}")
fi

echo "=========================================="
echo "MODEL=${MODEL}"
echo "TEST_FILE=${TEST_FILE}"
echo "OUTPUT_JSON=${output_json}"
echo "=========================================="

python eval_medcalc_qwen.py \
    --model "${MODEL}" \
    --test_file "${TEST_FILE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --out "${output_json}" \
    "${extra_args[@]}"

echo "done -- see ${output_json}"
