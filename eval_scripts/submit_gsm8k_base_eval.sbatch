#!/bin/bash
#SBATCH --job-name=gsm8k_base_eval
#SBATCH --output=gsm8k_base_eval_%j.out
#SBATCH --error=gsm8k_base_eval_%j.err
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

cd /home/lcpandia/Forgetting/open-instruct

# ---- configure the job here ------------------------------------------------
# NOTE: this is the genuine BASE model (no chat template) -- NOT an -Instruct
# checkpoint. --style base is specifically for a model like this one, without
# a chat template. NUM_SHOTS=0 gives a zero-shot read of the base model's own
# GSM8K ability (flat text completion, no chat template, no few-shot context)
# rather than borrowing the instruct-only "tulu" chat-templated methodology,
# which a genuine base checkpoint was never tuned to expect.
MODEL="allenai/OLMo-2-1124-7B"
OUTPUT_DIR="/net/scratch/lcpandia/eval_outputs"
INPUT_JSONL="data/eval/gsm8k_official_test.jsonl"
NUM_SHOTS=0
# -----------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"
output_jsonl="${OUTPUT_DIR}/gsm8k_base_${NUM_SHOTS}shot_official_olmo7b_${SLURM_JOB_ID}.jsonl"

echo "=========================================="
echo "STYLE=base, NUM_SHOTS=${NUM_SHOTS} (no chat template)"
echo "MODEL=${MODEL}"
echo "INPUT_JSONL=${INPUT_JSONL}"
echo "OUTPUT_JSONL=${output_jsonl}"
echo "=========================================="

python open_instruct/gsm8k_eval.py \
    --input_jsonl "${INPUT_JSONL}" \
    --style base \
    --num_shots "${NUM_SHOTS}" \
    --model_name_or_path "${MODEL}" \
    --output_jsonl "${output_jsonl}" \
    --tensor_parallel_size 1 \
    --print_n 3

echo "done -- see ${output_jsonl} and its _metrics.json"
