#!/bin/bash
#SBATCH --job-name=gsm8k_train_format
#SBATCH --output=gsm8k_train_format_%j.out
#SBATCH --error=gsm8k_train_format_%j.err
#SBATCH --partition=general
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --constraint='a100|h100|h200'

source ~/.bashrc
conda activate /net/projects/clab/lcpandia/forgetting

# Needed on this cluster -- NCCL segfaults on this node's IB setup otherwise.
export NCCL_IB_DISABLE=1

echo "hostname $(hostname)"
which python
nvidia-smi

cd /home/lcpandia/Forgetting/open-instruct

# ---- configure the job here ------------------------------------------------
# Runs retaining_by_doing (mismatched prompt) side by side with train_format
# (exact training-format prompt) across the full fine-tuning chain, on the
# full 1,319-row official GSM8K test set -- so you can see, per checkpoint,
# whether closing the prompt-format gap recovers accuracy.
OUTPUT_DIR="/net/scratch/lcpandia/eval_outputs"
INPUT_JSONL="data/eval/gsm8k_official_test.jsonl"

declare -A CKPTS=(
    [base]="Qwen/Qwen2.5-1.5B-Instruct"
    [ifeval]="/net/scratch/lcpandia/forgetting/ifeval_base_1.5B_full/finetune"
    [mmlu]="/net/scratch/lcpandia/forgetting/mmlu_ifeval_tuned_1.5B_full/finetune"
    [gsm8k]="/net/scratch/lcpandia/forgetting/gsm8k_mmlu_ifeval_tuned_1.5B_full/finetune"
)
CKPT_ORDER=(base ifeval mmlu gsm8k)
STYLES=(retaining_by_doing train_format)
# -----------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"

for name in "${CKPT_ORDER[@]}"; do
    model="${CKPTS[$name]}"
    for style in "${STYLES[@]}"; do
        output_jsonl="${OUTPUT_DIR}/gsm8k_${style}_official_${name}_${SLURM_JOB_ID}.jsonl"

        echo "=========================================="
        echo "CKPT=${name} STYLE=${style}"
        echo "MODEL=${model}"
        echo "INPUT_JSONL=${INPUT_JSONL}"
        echo "OUTPUT_JSONL=${output_jsonl}"
        echo "=========================================="

        python open_instruct/gsm8k_eval.py \
            --input_jsonl "${INPUT_JSONL}" \
            --style "${style}" \
            --model_name_or_path "${model}" \
            --output_jsonl "${output_jsonl}" \
            --tensor_parallel_size 1
    done
done

echo "done -- see ${OUTPUT_DIR}/gsm8k_*_official_*_${SLURM_JOB_ID}.jsonl and *_metrics.json"
