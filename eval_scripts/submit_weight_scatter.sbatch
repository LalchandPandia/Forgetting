#!/bin/bash
#SBATCH --job-name=weight_scatter
#SBATCH --output=weight_scatter_%j.out
#SBATCH --error=weight_scatter_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# No GPU requested -- weight_scatter.py only loads state dicts and diffs
# tensors on CPU (device_map={"": "cpu"} in load_state_dict()). If this
# cluster's "general" partition is GPU-only, switch to whatever CPU
# partition `sinfo` shows, or add back --gpus-per-node=1 --constraint=... as
# a formality (it just won't be used).

source ~/.bashrc
conda activate /net/projects/clab/lcpandia/forgetting

echo "hostname $(hostname)"
which python

cd /home/lcpandia/Forgetting/open-instruct

# ---- configure the job here ------------------------------------------------
OUT_DIR="plots/weight_scatter"

python open_instruct/weight_scatter.py \
    --ckpt base=Qwen/Qwen2.5-1.5B-Instruct \
    --ckpt ifeval=/net/scratch/lcpandia/forgetting/ifeval_base_1.5B_full/finetune \
    --ckpt mmlu=/net/scratch/lcpandia/forgetting/mmlu_ifeval_tuned_1.5B_full/finetune \
    --ckpt gsm8k=/net/scratch/lcpandia/forgetting/gsm8k_mmlu_ifeval_tuned_1.5B_full/finetune \
    --chain base,ifeval,mmlu,gsm8k \
    --highlight mmlu,gsm8k \
    --topk 8 \
    --sample 20000 \
    --out_dir "${OUT_DIR}"
# -----------------------------------------------------------------------------

echo "done -- see ${OUT_DIR}/"
