#!/bin/bash
#SBATCH --job-name=inspect_gen
#SBATCH --output=inspect_gen_%j.out
#SBATCH --error=inspect_gen_%j.err
#SBATCH --partition=general
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --constraint='a100|h100|h200'
# Unlike weight_scatter.py / delta_diagnostics.py, this one actually calls
# model.generate() with device_map="auto" -- it wants a GPU.

source ~/.bashrc
conda activate /net/projects/clab/lcpandia/forgetting

echo "hostname $(hostname)"
which python
nvidia-smi

cd /home/lcpandia/Forgetting/open-instruct

# fixed output filenames (generations.jsonl / generations.txt), no --out_dir --
# run from a dedicated directory so repeated runs don't clobber each other.
OUT_DIR="plots/inspect_generations/${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

python /home/lcpandia/Forgetting/open-instruct/inspect_generations.py \
    --ckpt base=Qwen/Qwen2.5-1.5B-Instruct \
    --ckpt mmlu=/net/scratch/lcpandia/forgetting/mmlu_ifeval_tuned_1.5B_full/finetune \
    --ckpt gsm8k=/net/scratch/lcpandia/forgetting/gsm8k_mmlu_ifeval_tuned_1.5B_full/finetune \
    --n 20

echo "done -- see ${OUT_DIR}/generations.txt"
