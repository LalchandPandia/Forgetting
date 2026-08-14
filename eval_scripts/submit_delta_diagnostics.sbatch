#!/bin/bash
#SBATCH --job-name=delta_diagnostics
#SBATCH --output=delta_diagnostics_%j.out
#SBATCH --error=delta_diagnostics_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# CPU-only, same as submit_weight_scatter.sbatch -- delta_diagnostics.py just
# loads safetensors and does SVD/numpy math, no CUDA involved.

source ~/.bashrc
conda activate /net/projects/clab/lcpandia/forgetting

echo "hostname $(hostname)"
which python

cd /home/lcpandia/Forgetting/open-instruct

# delta_diagnostics.py's load_state_dict() globs *.safetensors under a LOCAL
# directory -- it can't resolve a HF Hub id like weight_scatter.py can. Since
# weight_scatter.py already downloaded+cached the base model, this just
# returns that cached snapshot path (only re-downloads if it's missing).
BASE=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct'))")
IFEVAL=/net/scratch/lcpandia/forgetting/ifeval_base_1.5B_full/finetune
MMLU=/net/scratch/lcpandia/forgetting/mmlu_ifeval_tuned_1.5B_full/finetune
GSM8K=/net/scratch/lcpandia/forgetting/gsm8k_mmlu_ifeval_tuned_1.5B_full/finetune

echo "BASE resolved to: ${BASE}"

# delta_diagnostics.py only takes 3 checkpoints (base mid end) at a time, and
# writes FIXED filenames (delta_diagnostics.csv, layer_profile.png,
# spectra.png) into the current directory -- no --out_dir. Run each triple
# from its own subdirectory so the two invocations don't clobber each other.

mkdir -p plots/delta_diagnostics/base_ifeval_mmlu
( cd plots/delta_diagnostics/base_ifeval_mmlu && \
  python /home/lcpandia/Forgetting/open-instruct/open_instruct/delta_diagnostics.py "${BASE}" "${IFEVAL}" "${MMLU}" )

mkdir -p plots/delta_diagnostics/ifeval_mmlu_gsm8k
( cd plots/delta_diagnostics/ifeval_mmlu_gsm8k && \
  python /home/lcpandia/Forgetting/open-instruct/open_instruct/delta_diagnostics.py "${IFEVAL}" "${MMLU}" "${GSM8K}" )

echo "done -- see plots/delta_diagnostics/base_ifeval_mmlu/ and .../ifeval_mmlu_gsm8k/"
