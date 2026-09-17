#!/bin/bash
#SBATCH --job-name=eval_anli
#SBATCH --output=eval_anli_%j.out
#SBATCH --error=eval_anli_%j.err
#SBATCH --partition=general
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --constraint='a100|h100|h200'


source ~/.bashrc
export NCCL_IB_DISABLE=1

# Works around a conda numpy/mkl-service vs. libgomp conflict that otherwise
# crashes vLLM's model-registry inspection subprocess with:
#   "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library"
export MKL_THREADING_LAYER=GNU

echo "hostname $(hostname)"
which python
nvidia-smi

cd /home/lcpandia/Forgetting/dataefficiency_eval

# ---- configure the job here ------------------------------------------------
# Default: genuine base model, plain-text completion (--raw), since
# allenai/OLMo-2-1124-7B's tokenizer has no chat_template. For an instruct
# model instead (e.g. Qwen/Qwen2.5-7B-Instruct), set MODEL accordingly and
# change RAW to "" so the chat-template path is used.
MODEL="Qwen/Qwen2.5-7B-Instruct"
RAW=""   # base model: "--raw"; instruct model (chat template): ""
DATA="/net/spaces/scratch/lcpandia/data/processed/facebook/anli/r1/test_anli.jsonl"
OUTPUT_DIR="/net/spaces/scratch/lcpandia/eval_outputs"
MAX_NEW_TOKENS=8
LENIENT=""   # set LENIENT="--lenient" to accept a label as the first token of a longer completion
LIMIT=""     # e.g. LIMIT=50 for a quick debug run; leave empty to evaluate all rows
# -----------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"
model_short=$(basename "${MODEL}")
output_json="${OUTPUT_DIR}/anli_eval_${model_short}_${SLURM_JOB_ID}.json"

extra_args=()
[[ -n "${RAW}" ]] && extra_args+=("${RAW}")
[[ -n "${LENIENT}" ]] && extra_args+=("${LENIENT}")
if [[ -n "${LIMIT}" ]]; then
    extra_args+=(--limit "${LIMIT}")
fi

echo "=========================================="
echo "MODEL=${MODEL}"
echo "DATA=${DATA}"
echo "RAW=${RAW:-<chat-template mode>}"
echo "OUTPUT_JSON=${output_json}"
echo "=========================================="

python eval_anli.py \
    --model "${MODEL}" \
    --data "${DATA}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --print_n 3 \
    --out "${output_json}" \
    "${extra_args[@]}"

echo "done -- see ${output_json}"
