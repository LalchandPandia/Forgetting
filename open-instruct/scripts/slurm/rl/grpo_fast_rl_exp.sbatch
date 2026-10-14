#!/bin/bash
#SBATCH --job-name=grpo_fast_rl_exp
#SBATCH --output=grpo_fast_rl_exp_%j.out
#SBATCH --error=grpo_fast_rl_exp_%j.err
#SBATCH --partition=general
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --constraint='a100|h100|h200'

# open-instruct's grpo_fast.py configured to mirror RL_exp/scripts/rl.sbatch (retaining-by-doing GRPO).
#
# Usage (submit from anywhere):
#   sbatch scripts/slurm/rl/grpo_fast_rl_exp.sbatch                               # MedCalc
#   sbatch --export=ALL,DATASET=gsm8k scripts/slurm/rl/grpo_fast_rl_exp.sbatch    # GSM8K
#
# Prerequisite: convert the data once (writes ${DATA_DIR}/{gsm8k,medcalc}/{train,test}.jsonl):
#   python scripts/data/convert_rl_exp_to_rlvr.py --output_dir /net/spaces/scratch/lcpandia/data/processed/rlvr

set -eo pipefail

# TODO: env matching open-instruct's pyproject (vllm>=0.19.1 with vllm.distributed.weight_transfer, ray, transformers>=5.4).
OI_ENV="${OI_ENV:-/net/projects2/clab/lcpandia/forgetting}"
source ~/.bashrc
conda activate "${OI_ENV}"

# Needed on this cluster -- NCCL segfaults on this node's IB setup otherwise.
export NCCL_IB_DISABLE=1
# Works around a conda numpy/mkl-service vs. libgomp conflict that crashes vLLM.
export MKL_THREADING_LAYER=GNU
# Flush output immediately so logs show progress live and survive a time-limit kill.
export PYTHONUNBUFFERED=1

echo "hostname $(hostname)"
which python
nvidia-smi

# Fail fast if the env can't run grpo_fast.py (its vLLM weight sync needs vllm.distributed.weight_transfer).
python -c "import vllm, ray, transformers, deepspeed; from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine; print('vllm', vllm.__version__, '| ray', ray.__version__, '| transformers', transformers.__version__, '| deepspeed', deepspeed.__version__)"

cd /home/lcpandia/Forgetting/open-instruct

# ---- configure the job here --------------------------------------------------------------
DATASET="${DATASET:-medcalc}"   # medcalc | gsm8k
MODEL="Qwen/Qwen2.5-7B-Instruct"
DATA_DIR="/net/spaces/scratch/lcpandia/data/processed/rlvr"
OUTPUT_BASE="/net/spaces/scratch/lcpandia/forgetting/grpo_fast_runs"

LR=1e-6
TEMPERATURE=0.8
KL_BETA=0.05
NUM_PROMPTS_PER_STEP=32          # RL_exp: 1 prompt/GPU x 2 trainers x 16 grad-accum steps
NUM_SAMPLES_PER_PROMPT=5
NUM_EPOCHS_OVER_DATA=2
RESPONSE_LENGTH=3000             # RL_exp max_new_tokens
MAX_PROMPT_TOKENS=3072           # longest MedCalc prompt (incl. prefill) is ~2,880 tokens
EVAL_EVERY=20
SAVE_EVERY=20
DEEPSPEED_STAGE=3                # 7B on 2 learner GPUs; stage 2 should fit on H200s
SEED=0
# -------------------------------------------------------------------------------------------

TRAIN_FILE="${DATA_DIR}/${DATASET}/train.jsonl"
TEST_FILE="${DATA_DIR}/${DATASET}/test.jsonl"
for f in "${TRAIN_FILE}" "${TEST_FILE}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: ${f} not found; run scripts/data/convert_rl_exp_to_rlvr.py first" >&2
        exit 1
    fi
done

# grpo_fast trains for total_episodes // (prompts_per_step * samples_per_prompt) steps.
NUM_TRAIN_PROMPTS=$(wc -l < "${TRAIN_FILE}")
TOTAL_EPISODES=$((NUM_TRAIN_PROMPTS * NUM_SAMPLES_PER_PROMPT * NUM_EPOCHS_OVER_DATA))
PACK_LENGTH=$((MAX_PROMPT_TOKENS + RESPONSE_LENGTH + 72))   # must fit one prompt + response

RUN_NAME="grpo_fast_rl_exp_${DATASET}_$(basename ${MODEL})_lr${LR}_temp${TEMPERATURE}_beta${KL_BETA}_n${NUM_SAMPLES_PER_PROMPT}_rl${RESPONSE_LENGTH}_seed${SEED}"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo "DATASET=${DATASET} | train prompts=${NUM_TRAIN_PROMPTS} | total_episodes=${TOTAL_EPISODES} | output_dir=${OUTPUT_DIR}"

python open_instruct/grpo_fast.py \
    --exp_name "grpo_fast_rl_exp_${DATASET}" \
    --run_name "${RUN_NAME}" \
    --seed ${SEED} \
    --model_name_or_path "${MODEL}" \
    --dataset_mixer_list "${TRAIN_FILE}" 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list "${TEST_FILE}" 1.0 \
    --dataset_mixer_eval_list_splits train \
    --dataset_transform_fn rlvr_tokenize_prefill_v1 rlvr_max_length_filter_v1 \
    --max_prompt_token_length ${MAX_PROMPT_TOKENS} \
    --response_length ${RESPONSE_LENGTH} \
    --eval_response_length ${RESPONSE_LENGTH} \
    --pack_length ${PACK_LENGTH} \
    --num_unique_prompts_rollout ${NUM_PROMPTS_PER_STEP} \
    --num_samples_per_prompt_rollout ${NUM_SAMPLES_PER_PROMPT} \
    --total_episodes ${TOTAL_EPISODES} \
    --temperature ${TEMPERATURE} \
    --learning_rate ${LR} \
    --lr_scheduler_type constant \
    --warmup_ratio 0.0 \
    --max_grad_norm 1.0 \
    --beta ${KL_BETA} \
    --kl_estimator 1 \
    --advantage_normalization_type standard \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --non_stop_penalty true \
    --non_stop_penalty_value 0.0 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --async_steps 1 \
    --inflight_updates false \
    --per_device_train_batch_size 1 \
    --num_learners_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --deepspeed_stage ${DEEPSPEED_STAGE} \
    --local_eval_every ${EVAL_EVERY} \
    --eval_on_step_0 true \
    --save_freq ${SAVE_EVERY} \
    --checkpoint_state_freq ${SAVE_EVERY} \
    --checkpoint_state_dir "${OUTPUT_DIR}/checkpoint_state" \
    --keep_last_n_checkpoints 2 \
    --output_dir "${OUTPUT_DIR}" \
    --with_tracking \
    --wandb_project open_instruct_internal \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs_on_weka false
