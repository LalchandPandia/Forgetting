#!/bin/bash
#SBATCH --job-name=new_job_context_sbatch
#SBATCH --output=example_sbatch_subset_forgetting_qwen_full.out
#SBATCH --error=example_sbatch_subset_forgetting_qwen_full.err
#SBATCH --partition=general
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --constraint='a100|h100|h200'

# Activate conda
echo "hostname $(hostname)"
which python
nvcc --version
nvidia-smi

echo $PATH
echo "hostname $(hostname)"
dataset=ifeval
model_name="qwen2.5"

MODEL_SIZE=1.5B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training ${model_name} model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
model='Qwen/Qwen2.5-1.5B-Instruct'

export MASTER_PORT=$(shuf -i 29500-29999 -n 1)
#added to avoid NCCL from using RDMA/IB interface
export NCCL_IB_DISABLE=1
train_file=data/sft/ifeval.jsonl
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes ${NUM_GPUS} \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path ${model} \
    --tokenizer_name ${model} \
    --use_slow_tokenizer \
    --dataset_mixer_list ${train_file} 1.0 \
    --dataset_mixer_list_splits train \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps 64 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir /net/scratch/lcpandia/forgetting/${dataset}_${model_name}_${MODEL_SIZE}_full/ \
    --with_tracking \
    --report_to wandb \
    --logging_steps 5