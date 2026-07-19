#!/bin/bash
#SBATCH --job-name=new_job_context_sbatch
#SBATCH --output=example_sbatch_subset_confidence_olmo2_7B_5.out
#SBATCH --error=example_sbatch_subset_confidence_olmo2_7B_5.err
#SBATCH --partition=general
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --constraint='a100|h100|h200'

# Activate conda
source ~/.bashrc
echo "hostname $(hostname)"
which python
nvcc --version
nvidia-smi

echo $PATH
echo "hostname $(hostname)"
MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
dataset=MedQA
dataset=tulu3
percentage=25
model=OLMo_2_7B
metric=random_1456784730071470512

metric=vog_score_olmo2_7B
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.
#added nsys profile to get cuda level info
train_file=/net/scratch/lcpandia/data/processed/llama3/${dataset}/${metric}/${percentage}_${metric}_${dataset}.json
train_file=/net/scratch/lcpandia/activeSFT/training_data/training_data/random_one_shot_olmo_2_7B/train.jsonl
train_file=/net/scratch/lcpandia/data/processed/tulu3/random_subset_50K.jsonl
train_file=/net/scratch/lcpandia/data/processed/llama3/${dataset}/CCS/${metric}/${percentage}_${metric}_${dataset}.json
#train_file=/net/scratch/lcpandia/data/processed/100_MedQA_train.json
#train_file=/net/scratch/lcpandia/data/processed/chempile_reasoning.jsonl
#train_file=/net/scratch/lcpandia/data/processed/chempile_instruction_reasoning_education.jsonl
#validation_file=/net/scratch/lcpandia/data/processed/chempile_reasoning.jsonl
model_name='facebook/opt-6.7b'
model_name='allenai/OLMo-2-1124-7B'
#model_name='allenai/OLMo-2-1124-13B'
#model_name='meta-llama/Meta-Llama-3-8B'
#model_name='meta-llama/Llama-3.2-1B'
#model_name='allenai/OLMo-2-0425-1B'
model_name='Qwen-2.5-1.5B-Instruct'
export MASTER_PORT=$(shuf -i 29500-29999 -n 1)
#export WANDB_MODE=offline
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --main_process_port $MASTER_PORT \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/fineTune_lora_olmo.py \
    --model_name_or_path  ${model_name} \
    --use_flash_attn \
    --tokenizer_name ${model_name} \
    --use_slow_tokenizer \
    --dataset ${dataset} \
    --train_file ${train_file} \
    --validation_file ${train_file} \
    --max_seq_length 4096 \
    --reduce_loss 'sum' \
    --keep_last_n_checkpoints -1 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --timeout 360000 \
    --metric ${metric} \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 3 \
    --checkpointing_steps 32 \
    --output_dir /net/scratch/lcpandia/outputs/${dataset}_${percentage}_${model}_${metric}/ \
    --with_tracking True\
    --report_to wandb \
    --logging_steps 5 --push_to_hub False
