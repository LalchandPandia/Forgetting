#!/bin/bash
#SBATCH --job-name=rl_grpo
#SBATCH --output=rl_grpo_%j.out
#SBATCH --error=rl_grpo_%j.err
#SBATCH --partition=general
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --constraint='a100|h100|h200'



# Needed on this cluster -- NCCL segfaults on this node's IB setup otherwise.
export NCCL_IB_DISABLE=1
# Works around a conda numpy/mkl-service vs. libgomp conflict that crashes vLLM.
export MKL_THREADING_LAYER=GNU
# rl.py's weight sync needs vLLM's engine in the actor process; OnPolicyActor also sets this itself.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

echo "hostname $(hostname)"
which python
nvidia-smi



run_base_dir="/net/spaces/scratch/lcpandia/forgetting/rl_runs"
mkdir -p "${run_base_dir}"

num_gpus=4
num_gen_gpus=2

export OMP_NUM_THREADS=${num_gpus}
backend="nccl"
master_addr="127.0.0.1"
master_port=$(shuf -i 29500-29999 -n 1)

# Model ########################################################################
#model_name_short="llama-3.2-1b-inst"
#model_name_short="llama-3.1-8b-inst"
#model_name_short="qwen-2.5-1.5b-inst"
model_name_short="qwen-2.5-7b-inst"
model_name="Qwen/Qwen2.5-7B-Instruct"
###############################################################################

# Dataset ######################################################################
dataset_name="GSM8KDataset"
#dataset_name="IFEvalDataset"
#dataset_name="MMLUDataset"
#dataset_name="CountdownDataset"
###############################################################################

seed=0
wandb_project="forgetting-rl"  # "none" disables wandb

ref_model_name="${model_name}"
reward_model_name="none"

# Main params #################################################################
max_new_tokens=3000
lr=5e-6
num_epochs=2

num_train_datapoints="all"

temperature=0.8
kl_beta=0.05
num_generation_per_prompt=5

gradient_accumulation_steps=16
train_batch_size_per_device=1
eval_batch_size_per_device=2

max_seq_length=2000
show_gpu_usage=false
###############################################################################

data_module_path="core.data"
model_module_path="transformers.AutoModelForCausalLM"
loss_module_path="core.training.objectives.GRPOLoss"

if [[ "${model_name_short}" == *"llama"* ]]; then
    transformer_layer_module_path="transformers.models.llama.modeling_llama.LlamaDecoderLayer"
elif [[ "${model_name_short}" == *"gemma"* ]]; then
    transformer_layer_module_path="transformers.models.gemma2.modeling_gemma2.Gemma2DecoderLayer"
elif [[ "${model_name_short}" == *"qwen"* ]]; then
    transformer_layer_module_path="transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer"
fi

num_eval_datapoints="all"
# Greedy pass@1 on the eval split via vLLM before training and every N optimizer steps; 0 disables.
eval_generation_every_n_steps=20

warmup_ratio=0.0
scheduler_type="constant"
gradient_clipping=1.0

eval_every_n_steps=100000000
print_every_n_steps=1000000000
save_every_n_steps=20

total_batch_size=$((train_batch_size_per_device * (num_gpus - num_gen_gpus) * gradient_accumulation_steps))

run_name="algo=rl_data=${dataset_name}_m=${model_name_short}_n=${num_train_datapoints}_epoch=${num_epochs}_bsz=${total_batch_size}_lr=${lr}_temp=${temperature}_klbeta=${kl_beta}_mnt=${max_new_tokens}_seed=${seed}"
###############################################################################

command="python -m core.training.rl"
args=(
    --backend ${backend} \
    --master_addr ${master_addr} \
    --master_port ${master_port} \
    --run_name ${run_name} \
    --run_base_dir ${run_base_dir} \
    --seed ${seed} \
    --model_name ${model_name} \
    --ref_model_name ${ref_model_name} \
    --reward_model_name ${reward_model_name} \
    --max_new_tokens ${max_new_tokens} \
    --temperature ${temperature} \
    --num_generation_per_prompt ${num_generation_per_prompt} \
    --kl_beta ${kl_beta} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --lr ${lr} \
    --num_epochs ${num_epochs} \
    --warmup_ratio ${warmup_ratio} \
    --scheduler_type ${scheduler_type} \
    --gradient_clipping ${gradient_clipping} \
    --eval_every_n_steps ${eval_every_n_steps} \
    --eval_generation_every_n_steps ${eval_generation_every_n_steps} \
    --save_every_n_steps ${save_every_n_steps} \
    --print_every_n_steps ${print_every_n_steps} \
    --model_module_path ${model_module_path} \
    --transformer_layer_module_path ${transformer_layer_module_path} \
    --loss_module_path ${loss_module_path} \
    --data_module_path ${data_module_path} \
    --dataset_name ${dataset_name} \
    --num_train_datapoints ${num_train_datapoints} \
    --num_eval_datapoints ${num_eval_datapoints} \
    --train_batch_size_per_device ${train_batch_size_per_device} \
    --eval_batch_size_per_device ${eval_batch_size_per_device} \
    --total_batch_size ${total_batch_size} \
    --max_seq_length ${max_seq_length} \
    --num_gpus ${num_gpus} \
    --num_gen_gpus ${num_gen_gpus} \
    --show_gpu_usage ${show_gpu_usage} \
    --wandb_project ${wandb_project}
)

${command} "${args[@]}"
