"""
Based on Ray and FSDP.
"""
import os
import re
import sys
import copy
import time
import yaml
import time
import math
import shutil
import random
import signal
import functools
from pathlib import Path
import fire
import wandb

import numpy as np
import ray
import vllm
from vllm import LLM, SamplingParams
import transformers
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
)
from core.training.trainer import DistributedBaseTrainer
from core.utils import load_module, ExperimentManager, print_rank0
from core.data import tokenizer_factory

os.environ['RAY_DEDUP_LOGS'] = '0'


def print_gpu_usage(num_total_gpus):
    nvmlInit()

    print('========== GPU Utilization ==========')
    for gpu_id in range(num_total_gpus):
        h = nvmlDeviceGetHandleByIndex(gpu_id)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'GPU {gpu_id}')
        print(f'- Used:       {info.used / 1024 ** 3:>8.2f} B ({info.used / info.total * 100:.1f}%)')
        print(f'- Available:  {info.free / 1024 ** 3:>8.2f} B ({info.free / info.total * 100:.1f}%)')
        print(f'- Total:      {info.total / 1024 ** 3:>8.2f} B')
    print('=====================================')


def mean_sampled_logprob(completion):
    """Mean vLLM logprob of the tokens actually sampled in one completion (nan if unavailable)."""
    if not completion.logprobs or not completion.token_ids:
        return float('nan')
    values = [position[token_id].logprob for position, token_id in zip(completion.logprobs, completion.token_ids)]
    return sum(values) / len(values)


@ray.remote(num_gpus=1)
class OnPolicyActor:
    def __init__(self, model_name, num_gen_gpus, max_seq_length, max_new_tokens):
        # vLLM V1 (>= 0.11): run the engine core in this process so update_weight can reach the model.
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=num_gen_gpus,
            #max_model_len=max_seq_length + max_new_tokens,
            # Cached KV (e.g. the shared system-prompt prefix) would go stale after each weight update.
            enable_prefix_caching=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(
        self,
        input_texts,
        max_new_tokens,
        temperature=1.0,
        num_generation_per_prompt=1,
    ):
        outputs = self.model.generate(
            input_texts,
            sampling_params=SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=1.0,
                n=num_generation_per_prompt,
                logprobs=0,  # sampled token only; raw (pre-temperature) logprobs by default in vLLM V1
            ),
            use_tqdm=False,
        )
        results = [
            dict(
                input_text=output.prompt,
                generated_text=[output.outputs[i].text for i in range(num_generation_per_prompt)],
                finish_reason=[output.outputs[i].finish_reason for i in range(num_generation_per_prompt)],
                num_tokens=[len(output.outputs[i].token_ids) for i in range(num_generation_per_prompt)],
                logprob_mean=[mean_sampled_logprob(output.outputs[i]) for i in range(num_generation_per_prompt)],
            ) for output in outputs
        ]
        return results

    def update_weight(self, name, param):
        # `param` arrives via Ray from the paired trainer rank. Each actor has tensor_parallel_size=1, so the
        # original intra-actor broadcast was a no-op and is dropped.
        model = self.model.llm_engine.model_executor.driver_worker.worker.model_runner.model
        model.load_weights(weights=[(name, param)])


def generate_on_policy_data(
    on_policy_actor,
    tokenizer,
    datapoint_ids,
    dataset,
    max_new_tokens,
    temperature,
    num_generation_per_prompt=1,
):
    batch_for_gen = dataset.tokenize_for_inference_by_ids(datapoint_ids)
    input_texts = tokenizer.batch_decode(batch_for_gen['input_ids'], skip_special_tokens=False)
    results = ray.get(
        on_policy_actor.generate.remote(
            input_texts=input_texts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_generation_per_prompt=num_generation_per_prompt,
        )
    )
    on_policy_texts = [result['generated_text'] for result in results]
    parsed_on_policy_texts = [[dataset.parse_output_text(text) for text in on_policy_text] for on_policy_text in on_policy_texts]
    ends_with_eos = [[r == 'stop' for r in result['finish_reason']] for result in results]
    gen_stats = dict(
        num_tokens=[result['num_tokens'] for result in results],        # bsz x num_generation_per_prompt
        vllm_logprob_mean=[result['logprob_mean'] for result in results],
        input_texts=input_texts,                                        # exact prompt strings sent to vLLM
    )
    return on_policy_texts, parsed_on_policy_texts, ends_with_eos, gen_stats


@torch.no_grad()
def run_generation_eval(on_policy_actor, eval_dataset, tokenizer, local_rank, world_size, max_new_tokens):
    """Greedy pass@1 on this rank's shard of the eval set, using the paired (already weight-synced) vLLM actor."""
    datapoints = eval_dataset.datapoints[local_rank::world_size]
    if not datapoints:
        return dict(correct=[], ends_with_eos=[], num_tokens=[])
    # Same prompt construction as training rollouts, including the dataset's assistant prefill.
    input_texts = [eval_dataset.build_generation_prompt(datapoint['messages']) for datapoint in datapoints]
    results = ray.get(on_policy_actor.generate.remote(
        input_texts=input_texts,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        num_generation_per_prompt=1,
    ))
    return dict(
        correct=[
            float(eval_dataset.reward_fn(eval_dataset.parse_output_text(result['generated_text'][0]), datapoint))
            for result, datapoint in zip(results, datapoints)
        ],
        ends_with_eos=[float(result['finish_reason'][0] == 'stop') for result in results],
        num_tokens=[result['num_tokens'][0] for result in results],
    )


def gather_from_all_ranks(obj):
    """all_gather a picklable object; returns the list of every rank's object (collective: call on all ranks)."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, obj)
    return gathered


def summarize_step_stats(per_rank_stats):
    """Turn the per-rank lists collected over one optimizer step into scalar metrics."""
    def flat(key):
        return np.array([v for stats in per_rank_stats for v in stats[key]], dtype=np.float64)

    rewards = flat('reward')
    raw_rewards = flat('raw_reward')
    lengths = flat('num_tokens')
    advantages = flat('advantage')
    # Groups: one row of num_generation_per_prompt rewards per prompt.
    groups = [np.asarray(group, dtype=np.float64) for stats in per_rank_stats for group in stats['reward_groups']]
    solved = rewards > 0
    logprob_diff = np.abs(flat('local_logprob_mean') - flat('vllm_logprob_mean'))

    def safe(fn, values):
        return float(fn(values)) if values.size else float('nan')

    return {
        'train/loss': safe(np.mean, flat('loss')),
        'train/reward': safe(np.mean, rewards),
        'train/reward_raw': safe(np.mean, raw_rewards),
        'train/stop_rate': safe(np.mean, flat('ends_with_eos')),
        'train/groups_zero_std_frac': float(np.mean([g.std() == 0 for g in groups])),
        'train/groups_all_correct_frac': float(np.mean([(g > 0).all() for g in groups])),
        'train/groups_all_wrong_frac': float(np.mean([(g == 0).all() for g in groups])),
        'train/response_len_mean': safe(np.mean, lengths),
        'train/response_len_min': safe(np.min, lengths),
        'train/response_len_max': safe(np.max, lengths),
        'train/response_len_solved_mean': safe(np.mean, lengths[solved]),
        'train/response_len_unsolved_mean': safe(np.mean, lengths[~solved]),
        'train/advantages_mean': safe(np.mean, advantages),
        'train/advantages_std': safe(np.std, advantages),
        'train/advantages_min': safe(np.min, advantages),
        'train/advantages_max': safe(np.max, advantages),
        'debug/vllm_vs_local_logprob_diff_mean': safe(np.nanmean, logprob_diff),
        'debug/vllm_vs_local_logprob_diff_max': safe(np.nanmax, logprob_diff),
        'policy/entropy_mean': safe(np.mean, flat('entropy')),
        'objective/kl_penalty_k2': safe(np.mean, flat('kl')),
        'objective/kl_k1': safe(np.mean, flat('kl_k1')),
        'objective/kl_k3': safe(np.mean, flat('kl_k3')),
        # Wall-clock per rank, reported for the slowest rank (the step's critical path).
        'time/generation': max(stats['time_generation'] for stats in per_rank_stats),
        'time/forward_backward': max(stats['time_forward_backward'] for stats in per_rank_stats),
    }


def log_stats(metrics, step, wandb_project='none', exp_manager=None):
    log_msg = '[train] ' + ' | '.join(
        f'{k}={v:.6g}' if isinstance(v, float) else f'{k}={v}' for k, v in metrics.items()
    )
    if exp_manager is not None:
        exp_manager.write(log_msg)
    if wandb_project != 'none':
        wandb.log(metrics, step=step)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer(model, lr):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if 'bias' not in name]
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            'weight_decay': 0.0,
        },
        {
            'params': [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    return optimizer


@torch.no_grad()
def run_evaluation(model, eval_dataloader, local_rank):
    """
    Returns the averaged loss reduced across all processes.
    """
    model.eval()
    print_rank0('Running eval for loss...')

    losses = 0
    for step, batch in enumerate(eval_dataloader):
        inputs = {k: v.cuda() for k, v in batch.items() if k in ('input_ids', 'attention_mask', 'labels')}
        outputs = model(**inputs)
        loss = outputs.loss
        losses += loss.float()
    losses = losses / len(eval_dataloader)
    eval_loss = get_all_reduce_mean(losses.clone()).item()
    return eval_loss


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f'{name}.{n}'
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result


class RewardModel:
    def __init__(
        self,
        reward_model_name,
        reward_fn,
        tokenizer,
        **kwargs,
    ):
        self.reward_model_name = reward_model_name
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.model = None  # TODO: initialize model
    
    def __call__(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def compute_outcome_reward(self, text, datapoint):
        return self.reward_fn(text, datapoint)


@ray.remote(num_gpus=1)
class RLRayFSDPTrainer(DistributedBaseTrainer):
    def __init__(
        self,
        backend,
        master_addr,
        master_port,
        local_rank,
        world_size,
        on_policy_actor=None,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.world_size = world_size
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port
        self.on_policy_actor = on_policy_actor
    
    def run(self, **kwargs):
        run_name = kwargs.get('run_name', None)
        run_base_dir = kwargs.get('run_base_dir', None)
        seed = kwargs.get('seed', 0)
        model_name = kwargs.get('model_name', None)
        ref_model_name = kwargs.get('ref_model_name', 'none')
        reward_model_name = kwargs.get('reward_model_name', 'none')
        max_new_tokens = kwargs.get('max_new_tokens', 200)
        temperature = kwargs.get('temperature', 0.0)
        num_generation_per_prompt = kwargs.get('num_generation_per_prompt', 1)
        kl_beta = kwargs.get('kl_beta', 0.0)
        gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', None)
        lr = kwargs.get('lr', None)
        num_epochs = kwargs.get('num_epochs', None)
        warmup_ratio = kwargs.get('warmup_ratio', None)
        scheduler_type = kwargs.get('scheduler_type', None)
        gradient_clipping = kwargs.get('gradient_clipping', None)
        eval_every_n_steps = kwargs.get('eval_every_n_steps', None)
        # Greedy pass@1 on the eval split via vLLM every N optimizer steps (and before training); 0 disables.
        eval_generation_every_n_steps = kwargs.get('eval_generation_every_n_steps', 0)
        # Log the exact vLLM prompt + first sampled response for the first N training micro-batches (rank 0); 0 disables.
        num_prompts_to_print = kwargs.get('num_prompts_to_print', 0)
        save_every_n_steps = kwargs.get('save_every_n_steps', None)
        print_every_n_steps = kwargs.get('print_every_n_steps', None)
        model_module_path = kwargs.get('model_module_path', None)
        transformer_layer_module_path = kwargs.get('transformer_layer_module_path', None)
        loss_module_path = kwargs.get('loss_module_path', None)
        cache_path = kwargs.get('cache_path', None)
        data_module_path = kwargs.get('data_module_path', None)
        dataset_name = kwargs.get('dataset_name', None)
        num_train_datapoints = kwargs.get('num_train_datapoints', None)
        num_eval_datapoints = kwargs.get('num_eval_datapoints', None)
        train_batch_size_per_device = kwargs.get('train_batch_size_per_device', None)
        eval_batch_size_per_device = kwargs.get('eval_batch_size_per_device', None)
        total_batch_size = kwargs.get('total_batch_size', None)
        max_seq_length = kwargs.get('max_seq_length', None)
        wandb_project = kwargs.get('wandb_project', 'none')
        show_gpu_usage = kwargs.get('show_gpu_usage', False)
        num_gpus = kwargs.get('num_gpus', None)
        num_gen_gpus = kwargs.get('num_gen_gpus', None)

        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = str(self.master_port)
        os.environ['LOCAL_RANK'] = str(self.local_rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)

        backend = self.backend
        master_addr = self.master_addr
        master_port = self.master_port
        local_rank = self.local_rank
        world_size = self.world_size

        exp_start_time = time.time()
        assert run_name is not None, 'run_name must be specified'

        run_dir = Path(run_base_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        exp_manager = ExperimentManager(run_dir, world_size=world_size)
        exp_manager.write(f'backend={backend} | master_addr={master_addr} | master_port={master_port} | num_gpus={num_gpus} | num_gen_gpus={num_gen_gpus} | world_size={world_size}')

        torch.cuda.set_device(0)
        dist.init_process_group(
            backend=backend,
            rank=local_rank,
            world_size=world_size,
        )

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        transformers.set_seed(seed)

        config = copy.deepcopy(kwargs)
        print_rank0(config)

        model_module = load_module(model_module_path)
        transformer_layer_module = load_module(transformer_layer_module_path)
        model = model_module.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            use_cache=False,
            trust_remote_code=True,
            local_files_only=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='right')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer = tokenizer_factory(tokenizer, dataset_split='train')

        fsdp_config = dict(
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={transformer_layer_module},
            ),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            backward_prefetch=None,
            param_init_fn=None,
            cpu_offload=None,
            use_orig_params=True,  # Critical for generation with FSDP.summon_full_params()
        )
        model = FSDP(model, **fsdp_config)

        if ref_model_name != 'none' and kl_beta > 0.0:
            ref_model = model_module.from_pretrained(
                ref_model_name,
                torch_dtype=torch.bfloat16,
                use_cache=False,
                trust_remote_code=True,
                local_files_only=True,
            )
            ref_model.eval()
            ref_model = FSDP(ref_model, **fsdp_config)
        else:
            ref_model = None

        optimizer = get_optimizer(model, lr)

        dataset_module = load_module(data_module_path, dataset_name)
        train_dataloader, train_sampler, train_dataset = dataset_module.construct_dataloader(
            num_datapoints=num_train_datapoints,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            batch_size=train_batch_size_per_device,
            shuffle=True,
            use_distributed=True,
            seed=seed,
            local_rank=local_rank,
            world_size=world_size,
            dataset_split='train',
            model_name=model_name,
        )
        eval_dataloader, eval_sampler, eval_dataset = dataset_module.construct_dataloader(
            num_datapoints=num_eval_datapoints,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            batch_size=eval_batch_size_per_device,
            shuffle=False,
            use_distributed=True,
            seed=seed,
            local_rank=local_rank,
            world_size=world_size,
            dataset_split='eval',
            model_name=model_name,
        )

        reward_model = RewardModel(
            reward_model_name=reward_model_name,
            reward_fn=train_dataset.reward_fn,
            tokenizer=tokenizer,
            model_module=model_module,
            fsdp_config=fsdp_config,
        )

        training_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

        total_training_steps = training_steps_per_epoch * num_epochs
        scheduler = transformers.get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=math.ceil(total_training_steps * warmup_ratio),
            num_training_steps=total_training_steps,
        )

        loss_module = load_module(loss_module_path)(config=config)
        model.gradient_checkpointing_enable()
        model.train()
        dist.barrier()

        if local_rank == 0:
            with open(run_dir / 'config.yaml', 'w') as f:
                yaml.dump(config, f)

            if wandb_project != 'none':
                # Online logging (upstream forced WANDB_MODE=dryrun). Needs `wandb login` / WANDB_API_KEY on the node.
                os.environ['WANDB_PROJECT'] = wandb_project
                wandb.init(project=wandb_project, name=run_name, config=config, dir=run_base_dir)

        def log_generation_eval(log_step):
            eval_start = time.time()
            per_rank = gather_from_all_ranks(run_generation_eval(
                self.on_policy_actor, eval_dataset, tokenizer, local_rank, world_size, max_new_tokens,
            ))
            if local_rank == 0:
                correct = [v for stats in per_rank for v in stats['correct']]
                eos = [v for stats in per_rank for v in stats['ends_with_eos']]
                lengths = [v for stats in per_rank for v in stats['num_tokens']]
                log_stats({
                    'eval/pass_at_1': float(np.mean(correct)),
                    'eval/stop_rate': float(np.mean(eos)),
                    'eval/response_len_mean': float(np.mean(lengths)),
                    'eval/num_examples': len(correct),
                    'time/eval': time.time() - eval_start,
                }, step=log_step, wandb_project=wandb_project, exp_manager=exp_manager)

        def new_step_stats():
            return dict(
                loss=[], reward=[], raw_reward=[], ends_with_eos=[], num_tokens=[], advantage=[], reward_groups=[],
                local_logprob_mean=[], vllm_logprob_mean=[], entropy=[], kl=[], kl_k1=[], kl_k3=[],
                time_generation=0.0, time_forward_backward=0.0,
            )

        start_time = time.time()
        global_step = 0
        total_rewards = []
        total_kl = []
        best_model_reward = float('-inf')
        step_stats = new_step_stats()

        # wandb step = number of optimizer updates completed, so the pre-training eval lands at step 0.
        if eval_generation_every_n_steps:
            log_generation_eval(log_step=0)
        step_start_time = time.time()

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.run_dir = run_dir
        self.local_rank = local_rank
        self.global_step = global_step
        self.epoch = 0
        self.train_sampler = train_sampler
        self._register_signal_handlers()

        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            for step, batch in enumerate(train_dataloader):
                datapoint_ids = batch['datapoint_ids'].tolist()
                datapoints = [train_dataset.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids]

                ###############################################################
                # NOTE: generate on-policy data
                gen_start = time.time()
                batch_on_policy_texts, batch_parsed_on_policy_texts, batch_ends_with_eos, gen_stats = generate_on_policy_data(
                    self.on_policy_actor,
                    tokenizer,
                    datapoint_ids,
                    train_dataset,
                    max_new_tokens,
                    temperature,
                    num_generation_per_prompt,
                )
                step_stats['time_generation'] += time.time() - gen_start
                ###############################################################

                fwd_bwd_start = time.time()
                output = loss_module.compute_loss(
                    model,
                    ref_model,
                    reward_model,
                    batch_on_policy_texts,
                    batch_parsed_on_policy_texts,
                    batch_ends_with_eos,
                    datapoints,
                    tokenizer,
                    local_rank=local_rank,
                    assistant_prefix=train_dataset.assistant_prefix,
                    **kwargs,
                )
                loss = output.loss
                rewards = output.rewards
                kl = output.kl
                probs = output.logprobs.exp()

                total_rewards.append(rewards)
                total_kl.append(kl)

                if local_rank == 0 and epoch == 0 and step < num_prompts_to_print:
                    for prompt_idx, input_text in enumerate(gen_stats['input_texts']):
                        exp_manager.write(
                            f'{"=" * 100}\n[prompt sample: step={step} batch_idx={prompt_idx} '
                            f'datapoint_id={datapoint_ids[prompt_idx]}] EXACT PROMPT SENT TO vLLM:\n{input_text}\n'
                            f'{"-" * 100}\n[first of {num_generation_per_prompt} samples | '
                            f'reward={rewards[prompt_idx, 0].item()} | ends_with_eos={batch_ends_with_eos[prompt_idx][0]} | '
                            f'parsed={batch_parsed_on_policy_texts[prompt_idx][0]!r}]\n'
                            f'{batch_on_policy_texts[prompt_idx][0]}\n{"=" * 100}'
                        )
                ###############################################################
                if print_every_n_steps is not None and step != 0 and step % print_every_n_steps == 0 and local_rank == 0:
                    try:
                        for batch_idx, (group_rewards, on_policy_texts, parsed_on_policy_texts, datapoint_id) in enumerate(zip(rewards, batch_on_policy_texts, batch_parsed_on_policy_texts, datapoint_ids)):
                            if batch_idx >= 3:
                                break
                            datapoint = train_dataset.id_to_datapoint[datapoint_id]
                            target_text = datapoint['targets']
                            group_rewards = group_rewards.tolist()
                            for group_idx, (reward, on_policy_text, parsed_on_policy_text) in enumerate(zip(group_rewards, on_policy_texts, parsed_on_policy_texts)):
                                if group_idx >= 1:
                                    break
                                _input_text = datapoint["messages"][0]["content"]
                                _on_policy_text = on_policy_text    
                                _parsed_on_policy_text = parsed_on_policy_text
                                print(f'{"#" * 100}\nlocal_rank={local_rank}\nid={datapoint_id}\n### input_text ###\n{_input_text}\n### on_policy_text ###\n{_on_policy_text}\npred_text={_parsed_on_policy_text}\ntarget_text={target_text}\nreward={reward}\n{"#" * 100}')
                    except Exception as e:
                        print(f'Error in printing: {e}')
                ###############################################################

                # Save predictions ############################################
                exp_manager.write_prediction(dict(
                    datapoint_ids=datapoint_ids,
                    datapoints=datapoints,
                    rewards=rewards.tolist(),
                    probs=probs.tolist(),
                    on_policy_texts=batch_on_policy_texts,
                    parsed_on_policy_texts=batch_parsed_on_policy_texts,
                    ends_with_eos=batch_ends_with_eos,
                    local_rank=local_rank,
                    global_step=global_step,
                ))
                ###############################################################

                (loss / gradient_accumulation_steps).backward()
                step_stats['time_forward_backward'] += time.time() - fwd_bwd_start

                # Per-sample logging stats for this micro-batch (bsz x num_generation_per_prompt) ##########
                step_stats['loss'].append(loss.item())
                step_stats['reward'] += rewards.flatten().tolist()
                step_stats['raw_reward'] += output.raw_rewards.flatten().tolist()
                step_stats['ends_with_eos'] += [float(e) for group in batch_ends_with_eos for e in group]
                step_stats['num_tokens'] += [n for group in gen_stats['num_tokens'] for n in group]
                step_stats['advantage'] += output.advantages.flatten().tolist()
                step_stats['reward_groups'] += rewards.tolist()
                step_stats['local_logprob_mean'] += output.logprobs.flatten().tolist()
                step_stats['vllm_logprob_mean'] += [v for group in gen_stats['vllm_logprob_mean'] for v in group]
                step_stats['entropy'] += output.entropy.flatten().tolist()
                step_stats['kl'] += kl.flatten().tolist()
                step_stats['kl_k1'] += output.kl_k1.flatten().tolist()
                step_stats['kl_k3'] += output.kl_k3.flatten().tolist()
                ###################################################################################

                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer_start = time.time()
                    grad_norm = model.clip_grad_norm_(gradient_clipping).item()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    time_optimizer = time.time() - optimizer_start
                    loss = get_all_reduce_mean(loss.detach())

                    total_rewards = torch.cat(total_rewards, dim=0)
                    reward = get_all_reduce_mean(total_rewards.mean())
                    total_rewards = []

                    total_kl = torch.cat(total_kl, dim=0)
                    kl = get_all_reduce_mean(total_kl.mean())
                    total_kl = []

                    per_rank_stats = gather_from_all_ranks(step_stats)
                    step_stats = new_step_stats()
                    if local_rank == 0:
                        metrics = summarize_step_stats(per_rank_stats)
                        metrics.update({
                            'train/global_step': global_step,
                            'train/epoch': epoch,
                            'train/lr': scheduler.get_last_lr()[0],
                            'train/grad_norm': grad_norm,
                            'time/optimizer': time_optimizer,
                            'time/elapsed_min': (time.time() - exp_start_time) / 60,
                        })

                    if reward > best_model_reward:
                        best_model_reward = reward
                        self.save_model(local_rank, model, tokenizer, optimizer, run_dir, ckpt_name=f'checkpoint-best')
                        exp_manager.write(f'Best model saved at global_step={global_step} | best_model_reward={best_model_reward}')

                    ###########################################################
                    # NOTE: update on-policy actor weights
                    def update_on_policy_actor_weights(model, on_policy_actor, local_rank):
                        full_state_dict = None
                        with FSDP.summon_full_params(model, writeback=False):
                            full_state_dict = model.state_dict()
    
                            for name, param in full_state_dict.items():
                                dist.broadcast(param, src=0)
                                ray.get(self.on_policy_actor.update_weight.remote(name, param))
                        dist.barrier()
                    weight_sync_start = time.time()
                    update_on_policy_actor_weights(model, self.on_policy_actor, local_rank)
                    time_weight_sync = time.time() - weight_sync_start
                    print_rank0('Weights updated!!!')
                    ###########################################################

                    if local_rank == 0:
                        metrics['time/weight_sync'] = time_weight_sync
                        metrics['time/step'] = time.time() - step_start_time
                        log_stats(metrics, step=global_step + 1, wandb_project=wandb_project, exp_manager=exp_manager)
                        print_gpu_usage(num_gpus)

                    if eval_generation_every_n_steps and (global_step + 1) % eval_generation_every_n_steps == 0:
                        log_generation_eval(log_step=global_step + 1)

                    if global_step and global_step % eval_every_n_steps == 0:
                        eval_loss = run_evaluation(model, eval_dataloader, local_rank)
                        log_info = f'\[eval]  global_step={global_step} / {total_training_steps} | epoch={epoch} | eval_loss={eval_loss:.6f}'
                        if exp_manager is not None:
                            exp_manager.write(log_info)
                        model.train()
    
                    if global_step and global_step % save_every_n_steps == 0:
                        self.save_model(local_rank, model, tokenizer, optimizer, run_dir, ckpt_name=f'checkpoint-{global_step}')
                    
                    self.model = model
                    self.tokenizer = tokenizer
                    self.optimizer = optimizer
                    self.scheduler = scheduler
                    self.run_dir = run_dir
                    self.local_rank = local_rank
                    self.global_step = global_step
                    self.epoch = 0
                    self.train_sampler = train_sampler
    
                    dist.barrier()
                    global_step += 1
                    step_start_time = time.time()
        dist.barrier()
    
        self.save_model(local_rank, model, tokenizer, optimizer, run_dir, ckpt_name='final')
        print_rank0(f'Training finished in {(time.time() - start_time) / 60:.2f} minutes')
        dist.destroy_process_group()
        return True  # successful termination


def main(**kwargs):
    backend = kwargs.pop('backend', 'nccl')
    master_addr = kwargs.pop('master_addr', '127.0.0.1')
    master_port = kwargs.pop('master_port', '29500')
    num_gpus = kwargs.get('num_gpus')
    num_gen_gpus = kwargs.get('num_gen_gpus')
    world_size = num_gpus - num_gen_gpus
    print(f'run_name={kwargs.get("run_name")} | backend={backend} | master_addr={master_addr} | master_port={master_port} | num_gpus={num_gpus} | num_gen_gpus={num_gen_gpus} | world_size={world_size}')

    ray.init()

    on_policy_actors = [
        OnPolicyActor.remote(
            model_name=kwargs.get('model_name'),
            num_gen_gpus=1,
            max_seq_length=kwargs.get('max_seq_length'),
            max_new_tokens=kwargs.get('max_new_tokens'),
        ) for _ in range(num_gen_gpus)
    ]

    training_actors = [
        RLRayFSDPTrainer.remote(
            backend,
            master_addr,
            master_port,
            local_rank,
            world_size,
            on_policy_actors[local_rank % num_gen_gpus],
        ) for local_rank in range(world_size)
    ]

    status = ray.get([actor.run.remote(**kwargs) for actor in training_actors])
    print(status)


if __name__ == '__main__':
    fire.Fire(main)
