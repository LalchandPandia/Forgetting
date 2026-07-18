import json
import logging
import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional, Union
import numpy as np
import datasets
import deepspeed
import torch
import contextlib
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    get_scheduler,
)

from open_instruct.dataset_transformation import (
    INPUT_IDS_KEY,
    TOKENIZED_SFT_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.model_utils import save_with_accelerate
from open_instruct.utils import ArgumentParserPlus, clean_last_n_checkpoints, get_last_checkpoint_path


logger = get_logger(__name__)
@dataclass
class FlatArguments:
    """
    Full arguments class for all fine-tuning jobs.
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    run_name: Optional[str] = None
    """A unique name of this run"""
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    # NOTE: tokenizer_name, tokenizer_revision, trust_remote_code, and use_slow_tokenizer
    # deliberately live only on TokenizerConfig (see open_instruct/dataset_transformation.py)
    # and must not be redefined here: ArgumentParserPlus registers CLI flags for every
    # field of every dataclass passed to it, and a duplicate field name across
    # FlatArguments/TokenizerConfig would make argparse raise a conflicting-option-string
    # error at startup. finetune.py follows the same convention.
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention in the model training"},
    )
    model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, "
                "then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use for fine-tuning."}
    )
    dataset_mixer: Optional[dict] = field(
        default=None, metadata={"help": "A dictionary of datasets (local or HF) to sample from."}
    )
    dataset_mixer_list: Optional[list[str]] = field(
        default=None, metadata={"help": "A list of datasets (local or HF) to sample from."}
    )
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )
    """The list of transform functions to apply to the dataset."""
    dataset_target_columns: list[str] = field(default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS)
    """The columns to use for the dataset."""
    dataset_cache_mode: str = "local"
    """The mode to use for caching the dataset. Options: 'hf' or 'local'."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_hash: Optional[str] = None
    """The hash of the dataset configuration."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    dataset_mix_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory to save the mixed dataset to disk."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a json/jsonl file)."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "The input validation data file (a json/jsonl file)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    
    clip_grad_norm: float = field(
        default=-1,
        metadata={"help": "Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead)."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer."},
    )
    logging_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Log the training loss and learning rate every logging_steps steps."},
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."},
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."},
    )
    
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use for learning rate adjustment.",
            "choices": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        },
    )
    metric: str = field(
        default="not_specified",
        metadata={
            "help": "metric used for selection.",
        },
    )
    metric
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Total number of training epochs to perform."},
    )
    output_dir: str = field(
        default="output/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    
    use_8bit_optimizer: bool = field(
        default=False,
        metadata={"help": "Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed."},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "If True, will use LORA (low-rank parameter-efficient training) to train the model."},
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."},
    )
    timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout for the training process in seconds."
            "Useful if tokenization process is long. Default is 1800 seconds (30 minutes)."
        },
    )
    reduce_loss: str = field(
        default="mean",
        metadata={
            "help": "How to reduce loss over tokens. Options are 'mean' or 'sum'."
            "Using 'sum' can improve chat model performance."
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "Entity to use for logging to wandb."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "If the training should continue from a checkpoint folder."},
    )
    with_tracking: bool = field(
        default=False,
        metadata={"help": "Whether to enable experiment trackers for logging."},
    )
    report_to: Union[str, List[str]] = field(
        default="all",
        metadata={
            "help": "The integration(s) to report results and logs to. "
            "Can be a single string or a list of strings. "
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Specify multiple by listing them: e.g., ['tensorboard', 'wandb']"
        },
    )
    save_to_hub: Optional[str] = field(
        default=None,
        metadata={"help": "Save the model to the Hub under this name. E.g allenai/your-model"},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Turn on gradient checkpointing. Saves memory but slows training."},
    )
    baseline: bool = field(
        default=False,
        metadata={"help": "carry out experiments for baseline. "},
    )
    
    calculate_metric: bool = field(
        default=False,
        metadata={"help": "calculate our metric per epoch. "},
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "If set, overrides the number of training steps. Otherwise, num_train_epochs is used."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for initialization and dataset shuffling."})
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."  # noqa
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the content of the output directory. Means that resumption will always start from scratch."
        },
    )
    keep_last_n_checkpoints: int = field(
        default=-1,
        metadata={"help": "How many checkpoints to keep in the output directory. -1 for all."},
    )
    fused_optimizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use fused AdamW or not.",
        },
    )
    load_balancing_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to include a load balancing loss (for OLMoE) or not.",
        },
    )
    load_balancing_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for load balancing loss if applicable."},
    )
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    hf_metadata_dataset: Optional[str] = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""

    def __post_init__(self):
        if self.reduce_loss not in ["mean", "sum"]:
            raise ValueError("reduce_loss must be either 'mean' or 'sum'")
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.dataset_mixer is None
            and self.dataset_mixer_list is None
        ):
            raise ValueError("Need either a dataset name, dataset mixer, or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["json", "jsonl"], "`train_file` should be a json or a jsonl file."
        if (
            (self.dataset_name is not None and (self.dataset_mixer is not None or self.dataset_mixer_list is not None))
            or (self.dataset_name is not None and self.train_file is not None)
            or (
                (self.dataset_mixer is not None or self.dataset_mixer_list is not None) and self.train_file is not None
            )
            or (self.dataset_mixer is not None and self.dataset_mixer_list is not None)
        ):
            raise ValueError("Cannot provide two dataset selection mechanisms.")
        
def compute_global_grad_norm(model):
    grads = []
    for p in model.parameters():
        full_grad = deepspeed.utils.safe_get_full_grad(p)
        if full_grad is not None:
            grads.append(full_grad.detach().float())
    if not grads:
        return 0.0
    # Compute global L2 norm
    return torch.norm(torch.stack([g.norm() for g in grads]))

def loss_per_sample(logits, labels):

    BATCH_SIZE = logits.size(dim=0)
    VOCAB_SIZE = logits.size(dim=-1)
    #print('BATCH_SIZE ',BATCH_SIZE)
    #print('VOCAB_SIZE ',VOCAB_SIZE)

    # move labels to correct device to enable model parallelism
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    #print('Luke loss_per_sample ',flag)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(shift_logits.view(-1, VOCAB_SIZE), shift_labels.view(-1))

    # split concated batch loss values into samples
    #print('loss ',loss.shape)
    #This will contain tuple of size BATCH_SIZE. Each element of tuple contains loss per token
    
    loss_tensor_per_sample = torch.tensor_split(loss, BATCH_SIZE)
    
    # In the unlikely scenario that loss is exactly zero for
    # all tokens, torch.tensor(0.0) is used
    #This a list per gpu (so each gpu has its own list) which contains loss tensor of size BATCH_SIZE
    #[tensor(2.7506, device='cuda:0'), tensor(1.1847, device='cuda:0'), tensor(2.1895, device='cuda:0'), tensor(2.5760, device='cuda:0')]
    sample_loss = [torch.sum(tensor)/torch.count_nonzero(tensor)
                    if bool(torch.count_nonzero(tensor) > 0) 
                    else torch.tensor(0.0).to(tensor.device)
                    for tensor in loss_tensor_per_sample]
    
    sample_perplexity = [torch.exp(s).item() for s in sample_loss]
    sample_likelihood = [torch.exp(-s).item() for s in sample_loss]

    # labels[i] maps to sample_loss[i] and sample_perplexity[i]
    return sample_perplexity, sample_likelihood

def validate(model: AutoModelForCausalLM, dataloader: DataLoader, tokenizer: AutoTokenizer, accelerator, completed_steps):
    local_total_tokens = 0
    total_loss = 0
    validation_loss_per_step = []
    model.eval()
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            local_total_tokens += batch["attention_mask"].sum()
            loss = outputs.loss
            #print('step ',step, ' validation ',loss)
            local_batch_loss = loss.detach()
            total_tokens = accelerator.gather(local_total_tokens).sum().item()
            all_batch_loss = accelerator.gather_for_metrics(local_batch_loss)
            validation_loss_per_step.extend(all_batch_loss.tolist())
    avg_loss = sum(validation_loss_per_step) / len(validation_loss_per_step)
    metrics_to_log = {'validation_loss':avg_loss}
    accelerator.log(metrics_to_log, step=completed_steps)
    model.train()

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def main(args: FlatArguments, tc: TokenizerConfig):
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    project_name='open_instruct_OffTarget_sequence'
    args.run_name = f"open_instruct_{args.model_name_or_path.replace('/', '_')}_{args.dataset}_{args.metric}"
    
    
    '''
    if args.metric == 'variability_olmo2_7B':
        run_id = '8yim40gh'
    elif args.metric == 'vog_score_olmo2_7B':
        run_id = 'qusc6w8k'
    elif args.metric == 'diversity_olmo2_7B_random':
        run_id = '2zhwnbnd'
    elif args.metric == 'perplexity_p0_olmo2_7B':
        run_id = 'zmbkl77l'
    elif args.metric == 'random_6517878177422821419':
        run_id = 'oombgjl8'
    else:
        print('set new id error')
    '''
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = "wandb"
    accelerator_log_kwargs["project_dir"] = args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=360000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_seedable_sampler=True,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )

    if args.with_tracking:
        experiment_config = {}
        experiment_config["lr_scheduler_type"] = args.lr_scheduler_type

        #print('Luke experiment_config ',experiment_config)
        experiment_config['report_to'] = 'wandb'
        accelerator.init_trackers(
            project_name,
            experiment_config,
            #init_kwargs={"wandb"}
            init_kwargs={
                "wandb": {
                    "name": args.run_name,
                    "entity": args.wandb_entity,
                    "tags": 'None',
                    "group": 'olmo2_7B',
                    "settings": {'_service_wait':3000},
                }
            },
        )
        wandb_tracker = accelerator.get_tracker("wandb")

    # if you get timeouts (e.g. due to long tokenization) increase this.
    '''
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))


    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_seedable_sampler=True,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )'''
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    # Tokenizer setup: delegate entirely to TokenizerConfig, which already
    # knows how to handle Llama/OLMo/Qwen/etc. tokenizers and chat templates
    # (see open_instruct/dataset_transformation.py), instead of re-deriving
    # per-model special-token logic here.
    tc.tokenizer_revision = args.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if tc.tokenizer_revision != args.model_revision and tc.tokenizer_name_or_path != args.model_name_or_path:
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{args.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{args.model_name_or_path=}`."""
        logger.warning(warning)
    tokenizer = tc.tokenizer

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=tc.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # update embedding size after resizing for sum loss
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    if args.use_lora:
        print("Initializing LORA model...")
        #changed target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"] to
        target_modules=["q_proj", "o_proj", "v_proj", "k_proj"] #in order to keep training set up same as LESS
        #for MPT
        #target_modules=["gate_proj", "up_proj", "down_proj"]
        peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Dataset loading/tokenization: borrowed from finetune.py's pipeline
    # (get_cached_dataset_tulu + dataset_transform_fn) instead of the old
    # raw load_dataset() + manual encode_with_messages_format()/.map() path.
    # This also gets us on-disk dataset caching for free.
    transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]
    dataset_mixer_list = args.dataset_mixer_list
    if dataset_mixer_list is None:
        # Fall back to --train_file for backward compatibility; local .jsonl
        # paths are supported directly as dataset_mixer_list entries.
        assert args.train_file is not None, "Need either --dataset_mixer_list or --train_file."
        dataset_mixer_list = [args.train_file, "1.0"]
    with accelerator.main_process_first():
        train_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=dataset_mixer_list,
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=args.dataset_target_columns,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_hash,
            hf_entity=args.hf_entity,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
        )
    train_dataset.set_format(type="pt")
    if accelerator.is_main_process:
        visualize_token(train_dataset[0][INPUT_IDS_KEY], tokenizer)

    #for warmup in less
    #p = 0.05
    #for training after selection
    p = 1
    #********************************for debugging ***************************************
    sample_size = int(len(train_dataset) * p)
    #select only 5% of the data
    index = np.random.permutation(len(train_dataset))[:sample_size]

    train_dataset  = train_dataset.select(index)
    logger.info(f"len(train_dataset) after subset selection: {len(train_dataset)}")
    # debugging tool for fewer samples
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        logger.info(f"Limiting training samples to {max_train_samples} from {len(train_dataset)}.")
        train_dataset = train_dataset.select(range(max_train_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size,
    )
    #eval Dataloader:
    '''
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
    )
    '''

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, fused=args.fused_optimizer)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print('Luke num_update_steps_per_epoch ',num_update_steps_per_epoch)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    print('Luke num_update_steps_per_epoch ',num_update_steps_per_epoch, ' args.max_train_steps ',args.max_train_steps)

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler
    # for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set.
    # In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set.
    # So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the
    # entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of
    # updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    )
    print(f'Luke num_training_steps_for_scheduler {num_training_steps_for_scheduler}')
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    # Prepare evaldataloader with `accelerator`.
    #eval_dataloader = accelerator.prepare(eval_dataloader)
    print('Luke  len(train_dataloader) ',len(train_dataloader))

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print('Luke recalculate num_update_steps_per_epoch ',num_update_steps_per_epoch)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        print(' overrode_max_train_steps ',args.max_train_steps)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    print('Luke recalculated args.num_train_epochs ',args.num_train_epochs)
    #print('Luke args ',args)
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    '''
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]

        #print('Luke experiment_config ',experiment_config)
        experiment_config['report_to'] = 'wandb'
        accelerator.init_trackers(
            f"open_instruct_{args.model_name_or_path.replace('/', '_')}_{args.dataset}_less_warmup",
            experiment_config,
            #init_kwargs={"wandb"}
            init_kwargs={
                "wandb": {
                    "name": args.run_name,
                    "entity": args.wandb_entity,
                    "tags": 'None',
                    "group": 'olmo2_7B',
                    "settings": {'_service_wait':300},
                }
            },
        )
        wandb_tracker = accelerator.get_tracker("wandb")
    '''
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    last_checkpoint_path = get_last_checkpoint_path(args)
    print('Luke last_checkpoint_path ',last_checkpoint_path)
    if last_checkpoint_path:
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")
        accelerator.load_state(last_checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
            print('Luke resume_step ',resume_step)
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
    print(f"Starting from epoch {starting_epoch} and step {completed_steps}.")
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    local_total_tokens = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    total_token_including_padding = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    start_time = time.time()


    for epoch in range(starting_epoch, args.num_train_epochs):
        print('Luke epoch ',epoch)
        model.train()
        
        
        #train_data_confidence_per_epoch = []
        #train_data_ids_per_epoch = []

        train_dataloader.set_epoch(epoch)
        total_loss = 0
        total_aux_loss = 0
        if last_checkpoint_path and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        print('Luke len(active_dataloader) ',len(active_dataloader))
        #f = open(f'/net/scratch/lcpandia/open_instruct/dump_pplxs/epoch-{round(epoch)}-train_data.json', 'w')
        for step, batch in enumerate(active_dataloader):
            new_batch = {}
            #print('Luke batch ',batch )
            new_batch['input_ids'] = batch['input_ids']
            new_batch['attention_mask'] = batch['attention_mask']
            new_batch['labels'] = batch['labels']

            input_ids = new_batch['input_ids'][0]
            labels = new_batch['labels'][0]

            #print('new_batch ',new_batch)
            
            local_total_tokens += batch["attention_mask"].sum()
            total_token_including_padding += batch["attention_mask"].numel()
            with accelerator.accumulate(model):
                if args.load_balancing_loss:
                    outputs = model(**new_batch, use_cache=False, output_router_logits=True)
                else:
                    outputs = model(**new_batch, use_cache=False)

                if args.reduce_loss == "mean":
                    loss = outputs.loss
                else:
                    # reduce loss is sum
                    # this ensures that we weight all tokens in the dataset equally,
                    # rather than weighting each overall example equally when
                    # using high amounts of gradient accumulation.
                    # this can result in > 5 point improvements in AlpacaEval
                    # see https://github.com/huggingface/transformers/issues/24725 for
                    # more discussion and details.
                    logits = outputs.logits
                    labels = batch["labels"]
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                    shift_logits = shift_logits.view(-1, embedding_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    if args.load_balancing_loss:
                        aux_loss = args.load_balancing_weight * outputs.aux_loss
                        loss += aux_loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                if args.load_balancing_loss:
                    total_aux_loss += aux_loss.detach().float()
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients:
                    grad_norm = compute_global_grad_norm(model)
                    if args.clip_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item()
                        / args.gradient_accumulation_steps
                        / args.logging_steps
                    )
                    total_tokens = accelerator.gather(local_total_tokens).sum().item()
                    total_tokens_including_padding = accelerator.gather(total_token_including_padding).sum().item()
                    metrics_to_log = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm.item(),
                        "train_loss": avg_loss,
                        "total_tokens": total_tokens,
                        "per_device_tps": total_tokens / accelerator.num_processes / (time.time() - start_time),
                        "total_tokens_including_padding": total_tokens_including_padding,
                        "per_device_tps_including_padding": total_tokens_including_padding
                        / accelerator.num_processes
                        / (time.time() - start_time),
                    }
                    if args.load_balancing_loss:
                        avg_aux_loss = (
                            accelerator.gather(total_aux_loss).mean().item()
                            / args.gradient_accumulation_steps
                            / args.logging_steps
                        )
                        logger.info(
                                f" Completed Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, Aux Loss: {avg_aux_loss}, TPS: {total_tokens / (time.time() - start_time)} current step: {step} "
                        )
                        metrics_to_log["aux_loss"] = avg_aux_loss
                    else:
                        logger.info(
                                f" Completed  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, TPS: {total_tokens / (time.time() - start_time)} current step: {step}"
                        )
                    if args.with_tracking:
                        accelerator.log(
                            metrics_to_log,
                            step=completed_steps,
                        )
                    total_loss = 0
                    total_aux_loss = 0
                
                #Temporarily puasing the checkpointing
                            
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                        # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
                        with open(
                            os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"), "w"
                        ) as f:
                            f.write("COMPLETED")  # annoyingly, empty files arent uploaded by beaker.
                        if accelerator.is_local_main_process:
                            clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
                        accelerator.wait_for_everyone()
                        #validate(model, eval_dataloader, tokenizer, accelerator, completed_steps)
                
                        
                
                if completed_steps >= args.max_train_steps:
                    print('Luke completed_steps ',completed_steps, ' args.max_train_steps ',args.max_train_steps)
                    break
        #we want to save the metric after each epoch so I am removing this check
        #if checkpointing_steps == "epoch":
        #skipping saving per epoch to avoid space issue
        #temporarily skipping per epoch saving
        '''
        output_dir = f"epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)   
        #accelerator.save_state(output_dir)
        save_with_accelerate(
            accelerator,
            model,
            tokenizer,
            output_dir,
            args.use_lora,
        )
        # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
        with open(os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"), "w") as f:
            f.write("COMPLETED")  # annoyingly, empty files arent uploaded by beaker.
        
        #once my runs are over I should enable saving per epoch
        #This ensures that complete dataloader is used at the start of each successful epoch
        #otherwise we will skip resume_step times batches in dataloader in this code if last_checkpoint_path and resume_step is not None
        resume_step = None
        if accelerator.is_local_main_process:
            clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
        accelerator.wait_for_everyone()
        '''

    if args.output_dir is not None:
        #output_dir = f"epoch_{epoch}"
        #output_dir = os.path.join(args.output_dir, output_dir)
        #accelerator.save_state(output_dir)
        save_with_accelerate(
            accelerator,
            model,
            tokenizer,
            args.output_dir,
            args.use_lora,
        )

    # remove all checkpoints to save space
    if accelerator.is_local_main_process:
        clean_last_n_checkpoints(args.output_dir, keep_last_n_checkpoints=-1)
       

    
    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()
if __name__ == "__main__":
    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = parser.parse()
    main(args, tc)
