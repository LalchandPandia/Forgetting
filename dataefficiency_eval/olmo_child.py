import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.base_model import BaseModel


class Olmo(BaseModel):
    """OLMo-2-Instruct family, trained with the Tulu chat template
    ("<|system|>\\n...\\n<|user|>\\n...\\n<|assistant|>\\n...{eos_token}", see
    open-instruct/open_instruct/dataset_transformation.py's "tulu" template).

    Lives in dataefficiency_eval/ rather than dataefficiency/scripts/ (unlike
    Llama/Mistral/Qwen) to keep dataefficiency/ itself untouched - it's picked
    up via sys.path the same way the scripts/utils modules from dataefficiency
    are, see eval_all_datasets.py.

    Unlike Llama/Mistral/Qwen, OLMo's eos_token and pad_token are not one
    fixed literal string across checkpoints - open-instruct's own chat
    template inserts the tokenizer's actual `eos_token` at render time rather
    than a hardcoded value. So this class resolves both from the real
    tokenizer once it's loaded (get_model_and_tokenizer below) instead of
    guessing a literal up front like the other *_child.py classes do.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.assistant_start_token = "<|assistant|>"
        # Placeholders until get_model_and_tokenizer() resolves the real
        # values from the loaded tokenizer.
        self.eos_token = None
        self.pad_token = None

    def format_chat_template(self, row):
        instruction, usr_query, assistant_output = self.create_prompt(row)

        row_json = [{"role": "system", "content": instruction},
                {"role": "user", "content": usr_query},
                {"role": "assistant", "content": assistant_output}]

        return row_json

    def get_model_and_tokenizer(self, tokenizer_path=None, use_flash_attention=True, use_safetensors=False, load_dtype=None, on_vector=False):
        # Reimplemented (rather than calling BaseModel.get_model_and_tokenizer)
        # because that method forces tokenizer.pad_token to match self.pad_token
        # via `tokenizer(self.pad_token, ...)`, which would crash here since
        # self.pad_token is intentionally None until the tokenizer is loaded.
        if on_vector:
            args = {'pretrained_model_name_or_path': "/model-weights/Meta-Llama-3.1-8B-Instruct"}
        else:
            args = {'pretrained_model_name_or_path': self.model_name}

        if use_flash_attention:
            args['attn_implementation'] = "flash_attention_2"
        if use_safetensors:
            args['use_safetensors'] = use_safetensors

        if self.use_quantized:
            args['quantization_config'] = self.bnb_config
            args['torch_dtype'] = torch.bfloat16 if not load_dtype else load_dtype
        else:
            args['torch_dtype'] = torch.bfloat16 if not load_dtype else load_dtype

        print('using args: ', args)
        model = AutoModelForCausalLM.from_pretrained(**args)

        if tokenizer_path is None or tokenizer_path == "":
            tokenizer_path = self.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"OLMo tokenizer had no pad_token; set pad_token = eos_token ({tokenizer.eos_token!r})")

        self.eos_token = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token

        return model, tokenizer
