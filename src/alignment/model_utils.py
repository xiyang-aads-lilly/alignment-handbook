# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint

from accelerate import Accelerator
from huggingface_hub import list_repo_files
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from peft import LoraConfig, PeftConfig

from .configs import DataArguments, ModelArguments
from .data import DEFAULT_CHAT_TEMPLATE, DEFAULT_PAD_TOKEN


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    print(Accelerator().local_process_index)
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map():
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None


def get_quantization_config(model_args: ModelArguments) -> BitsAndBytesConfig | None:
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        if model_args.torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
            bnb_4bit_quant_storage=model_args.bnb_4bit_quant_storage,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def tokenizer_and_embedding_resize(
    data_args: DataArguments,
    tokenizer: PreTrainedTokenizer,
    model: AutoModelForCausalLM,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    special_tokens_to_add = data_args.additional_special_tokens
    non_special_tokens_to_add = data_args.additional_non_special_tokens

    if special_tokens_to_add:
        for k, v in special_tokens_to_add.items():
            # get exsiting special token
            stk = tokenizer.special_tokens_map.get(k, None)
            if stk:
                idx = tokenizer.convert_tokens_to_ids(stk)
                tk_emb = model.get_input_embeddings().weight.data[idx]
            else:
                tk_emb = model.get_input_embeddings().weight.data.mean(
                    dim=0, keepdim=False
                )

            tokenizer.add_special_tokens({k: v})
            model.resize_token_embeddings(len(tokenizer))
            model.get_input_embeddings().weight.data[-1] = tk_emb

    # add non special extra tokens
    if non_special_tokens_to_add:
        num_new_tokens = tokenizer.add_tokens(non_special_tokens_to_add)
        model.resize_token_embeddings(len(tokenizer))
        if num_new_tokens > 0:
            input_embeddings_data = model.get_input_embeddings().weight.data
            output_embeddings_data = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
            output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def get_tokenizer(
    model_args: ModelArguments,
    data_args: DataArguments,
    train_args,
    auto_set_chat_template: bool = True,
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.model_name_or_path
            if model_args.tokenizer_name_or_path is None
            else model_args.tokenizer_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    # set tokenizer model max length from args
    if train_args:
        tokenizer.model_max_length = train_args.max_seq_length

    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_peft_config(model_args: ModelArguments) -> PeftConfig | None:
    if model_args.use_peft is False:
        return None

    if model_args.lora_target_modules == "all":
        model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        model_config.num_hidden_layers = 1
        temp_model = AutoModelForCausalLM.from_config(model_config)
        model_args.lora_target_modules = find_all_linear_names(temp_model)
        del temp_model
        del model_config

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config


def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    try:
        # Try first if model on a Hub repo
        repo_files = list_repo_files(model_name_or_path, revision=revision)
    except (HFValidationError, RepositoryNotFoundError):
        # If not, check local repo
        repo_files = os.listdir(model_name_or_path)
    return (
        "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files
    )


def get_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint
