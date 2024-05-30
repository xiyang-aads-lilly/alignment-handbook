#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2024 The Eli Lilly AADS Team. All rights reserved.
# Author: Xi Yang (xi.yang5@lilly.com)

# this is copied from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/utils/unsloth.py

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch


logger = logging.getLogger(__name__)


def get_current_device() -> torch.device:
    r"""
    Gets the current available device.
    """
    if torch.cuda.is_available():
        # might cause problem if we have multi-GPU and export CUDA VISIBLE DEVICE as not 0
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)


def _get_unsloth_kwargs(config, model_name_or_path: str, model_args) -> Dict[str, Any]:
    return {
        "model_name": model_name_or_path,
        "max_seq_length": model_args.model_max_length or 4096,
        "dtype": model_args.compute_dtype,
        "load_in_4bit": model_args.quantization_bit == 4,
        "token": model_args.hf_hub_token,
        "device_map": {"": get_current_device()},
        "rope_scaling": getattr(config, "rope_scaling", None),
        "fix_tokenizer": False,
        "trust_remote_code": True,
        "use_gradient_checkpointing": "unsloth",
    }


def load_unsloth_pretrained_model(config, model_args):
    r"""
    Optionally loads pretrained model with unsloth. Used in training.
    """
    from unsloth import FastLanguageModel

    unsloth_kwargs = _get_unsloth_kwargs(
        config, model_args.model_name_or_path, model_args
    )
    try:
        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
    except NotImplementedError:
        logger.warning(
            "Unsloth does not support model type {}.".format(
                getattr(config, "model_type", None)
            )
        )
        model = None
        model_args.use_unsloth = False

    return model


def get_unsloth_peft_model(model_name, max_seq_length, peft_kwargs: Dict[str, Any]):
    r"""
    Gets the peft model for the pretrained model with unsloth. Used in training.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    unsloth_peft_kwargs = {
        "model": model,
        "max_seq_length": max_seq_length,
        "use_gradient_checkpointing": "unsloth",
    }

    peft_kwargs["lora_dropout"] = 0.0
    peft_kwargs.pop("task_type", None)

    return (
        FastLanguageModel.get_peft_model(**unsloth_peft_kwargs, **peft_kwargs),
        tokenizer,
    )


def load_unsloth_peft_model(config, model_args, is_trainable: bool):
    r"""
    Loads peft model with unsloth. Used in both training and inference.
    """
    from unsloth import FastLanguageModel

    unsloth_kwargs = _get_unsloth_kwargs(
        config, model_args.adapter_name_or_path[0], model_args
    )
    try:
        if not is_trainable:
            unsloth_kwargs["use_gradient_checkpointing"] = False

        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
    except NotImplementedError:
        raise ValueError(
            "Unsloth does not support model type {}.".format(
                getattr(config, "model_type", None)
            )
        )

    if not is_trainable:
        FastLanguageModel.for_inference(model)

    return model
