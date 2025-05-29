#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import random
import sys
from pathlib import Path


p = Path(__file__).parent.parent / "src"
sys.path.append(p.as_posix())

from importlib.metadata import version as get_pkg_version

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    DataCollatorForPlw,
    GpuUtilPrintCallBack,
    H4ArgumentParser,
    ModelArguments,
    PLW_apply_chat_template,
    PLWTrainer,
    SFTConfig,
    apply_chat_template,
    fix_lr_scheduler_kwargs_float,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import SFTTrainer, setup_chat_format


logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()
    fix_lr_scheduler_kwargs_float(training_args)

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=[
            "messages",
            "chosen",
            "rejected",
            "prompt",
            "completion",
            "label",
        ],
    )
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    ################
    # Load tokenizer
    ################
    # Truncate from left to ensure we don't lose labels in final turn
    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(
        model_args, data_args, training_args, auto_set_chat_template=True
    )
    logger.info(f"tokenizer: {tokenizer}")

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path

    # For ChatML we need to add special tokens and resize the embedding layer
    # if (
    #     "<|im_start|>" in tokenizer.chat_template
    #     and "gemma-tokenizer-chatml" not in tokenizer.name_or_path
    #     and not model_args.use_unsloth
    # ):
    #     model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    #     model, tokenizer = setup_chat_format(model, tokenizer)
    #     model_kwargs = None

    # assign model_args to model_init_kwargs
    training_args.model_init_kwargs = model_kwargs

    #####################
    # Apply chat template
    #####################
    logger.info("*** apply chat template ***")

    if training_args.use_plw:
        raw_datasets = raw_datasets.map(
            PLW_apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "use_sample_template": training_args.use_plw_sample_template,
            },
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Applying chat template",
        )
    else:
        raw_datasets = raw_datasets.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "sft",
                "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            },
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Applying chat template",
        )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    # this is hard coded - move to config.yaml
    training_args.dataset_text_field = "text"

    # # no need for logging samples
    # with training_args.main_process_first(
    #     desc="Log a few random samples from the processed training set"
    # ):
    #     for index in random.sample(range(len(raw_datasets["train"])), 3):
    #         logger.info(
    #             f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}"
    #         )

    ##################
    # PYTORCH profiler
    ##################
    # prof = torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=2, repeat=2),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=training_args.logging_dir),
    #     profile_memory=True,
    #     with_stack=True,
    #     record_shapes=True,
    #     with_flops=True,
    #     with_modules=True,
    # )
    # ProfCallback(prof)

    ########################
    # Initialize the Trainer
    ########################
    peft_config = get_peft_config(model_args)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        # tokenizer=tokenizer,  #  need to change to procee `processing_class` for later transformer pkg update
        # callbacks=[GpuUtilPrintCallBack()],
    )

    if get_pkg_version("transformers") >= "5.0.0" or get_pkg_version("trl") >= "0.16.0":
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    if get_pkg_version("trl") < "0.13.0":
        trainer_kwargs["dataset_kwargs"] = training_args.dataset_kwargs

    if training_args.use_plw:
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 1
        trainer_kwargs["prompt_loss_weight"] = training_args.prompt_loss_weight
        # plw data collator
        # trainer_kwargs["data_collator"] = DataCollatorForPlw(tokenizer.pad_token_id)

    if model_args.use_unsloth:
        logger.info("*** use unsloth ***")
        from alignment.unsloth import get_unsloth_peft_model

        model, tokenizer = get_unsloth_peft_model(
            model_args.model_name_or_path,
            training_args.max_length,
            peft_config.to_dict(),
        )

        if (
            "<|im_start|>" in tokenizer.chat_template
            and "gemma-tokenizer-chatml" not in tokenizer.name_or_path
        ):
            try:
                model, tokenizer = setup_chat_format(model, tokenizer)
            except:
                pass

        if training_args.use_plw:
            trainer = PLWTrainer(**trainer_kwargs)
        else:
            trainer = SFTTrainer(**trainer_kwargs)
    else:
        if training_args.use_plw:
            trainer = PLWTrainer(**trainer_kwargs)
        else:
            trainer = SFTTrainer(**trainer_kwargs)

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    logger.info(f"Checkpoint detected, resuming training at {checkpoint}.")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        # "finetuned_from": model_args.model_name_or_path, # modify for sfttrainer latest update
        "model_name": model_args.model_name_or_path,
        "dataset_name": "\n".join(list(data_args.dataset_mixer.keys())),
        "tags": ["alignment-handbook"],
    }

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # torch.cuda.memory._dump_snapshot(Path(training_args.output_dir) / "GPU_RAM_PROFILE.pickle")
    # prof.close()
    logger.info("*** Training complete ***")


if __name__ == "__main__":
    # torch.cuda.memory._record_memory_history()
    main()
