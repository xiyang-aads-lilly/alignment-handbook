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
import logging
import random
import sys
from importlib.metadata import version as get_pkg_version
from pathlib import Path


p = Path(__file__).parent.parent / "src"
sys.path.append(p.as_posix())

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (  # decontaminate_humaneval,
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    ProfCallback,
    apply_chat_template,
    fix_lr_scheduler_kwargs_float,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftModel
from trl import DPOTrainer


logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    # fix lr_scheduler_kwargs parse float to str
    fix_lr_scheduler_kwargs_float(training_args)

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

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
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = (
        "left"  # Truncate from left to ensure we don't lose labels in final turn
    )

    tokenizer = get_tokenizer(
        model_args, data_args, training_args, auto_set_chat_template=False
    )
    # sync tokenizer max_length with model max_length
    training_args.max_length = tokenizer.model_max_length

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "dpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {
                "text_prompt": "prompt",
                "text_chosen": "chosen",
                "text_rejected": "rejected",
            }
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(
            f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}"
        )
        logger.info(
            f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}"
        )
        logger.info(
            f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}"
        )

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

    # peft_config = get_peft_config(model_args)
    peft_config = None
    if is_adapter_model(model, model_args.model_revision) is True:
        # Load the base model, merge the adapter weights and unload the adapter
        # Note: to run QLoRA, you will need to merge the base model separately as the merged model in 16bit
        logger.info(f"Merging PEFT adapters for {model_args.model_name_or_path}")

        # peft_config = PeftConfig.from_pretrained(
        #     model_args.model_name_or_path, revision=model_args.model_revision
        # )
        peft_config = get_peft_config(model_args)

        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=(
                get_kbit_device_map() if quantization_config is not None else None
            ),
            quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    ##################
    # PYTORCH profiler
    ##################
    # prof = torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=2, repeat=2),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #         dir_name=training_args.logging_dir
    #     ),
    #     profile_memory=True,
    #     with_stack=True,
    #     record_shapes=True,
    #     with_flops=True,
    #     with_modules=True,
    # )

    #########################
    # Instantiate DPO trainer
    #########################
    training_args.model_init_kwargs = model_kwargs
    training_args.ref_model_init_kwargs = ref_model_kwargs

    trainer_kwargs = dict(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        peft_config=peft_config,
        # callbacks=[GpuUtilPrintCallBack()],
    )

    if get_pkg_version("transformers") >= "5.0.0" or get_pkg_version("trl") >= "0.16.0":
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**trainer_kwargs)

    ###############
    # Training loop
    ###############
    checkpoint = None

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)

    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
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
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # torch.cuda.memory._dump_snapshot(
    #     Path(training_args.output_dir) / "GPU_RAM_PROFILE.pickle"
    # )
    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    # torch.cuda.memory._record_memory_history()
    main()
