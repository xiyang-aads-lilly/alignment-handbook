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
import sys
from importlib.metadata import version as get_pkg_version
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed


p = Path(__file__).parent.parent / "src"
sys.path.append(p.as_posix())

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    ORPOTrainerPatch,
    apply_chat_template,
    fix_lr_scheduler_kwargs_float,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format


logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, ORPOConfig))
    model_args, data_args, training_args = parser.parse()

    # fix lr_scheduler_kwargs parse float to str
    fix_lr_scheduler_kwargs_float(training_args)

    # Truncate from left to ensure we don't lose labels in final turn
    data_args.truncation_side = "left"

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
            "prompt",
            "chosen",
            "rejected",
        ],
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    tokenizer = get_tokenizer(
        model_args, data_args, training_args, auto_set_chat_template=False
    )
    # training_args.max_length = tokenizer.model_max_length
    # training_args.max_prompt_length = training_args.max_prompt_length
    training_args.max_completion_length = (
        training_args.max_length - training_args.max_prompt_length
    )

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    peft_config = get_peft_config(model_args)

    # For ChatML we need to add special tokens and resize the embedding layer
    if "<|im_start|>" in tokenizer.chat_template:
        model, tokenizer = setup_chat_format(model, tokenizer)

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "orpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    #############################
    # Filter out seq > max_length
    # #############################
    max_cls = []
    max_pls = []

    def check_length(sample):
        prompt_length = tokenizer(
            sample["text_prompt"],
            return_tensors="pt",
        )[
            "input_ids"
        ].size(dim=-1)

        chosen_length = tokenizer(
            sample["text_chosen"],
            return_tensors="pt",
        )[
            "input_ids"
        ].size(dim=-1)

        max_pls.append(prompt_length), max_cls.append(chosen_length)

        if prompt_length > training_args.max_prompt_length:
            logger.warning(
                f"{'*' * 20}\n\nSample {sample['text_prompt']}'s prompt length is greater than max prompt length {training_args.max_prompt_length}\n\n{'*' * 20}"
            )

        if chosen_length > training_args.max_completion_length:
            logger.warning(
                f"{'*' * 20}\n\nSample {sample['text_chosen']}'s chosen length is greater than max prompt length {training_args.max_prompt_length}\n\n{'*' * 20}"
            )

    raw_datasets.filter(check_length)

    # Replace column names with what TRL needs, text_prompt -> prompt, text_chosen -> chosen and text_rejected -> rejected
    for split in raw_datasets.keys():
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {
                "text_prompt": "prompt",
                "text_chosen": "chosen",
                "text_rejected": "rejected",
            }
        )

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(raw_datasets["train"])), 3):
    #     logger.info(
    #         f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}"
    #     )
    #     logger.info(
    #         f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}"
    #     )
    #     logger.info(
    #         f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}"
    #     )

    ##########################
    # Instantiate ORPO trainer
    ##########################
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if "test" in raw_datasets else None,
        peft_config=peft_config,
        # callbacks=[GpuUtilPrintCallBack()],
    )

    if get_pkg_version("transformers") >= "5.0.0" or get_pkg_version("trl") >= "0.15.0":
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    # trainer = ORPOTrainer(**trainer_kwargs)
    # TODO: check if TRL fix the ORPO trainer clip issue which can cause overflow error
    # IF not, use the ORPOTrainerPatch
    trainer = ORPOTrainerPatch(**trainer_kwargs)

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    checkpoint = None
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
    if training_args.do_eval and "test" in raw_datasets:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
