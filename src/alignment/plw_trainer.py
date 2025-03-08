import warnings

import datasets
import torch
from torch.nn import CrossEntropyLoss

from trl import SFTTrainer


def PLW_sample_chat_template():
    template = "{% if messages[0]['role'] == 'system' %}{% set system_message =  bos_token + '### System Instruction: ' + messages[0]['content'] | trim + '' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{ system_message }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Context: ' + message['content'] | trim + '\n' }}{% elif message['role'] == 'assistant' %}{{ '### Result: ' + message['content'] | trim + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '### Result: ' }}{% endif %}"

    return template


def PLW_apply_chat_template(example, tokenizer=None, use_sample_template=False):
    messages = example["messages"]
    prompts = list(filter(lambda x: x["role"] != "assistant", messages))
    labels = list(filter(lambda x: x["role"] == "assistant", messages))

    if use_sample_template:
        tokenizer.chat_template = PLW_sample_chat_template()

    example["prompt"] = tokenizer.apply_chat_template(
        prompts, tokenize=False, add_generation_prompt=False
    )
    example["completion"] = tokenizer.apply_chat_template(
        labels, tokenize=False, add_generation_prompt=False
    )
    return example


class PLWTrainer(SFTTrainer):
    def __init__(self, *args, prompt_loss_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.plw = prompt_loss_weight
        # need to add prompt_mask and completion_mask to dataset generation

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # get outputs without computing loss (by not passing in labels)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        logits = outputs.get("logits")
        labels = inputs.pop("labels")
        weights = self.plw * inputs["prompt_mask"] + inputs["completion_mask"]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()

        shift_labels = shift_labels.to(shift_logits.device)
        shift_weights = shift_weights.to(shift_logits.device, shift_logits.dtype)

        # Compute per-token losses
        loss_fct = CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # Compute weighted average of losses
        loss = token_losses @ shift_weights.view(-1) / shift_weights.sum()

        return (loss, outputs) if return_outputs else loss

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        packing,
        formatting_func,
        dataset_name,
    ):
        if dataset is None:
            raise ValueError("The dataset should not be None")

        # If the dataset is already preprocessed (tokenized), return as-is. Only works if dataset is
        # a datasets.Dataset or datasets.IterableDataset -- not for torch Dataset
        column_names = (
            dataset.column_names
            if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset))
            else None
        )
        if column_names and "input_ids" in column_names:
            if formatting_func is not None:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a valid formatting function. Therefore `formatting_func` will be ignored."
                )

            return dataset

        # check if torch dataset / dataloader and do nothing
        # see https://github.com/huggingface/trl/pull/1468 for why datasets.IterableDataset needs a separate check
        if isinstance(
            dataset,
            (torch.utils.data.IterableDataset, torch.utils.data.Dataset),
        ) and not isinstance(dataset, datasets.IterableDataset):
            return dataset

        self._dataset_sanity_checked = False

        def tokenize(element):
            prompts = element["prompt"]
            labels = element["completion"]

            input_ids = []
            attention_mask = []
            prompt_masks = []
            completion_masks = []

            for prmp, lab in zip(prompts, labels):
                p = processing_class(
                    prmp,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_overflowing_tokens=False,
                    return_length=True,
                )

                l = processing_class(
                    lab,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_overflowing_tokens=False,
                    return_length=True,
                )

                p_len = p["length"][0]
                l_len = l["length"][0]
                gap_len = p_len + l_len - processing_class.model_max_length

                new_input_ids = p["input_ids"] + l["input_ids"]
                new_attn_mask = p["attention_mask"] + l["attention_mask"]
                prompt_mask = p["attention_mask"] + [0] * l_len
                completion_mask = [0] * p_len + l["attention_mask"]

                # truncate from left side
                if gap_len > 0:
                    new_input_ids = new_input_ids[gap_len:]
                    new_attn_mask = new_attn_mask[gap_len:]
                    prompt_mask = prompt_mask[gap_len:]
                    completion_mask = completion_mask[gap_len:]

                input_ids.append(new_input_ids)
                attention_mask.append(new_attn_mask)
                prompt_masks.append(prompt_mask)
                completion_masks.append(completion_mask)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "prompt_mask": prompt_masks,
                "completion_mask": completion_masks,
            }

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=args.dataset_num_proc,
        )
