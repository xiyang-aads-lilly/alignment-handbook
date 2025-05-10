# DEMO

## data format
- save data use parquet format
- the actual data should be saved as list of dict under `messages` column
- see `https://huggingface.co/datasets/HuggingFaceH4/SystemChat?row=0` as example

sample code for data process:
```python
from datasets import load_dataset, Dataset

# prepare data and save to local
my_data = [
    {
        "messages": [
            {"role": "system", "content": "This is system msg1"},
            {"role": "user", "content": "This is user msg1"},
            {"role": "assistant", "content": "This is assistant msg1"},
        ]
    },
     {
        "messages": [
            {"role": "system", "content": "This is system msg2"},
            {"role": "user", "content": "This is user msg2"},
            {"role": "assistant", "content": "This is assistant msg2"},
        ]
    },
]

Dataset.from_list(my_data).to_parquet("my_dataset/train/data.parquet")
Dataset.from_list(my_data).to_parquet("my_dataset/test/data.parquet")

# load
ds = load_dataset("my_dataset")
print(ds)
"""
DatasetDict({
    train: Dataset({
        features: ['messages'],
        num_rows: 2
    })
    test: Dataset({
        features: ['messages'],
        num_rows: 2
    })
})
"""
```

## config
- accelerate configuration is at `recipes/accelerate_configs`
    - recommend to use deepspeed config to launch experiment
- specific project config examples can be found under `recipes` asscociated with model names
- see `recipes/llama3-8b` for config example on `sft`, `dpo`, `orpo`


## What we support
- SFT (include qlora)
- DPO (include qlora)
- ORPO

## special trainer
- we support a new trainer called prompt loss token weights (PLW) at `alginment/plw_trainer` for sft, we find it is better than naive sft trainer. ref: https://arxiv.org/abs/2401.13586
- we also support dpo-shift trainer ref: https://arxiv.org/html/2502.07599v1

## TODO:
- GRPO pipeline support

## deepspeed
- we use deepspeed to control distrbuted training
- check `recipes/accelerate_configs/deepspeed_zs*.json` for different stage config
- Commonly we use stage2: deepspeed_zs2.json
- since v0.16.4, HF auto tensor parallel is supported: https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/huggingface-tp/README.md
- see deepspeed_zs2_with_tp.json for auto tensor parallel usage
