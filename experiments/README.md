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

## ENV
- update the singulairty container path in `demo_dgx2_launcher.sh`
- update python virtual environment path in `demo_dgx2.sh`

## config
- accelerate configuration is at `recipes/accelerate_configs`
    - recommend to use deepspeed config to launch experiment
- llama3-8b experiment config is under `recipes/llama3-8b`
- you can put config whereever you like