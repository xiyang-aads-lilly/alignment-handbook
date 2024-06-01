## Accelerate launch only support partial parameters in deepspeed
- to avoid, we need to launch with deepspeed not accelerate

## more info HF-deepspeed integration
- https://huggingface.co/docs/transformers/deepspeed?zero-config=ZeRO-2

## deepspeed optimizers
- DeepSpeed natively supports Adam, AdamW, OneBitAdam, Lamb, OneBitLamb, FusedLamb, FusedAdam
- see for details on how to config https://deepspeed.readthedocs.io/en/latest/optimizers.html
```json
// You can set the parameters to "auto" or manually input your own desired values.
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": "auto",
         "betas": "auto",
         "eps": "auto",
         "weight_decay": "auto"
       }
   }
}

{
  "optimizer": {
    "type": "OneBitLamb",
    "params": {
      "lr": 1e-3,
      "weight_decay": 0.01,
      "bias_correction": false,
      "max_coeff": 0.3,
      "min_coeff": 0.01,
      "freeze_step": 1000,
      "cuda_aware": false,
      "comm_backend_name": "nccl",
      "coeff_beta": 0.9,
      "factor_max": 4.0,
      "factor_min": 0.5,
      "factor_threshold": 0.1
    }
  }
}

{
  "optimizer": {
    "type": "Lamb",
    "params": {
      "lr": 1e-3,
      "weight_decay": 0.01,
      "bias_correction": false,
      "max_coeff": 0.3,
      "min_coeff": 0.01
    }
  }
}
```

- fp16 vs bf16
```json
{
    "fp16": {
        "enabled": false,
        "loss_scale": 0,
        "auto_cast": false,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "consecutive_hysteresis": false,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": true
    }
}
```

- offload
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true,
        "round_robin_gradients": true
    }
}

{
  "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        },
        "aio": {
            "block_size": 262144,
            "queue_depth": 32,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
}
```

- communication data type
> Choosing fp32 adds a small amount of overhead but ensures the reduction operation is accumulated in fp32 and when it is ready, it is downcasted to whichever half-precision dtype you’re training in.
> Default is fp16 if you use AMP.
```json
{ "communication_data_type": "fp32"}
```

- launch

```sh
deepspeed --num_gpus=2 examples/pytorch/translation/run_translation.py \
  --deepspeed tests/deepspeed/ds_config_zero3.json \
...


torchrun --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=hostname1 --master_port=9901 \
  your_program.py <normal cl args> \
  --deepspeed ds_config.json
```
