## deepspeed optimizers
- DeepSpeed natively supports Adam, AdamW, OneBitAdam, Lamb, OneBitLamb, FusedLamb, FusedAdam
- see for details on how to config https://deepspeed.readthedocs.io/en/latest/optimizers.html
- 
```json
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
  },
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
  },
}
```