WORLD_SIZE=$(($SLURM_JOB_NUM_NODES*$SLURM_GPUS_PER_TASK))

# HF cache
export TMPDIR="~/ctmp"
export HF_DATASETS_CACHE="~/ctmp"
export HF_HOME="~/ctmp"
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export NCCL_SOCKET_NTHREADS=16

# Global environment constants recommended by Nvidia.
EXCLUDE_IB_LIST=mlx5_4,mlx5_5,mlx5_10,mlx5_11
export NCCL_IB_HCA=^${EXCLUDE_IB_LIST}
export NCCL_SOCKET_IFNAME=bridge-1145
export NCCL_ASYNC_ERROR_HANDLING=1

echo $PRIMARY:$PRIMARY_PORT:$SLURM_PROCID
echo $SLURM_JOB_NUM_NODES:$SLURM_GPUS_PER_TASK:$WORLD_SIZE
echo $SLURM_NODEID 

# ACCELERATE_LOG_LEVEL=info accelerate launch \
#     --config_file /red/gatortron-phi/workspace/zzz/alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml \
#     --main_process_ip $PRIMARY \
#     --main_process_port $PRIMARY_PORT \
#     --machine_rank $SLURM_PROCID \
#     --num_machines $SLURM_JOB_NUM_NODES \
#     --num_processes $WORLD_SIZE \
#     --tee 3 \
#     /red/gatortron-phi/workspace/zzz/alignment-handbook/scripts/run_sft.py \
#     /red/gatortron-phi/workspace/zzz/alignment-handbook/recipes/zephyr-7b-beta/sft/config_full.yaml

# --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$PRIMARY:$PRIMARY_PORT" \
export ACCELERATE_LOG_LEVEL=debug 
export ACCELERATE_DEBUG_MODE="1"
export DEEPSPEED_TIMEOUT=120

# pip install -U git+https://github.com/huggingface/trl

# export WANDB_PROJECT="alignment"
# export WANDB_WATCH="parameters"
# export WANDB_API_KEY="KEY"
# export WANDB_USERNAME="copy-o0o-paste"

accelerate launch \
    --config_file ~/alignment-handbook/recipes/accelerate_configs/deepspeed_zero2.yaml \
    --main_process_ip $PRIMARY \
    --main_process_port $PRIMARY_PORT \
    --machine_rank $SLURM_PROCID \
    --num_machines $SLURM_JOB_NUM_NODES \
    --num_processes $WORLD_SIZE \
    --tee 3 \
   ~/alignment-handbook/scripts/run_sft.py \
   ~/alignment-handbook/recipes/zephyr-7b-beta/sft/config_full.yaml

# ACCELERATE_LOG_LEVEL=info accelerate launch \
#     --config_file ~/alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml \
#     --main_process_ip $PRIMARY \
#     --main_process_port $PRIMARY_PORT \
#     --machine_rank $SLURM_PROCID \
#     --num_machines $SLURM_JOB_NUM_NODES \
#     --num_processes $WORLD_SIZE \
#     --tee 3 \
#     ~/alignment-handbook/scripts/run_sft.py \
#     ~/alignment-handbook/recipes/zephyr-7b-beta/sft/config_lora.yaml



#launch with deepspeed
# deepspeed \
#     --num_gpus $WORLD_SIZE \
#     --num_nodes $SLURM_JOB_NUM_NODES \
#     --master_addr \
#     --master_port=9901 \
#     --deepspeed --deepspeed_config ds_config.json