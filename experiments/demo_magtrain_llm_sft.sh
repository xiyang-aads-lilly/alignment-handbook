#!/usr/bin/bash
whoami
pwd
ds_report

echo $LD_LIBRARY_PATH

HOME=/home/l069561

ROOT=${HOME}/project/alignment-handbook

SCRIPTPATH=${ROOT}/experiments
source ${SCRIPTPATH}/wandb.sh

echo $SLURM_TMPDIR
export TMPDIR="/cache"

export TRITON_HOME=${HOME}/project/cache/triton
export TRITON_CACHE_DIR=${HOME}/project/cache/triton/cache
export TRITON_DUMP_DIR=${HOME}/project/cache/triton/dump
export HF_DATASETS_CACHE=${HOME}/project/cache/dataset
export HF_HOME=${HOME}/project/cache/huggingface

# TORCH and NCCL
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_DEBUG=INFO
export NCCL_SOCKET_NTHREADS=16
export DEEPSPEED_TIMEOUT=120

echo $PRIMARY
echo $PRIMARY_PORT

# TRAIN_CONF=${ROOT}/recipes/llama3-8b/sft/config_full.yaml
# TRAIN_CONF=${ROOT}/recipes/phi3/sft/config_full.yaml
# TRAIN_CONF=${ROOT}/recipes/qwen/sft/config_full.yaml
# TRAIN_CONF=${ROOT}/recipes/falcon_mamba/sft/config_full.yaml        # need futher debug, training stuck

# manually set
export WANDB_PROJECT="sang"
# TRAIN_CONF=${ROOT}/recipes/sang_project/config_full_1.yaml
TRAIN_CONF=${ROOT}/recipes/sang_project/config_full_2.yaml

DEEPSPEED_CONF=${ROOT}/recipes/accelerate_configs/deepspeed_zs2.json

torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE  \
    --nnode=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID  \
    --master_addr=$PRIMARY \
    --master_port=$PRIMARY_PORT \
    ${ROOT}/scripts/run_sft.py \
    $TRAIN_CONF \
    --deepspeed=$DEEPSPEED_CONF \
    --tee=2
