#!/usr/bin/bash
whoami
pwd

HOME=/home/l069561

ROOT=${HOME}/project/alignment-handbook


SCRIPTPATH=${ROOT}/experiments
source ${SCRIPTPATH}/wandb.sh

echo $SLURM_TMPDIR
export TMPDIR="/cache"
export HF_DATASETS_CACHE="${HOME}/cache/dataset"
export HF_HOME="${HOME}/cache/hf"
export TRITON_CACHE_DIR="/cache"


# TORCH and NCCL
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_NTHREADS=16
export DEEPSPEED_TIMEOUT=120

# export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES*$SLURM_NTASKS_PER_NODE))

echo $PRIMARY
echo $PRIMARY_PORT

torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE  \
    --nnode=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID  \
    --master_addr=$PRIMARY \
    --master_port=$PRIMARY_PORT \
    ${ROOT}/scripts/run_sft.py \
    ${ROOT}/recipes/llama3-8b/sft/config_qlora.yaml \
    --deepspeed=${ROOT}/recipes/accelerate_configs/deepspeed_zs2.json \
    --tee=2
