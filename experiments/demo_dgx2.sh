#!/usr/bin/bash

ROOT=$(realpath ~)

# location
echo activate virtual ENV
PYTHON_ENV=${ROOT}/project/scripts/v2306.sh
source $PYTHON_ENV

# get this script location
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

# CUDA
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING="1"

# number of GPUs; here we use all GPUs for demo
WORLD_SIZE=2

# HF cache
export TMPDIR="${ROOT}/project/.cache/"
export HF_DATASETS_CACHE="${ROOT}/project/.cache/dataset"
export HF_HOME="${ROOT}/project/.cache/"

# Wandb
source ${SCRIPTPATH}/wandb.sh

# TORCH and NCCL
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_NTHREADS=16

export ACCELERATE_LOG_LEVEL=debug
export ACCELERATE_DEBUG_MODE="1"
export DEEPSPEED_TIMEOUT=120

# accelerate launch
# accelerate launch \
#     --config_file ${ROOT}/project/alignment_handbook/recipes/accelerate_configs/deepspeed_zero2.yaml \
#     --num_processes $WORLD_SIZE \
#     --tee 3 \
#    ${ROOT}/project/alignment_handbook/scripts/run_sft.py \
#    ${ROOT}/project/alignment_handbook/recipes/llama3-8b/sft/config_qlora.yaml


# torch launch
# source ${SCRIPTPATH}/util.sh
# --master_addr=$PRIMARY --master_port=$PRIMARY_PORT
# python -m torch.distributed.run

# need to add virtual env package path as PYTHONPATH
export PYTHONPATH=${ROOT}/project/pyenv/2306/lib/python3.10/site-packages
torchrun --nproc_per_node=$WORLD_SIZE --nnode=1 --node_rank=0 \
  ${ROOT}/project/alignment_handbook/scripts/run_sft.py \
  ${ROOT}/project/alignment_handbook/recipes/llama3-8b/sft/config_qlora.yaml \
  --deepspeed=${ROOT}/project/alignment_handbook/recipes/accelerate_configs/deepspeed_zs2.json \
  --tee=2 >> ${SCRIPTPATH}/log.txt

# python -m torch.distributed.run --nproc_per_node=$WORLD_SIZE --nnode=1 --node_rank=0 \
#   ${ROOT}/project/alignment_handbook/scripts/run_sft.py \
#   ${ROOT}/project/alignment_handbook/recipes/llama3-8b/sft/config_qlora.yaml \
#   --deepspeed=${ROOT}/project/alignment_handbook/recipes/accelerate_configs/deepspeed_zs2.json \
#   --tee=2
