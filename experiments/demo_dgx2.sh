#!/usr/bin/bash 

ROOT=$(realpath ~)

# location
echo activate virtual ENV
PYTHON_ENV=${ROOT}/project/scripts/v2306.sh
source $PYTHON_ENV


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
export WANDB_API_KEY="05411100e08ac02e3fcbdc821b4116cf1c066e99"
# export WANDB_API_KEY="<key>"
export WANDB_USERNAME="xi-yang5"
export WANDB_PROJECT="demo_dgx2"
export WANDB_LOG_MODEL="false"
export WANDB_WATCH="false"

# TORCH and NCCL
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_NTHREADS=16

export ACCELERATE_LOG_LEVEL=debug 
export ACCELERATE_DEBUG_MODE="1"
export DEEPSPEED_TIMEOUT=120

# accelerate launch
accelerate launch \
    --config_file ${ROOT}/project/alignment_handbook/recipes/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes $WORLD_SIZE \
    --tee 3 \
   ${ROOT}/project/alignment_handbook/scripts/run_sft.py \
   ${ROOT}/project/alignment_handbook/recipes/llama3-8b/sft/config_qlora.yaml 
#    ${ROOT}/project/alignment_handbook/recipes/llama3-8b/sft/config_full.yaml 


# deepspeed launch


# torch launch