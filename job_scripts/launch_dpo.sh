#!/bin/bash

#SBATCH --job-name=alignment
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=2000gb
#SBATCH --time=72:00:00
#SBATCH --output=/red/gatortron-phi/workspace/zzz/log/dpo_%j.out
#SBATCH --partition=hpg-ai
#SBATCH --reservation=gatortrongpt

# default setup
module load apptainer
# CONTAINER=/red/gatortron-phi/workspace/containers/hfacc.sif
CONTAINER=/red/gatortron-phi/workspace/containers/alignment.sif

get_unused_port() {
    # Well-known ports end at 1023.  On Linux, dynamic ports start at 32768
    # (see /proc/sys/net/ipv4/ip_local_port_range).
    local MIN_PORT=10001
    local MAX_PORT=32767

    local USED_PORTS=$(netstat -a -n -t | tail -n +3 | tr -s ' ' | \
        cut -d ' ' -f 4 | sed 's/.*:\([0-9]\+\)$/\1/' | sort -n | uniq)

    # Generate random port numbers within the search range (inclusive) until we
    # find one that isn't in use.
    local RAN_PORT
    while
        RAN_PORT=$(shuf -i 10001-32767 -n 1)
        [[ "$USED_PORTS" =~ $RAN_PORT ]]
    do
        continue
    done

    echo $RAN_PORT
}

init_node_info() {
    export PRIMARY=$(hostname -s) #$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    SECONDARIES=$(scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $PRIMARY)
    ALL_NODES="$PRIMARY $SECONDARIES"
    export PRIMARY_PORT=$(get_unused_port)
}

init_node_info

# HF cache
export TMPDIR="/blue/yonghui.wu/alexgre/ctmp"
export HF_DATASETS_CACHE="/blue/yonghui.wu/alexgre/ctmp"
export HF_HOME="/blue/yonghui.wu/alexgre/ctmp"

# Global environment constants recommended by Nvidia.
EXCLUDE_IB_LIST=mlx5_4,mlx5_5,mlx5_10,mlx5_11
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=^${EXCLUDE_IB_LIST}
export NCCL_SOCKET_IFNAME=bridge-1145
export NCCL_ASYNC_ERROR_HANDLING=1

# other configs
WORLD_SIZE=$(($SLURM_JOB_NUM_NODES*$SLURM_GPUS_PER_TASK))

# HF cache
export TMPDIR="/blue/yonghui.wu/alexgre/ctmp"
export HF_DATASETS_CACHE="/blue/yonghui.wu/alexgre/ctmp"
export HF_HOME="/blue/yonghui.wu/alexgre/ctmp"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=DEBUG
export ACCELERATE_LOG_LEVEL=DEBUG 
export ACCELERATE_DEBUG_MODE="1"
export DEEPSPEED_TIMEOUT=360

echo $PRIMARY:$PRIMARY_PORT:$SLURM_PROCID
echo $SLURM_JOB_NUM_NODES:$SLURM_GPUS_PER_TASK:$WORLD_SIZE
echo $SLURM_NODEID 

# sft replicate HF model 
srun --jobid $SLURM_JOB_ID singularity exec --nv $CONTAINER bash /red/gatortron-phi/workspace/zzz/alignment-handbook/recipes/run_dpo.sh 2>&1