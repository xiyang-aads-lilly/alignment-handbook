#!/bin/bash

#SBATCH --job-name=llm_sft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xi.yang5@lilly.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=512gb
#SBATCH --time=48:00:00
#SBATCH --output=/home/l069561/project/log/alignment/sft_%j.out
#SBATCH --partition=batch

HOME=/home/l069561
SCRIPTPATH=${HOME}/project/alignment-handbook/experiments

echo $SCRIPTPATH
echo $SLURM_NTASKS_PER_NODE
echo $SLURM_JOB_NUM_NODES
echo $SLURM_GPUS_ON_NODE
source ${SCRIPTPATH}/util.sh

# CONTAINER=${HOME}/container/pt2402.sif
CONTAINER=${HOME}/container/pt2402

export TRITON_HOME=${HOME}/project/cache/triton
export TRITON_CACHE_DIR=${HOME}/project/cache/triton/cache
export TRITON_DUMP_DIR=${HOME}/project/cache/triton/dump
export HF_DATASETS_CACHE=${HOME}/project/cache/dataset
export HF_HOME=${HOME}/project/cache/huggingface

# srun --jobid $SLURM_JOB_ID apptainer exec -B $SLURM_TMPDIR:/cache  --nv $CONTAINER bash ${SCRIPTPATH}/demo_magtrain_llm_sft.sh

# use nsys to profile training process
srun --jobid $SLURM_JOB_ID \
    apptainer exec -B $SLURM_TMPDIR:/cache --nv --fakeroot $CONTAINER \
    nsys profile  -s none -t cuda,nvtx \
    --gpu-metrics-device=all \
    --gpu-metrics-frequency=100 \
    --nic-metrics=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -o $SLURM_TMPDIR/nsys_${SLURM_JOB_ID} \
    bash ${SCRIPTPATH}/demo_magtrain_llm_sft.sh

cp $SLURM_TMPDIR/nsys_${SLURM_JOB_ID}.nsys-rep ${HOME}/project/log/nsys/
