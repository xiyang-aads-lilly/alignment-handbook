#!/bin/bash

#SBATCH --job-name=llm_sft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xi.yang5@lilly.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=80
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

CONTAINER=${HOME}/container/pt2402.sif

# srun --jobid $SLURM_JOB_ID apptainer exec -B $SLURM_TMPDIR:/cache  --nv $CONTAINER bash ${SCRIPTPATH}/demo_magtrain_llm_sft.sh

# use nsys to profile training process
srun --jobid $SLURM_JOB_ID apptainer exec -B $SLURM_TMPDIR:/cache  --nv $CONTAINER nsys profile -t cuda,nvtx -o /cache/nsys_${SLURM_JOB_ID} bash ${SCRIPTPATH}/demo_magtrain_llm_sft.sh
cp $SLURM_TMPDIR/nsys_${SLURM_JOB_ID}.nsys-rep ${HOME}/project/log/nsys/
