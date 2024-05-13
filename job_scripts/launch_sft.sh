#!/bin/bash

#SBATCH --job-name=alignment
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1024gb
#SBATCH --time=48:00:00
#SBATCH --output=/red/gatortron-phi/workspace/zzz/log/sft_%j.out
#SBATCH --partition=hpg-ai
#SBATCH --exclusive
#SBATCH --reservation=gatortrongpt

# default setup
module load apptainer
CONTAINER=/red/gatortron-phi/workspace/containers/alignment.sif

source /red/gatortron-phi/workspace/zzz/alignment-handbook/recipes/util.sh

# sft replicate HF model 
srun --jobid $SLURM_JOB_ID singularity exec --nv $CONTAINER bash /red/gatortron-phi/workspace/zzz/alignment-handbook/recipes/run_sft.sh
