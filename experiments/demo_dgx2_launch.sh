#!/usr/bin/bash 

ROOT=$(realpath ~)

# singularity container
CONTAINER=${ROOT}/project/singularity_containers/py2402.sig

# CUDA
export CUDA_VISIBLE_DEVICES=0,1

# PATH
DEMO_PATH=${ROOT}/project/alignment_handbook/experiments

# launch
singularity exec --nv $CONTAINER bash ${DEMO_PATH}/demo_dgx2.sh