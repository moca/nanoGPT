#!/bin/bash
#SBATCH --job-name=gpt2_50M_dist_container
#SBATCH --nodes=2                   # Number of nodes
#SBATCH --output=logs/%x_%j.out     # logfile for stdout/stderr
#SBATCH --exclusive                 # Job has exclusive use of the resource, no sharing
#SBATCH --wait-all-nodes=1          # Wait for all nodes before starting job

set -ex;

###########################
###### User Variables #####
###########################

GPUS_PER_NODE=4 # 4 for G5.12x, 8 for P4/P5
IMAGE=~/.cache/enroot/nv-pt.sqsh
WORKDIR=/workspace
DATA_MOUNT=$(pwd):${WORKDIR}


###########################
## Environment Variables ##
###########################
# Disclaimer  most of this config variables are not necessary anymore 
# See https://github.com/Stability-AI/hyperpod/blob/main/1.architectures/efa-cheatsheet.md

export FI_LOG_LEVEL=1
export FI_PROVIDER=efa
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # For G5s
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1


###########################
####### Torch Dist  #######
###########################

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname) \
)

#########################
## Command and Options ##
#########################

declare -a ARGS=(
    --container-image $IMAGE
    --container-mounts $DATA_MOUNT
)

srun -l "${ARGS[@]}" torchrun "${TORCHRUN_ARGS[@]}" train.py config/train_gpt2_baby_dist.py