#!/bin/bash
#SBATCH --job-name=gpt2_50M_distributed
#SBATCH --nodes=2                   # Number of nodes
#SBATCH --output=logs/%x_%j.out     # logfile for stdout/stderr
#SBATCH --exclusive                 # Job has exclusive use of the resource, no sharing
#SBATCH --wait-all-nodes=1          # Wait for all nodes before starting job


set -ex;


###########################
###### User Variables #####
###########################

GPUS_PER_NODE=4 # 4 for G5.12x, 8 for P4/P5

###########################
## Environment Variables ##
###########################
# Disclaimer  most of this config variables are not necessary anymore 
# See https://github.com/Stability-AI/hyperpod/blob/main/1.architectures/efa-cheatsheet.md


## Plenty of EFA level variables
## Comment out for non-efa instances (G4d, P3)
## For G5.12x, Comment out RDMA and Fork safe
## For G4dn and other G5, comment out all
# export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d/P5s
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_PROVIDER=efa
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # For G5s

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

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nanogpt

srun -l torchrun "${TORCHRUN_ARGS[@]}" train.py config/train_gpt2_baby_dist.py