#!/bin/bash
#SBATCH --job-name=gpt2_50M_standalone
#SBATCH --nodes=1               # Number of nodes
#SBATCH --output=out.log        # Standard output log
#SBATCH --error=err.log         # Error log
#SBATCH --exclusive             # Job has exclusive use of the resource, no sharing

set -ex;


###########################
###### User Variables #####
###########################

GPUS_PER_NODE=4 # 4 for G5.12x, 8 for P4/P5

###########################
## Environment Variables ##
###########################

## Plenty of EFA level variables
## Comment out for non-efa instances (G4d, P3)
## For G5.12x, Comment out RDMA and Fork safe
## For G4dn and other G5, comment out all
# export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
# export FI_EFA_FORK_SAFE=1
# export FI_LOG_LEVEL=1
# export FI_PROVIDER=efa
# export NCCL_DEBUG=INFO

###########################
####### Torch Dist  #######
###########################

declare -a TORCHRUN_ARGS=(
    --standalone \
    --nproc_per_node=$GPUS_PER_NODE \
    # --nnodes=$SLURM_JOB_NUM_NODES \
    # --rdzv_id=$SLURM_JOB_ID \
    # --rdzv_backend=c10d \
    # --rdzv_endpoint=$(hostname) \
)

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nanogpt

srun -l torchrun "${TORCHRUN_ARGS[@]}" train.py config/train_gpt2_baby.py