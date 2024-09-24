#!/bin/bash

GPUS_PER_NODE=4

docker run \
  --gpus all \
  -it \
  --rm \
  -v .:/workspace \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nv-pt \
  torchrun --standalone --nproc_per_node=$GPUS_PER_NODE train.py config/train_gpt2_baby.py