#!/bin/bash

IMG_NAME=${1:-nv-pt}
IMG_TAG=${2:-latest}

docker build -t $IMG_NAME:$IMG_TAG .
enroot import -o ~/.cache/enroot/${IMG_NAME}.sqsh dockerd://${IMG_NAME}:${IMG_TAG} 
