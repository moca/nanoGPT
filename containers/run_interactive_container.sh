#!/bin/bash

# Function to display usage/help
usage() {
  echo "Usage: $0 [--cmd=<command>] [--img_name=<image_name>] [--img_tag=<tag>]"
  echo
  echo "Options:"
  echo "  --cmd       Command to run inside the container (default: /bin/bash)"
  echo "  --img_name  Docker image name (default: nv-pt)"
  echo "  --img_tag   Docker image tag (default: latest)"
  echo "  -h, --help  Display this help message"
  exit 1
}

# Default values
CMD="/bin/bash"
IMG_NAME="nv-pt"
IMG_TAG="latest"

# Parse named arguments
while [[ "$1" != "" ]]; do
  case $1 in
    --cmd=*) CMD="${1#*=}" ;;
    --img_name=*) IMG_NAME="${1#*=}" ;;
    --img_tag=*) IMG_TAG="${1#*=}" ;;
    -h | --help) usage ;;  # Display help and exit
    *) echo "Unknown option: $1"; usage ;;  # Handle unknown options
  esac
  shift
done

# Validate that the image name is not empty
if [ -z "$IMG_NAME" ]; then
  echo "Error: Image name cannot be empty."
  usage
fi

# Run the Docker container with the specified or default options
docker run \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --rm \
  -it "${IMG_NAME}:${IMG_TAG}" \
  ${CMD}
