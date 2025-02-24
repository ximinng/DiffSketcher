#!/bin/bash

set -e

# Docker image and container name
IMAGE_NAME="ximingxing/svgrender:v1"
CONTAINER_NAME="diffsketcher"

echo "==== 1. Pulling Docker image ===="
docker pull $IMAGE_NAME

echo "==== 2. Starting Docker container ===="
# Check if a container with the same name exists, and remove it if necessary
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Existing container found, removing old container..."
    docker rm -f $CONTAINER_NAME
fi

# Run the Docker container with GPU support and mount the current directory
docker run --name $CONTAINER_NAME --gpus all -it --ipc=host -v $(pwd):/workspace $IMAGE_NAME /bin/bash -c "
    echo '==== 3. Activating Conda environment ===='
    source /opt/conda/etc/profile.d/conda.sh
    conda activate svgrender

    echo '==== 4. Installing missing dependencies ===='
    pip install cairosvg
    apt update && apt install -y libcairo2-dev

    echo '==== 5. Running DiffSketcher example ===='
    python run_painterly_render.py \
      -c diffsketcher.yaml \
      -eval_step 10 -save_step 10 \
      -update 'token_ind=4 num_paths=96 num_iter=800' \
      -pt 'a photo of Sydney opera house' \
      -respath ./workdir/sydney_opera_house \
      -d 8019 \
      --download
"

echo "==== DiffSketcher execution completed! ===="
