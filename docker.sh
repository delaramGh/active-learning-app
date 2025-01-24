#!/bin/bash

IMAGE_NAME="dghmh"
CONTAINER_NAME="streamlit-container"
PORT=8501

echo "Building the Docker image..."
docker build -t $IMAGE_NAME .

echo "Running the Docker container..."
docker run -v $PWD:/usr/src --rm -p $PORT:8501 -it --name $CONTAINER_NAME $IMAGE_NAME bash -l


