#!/bin/bash

# Script to be run on docker container launch
# Assumes that the project root is mounted to /workspace.

cd /workspace

export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/targets/x86_64-linux/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-11.7


poetry install 
# pass through executable
poetry run --  $@
