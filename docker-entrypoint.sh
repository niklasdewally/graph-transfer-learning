#!/bin/bash

# Script to be run on docker container launch
# Assumes that the project root is mounted to /workspace.

cd /workspace

pip install -e .

# pass through executable
exec $@
