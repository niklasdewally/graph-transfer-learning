#Development container for this project

FROM nvcr.io/nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04

RUN apt update && \
    apt upgrade -y && \
    apt install -y python3.11 python3-pip python3.11-dev curl git vim

RUN mkdir /workspace
WORKDIR /workspace

RUN pip install --upgrade pip 
RUN pip install poetry cuda-python

COPY ./docker-entrypoint.sh /entrypoint.sh
ENTRYPOINT /bin/bash /entrypoint.sh

