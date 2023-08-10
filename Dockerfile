#Development container for this project

FROM nvcr.io/nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04

ARG local_uid
ARG local_user

RUN adduser --uid ${local_uid} --gecos "" --disabled-password ${local_user}
RUN usermod -aG sudo ${local_user}


RUN apt update && \
    apt upgrade -y && \
    apt install -y python3.11 python3-pip python3.11-dev curl git


RUN mkdir -p /home/${local_user}/workspace
WORKDIR /home/${local_user}/workspace

USER ${local_user}
ENV PATH="/home/${local_user}/.local/bin:${PATH}"

RUN pip install --upgrade pip 
RUN pip install poetry cuda-python

COPY ./docker-entrypoint.sh /home/${local_user}/entrypoint.sh
ENTRYPOINT /bin/bash $HOME/entrypoint.sh

