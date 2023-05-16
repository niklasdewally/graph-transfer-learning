# Development container for this project
# Automatically runs a jupyter notebook server

FROM nvcr.io/nvidia/cuda:11.7.0-runtime-ubuntu22.04

RUN apt update && \
    apt upgrade -y && \
    apt install -y python3.10 python3-pip

RUN mkdir /workspace
WORKDIR /workspace

COPY ./docker-entrypoint.sh /entrypoint.sh
COPY ./requirements.txt /requirements.txt
ENTRYPOINT ["/bin/bash","/entrypoint.sh"]


RUN pip install --upgrade pip
RUN pip install jupyter_contrib_nbextensions

RUN pip install -r /requirements.txt
RUN pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html &&\
    pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html

# Jupyter extensions
RUN jupyter contrib nbextension install --user

# CMD ["jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password=''"]
CMD ["bash"]

