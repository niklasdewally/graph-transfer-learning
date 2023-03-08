FROM nvcr.io/nvidia/pytorch:21.09-py3


#RUN apt update 
#
#RUN apt install -y python3-pip git gpg wget
#
## Install miniconda
ENV CONDA_DIR /opt/conda
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda
#
RUN git clone 'https://github.com/niklasdewally/EGI' repository
#
ENV PATH=$CONDA_DIR/bin:$PATH


#https://pythonspeed.com/articles/activate-conda-dockerfile/
#RUN conda create --name myenv 
#SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

RUN cd repository && \
    python3 -m pip install --user --upgrade pip && \
    python3 -m pip install tqdm  dgl-cu110==0.5.3


#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "bash"]

