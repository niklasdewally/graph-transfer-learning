FROM nvcr.io/nvidia/pytorch:21.09-py3

RUN git clone 'https://github.com/niklasdewally/EGI' egi

ENV PATH=$CONDA_DIR/bin:$PATH

RUN cd egi && \
    python3 -m pip install --user --upgrade pip && \
    python3 -m pip install tqdm  dgl-cu110==0.5.3

WORKDIR egi
