
FROM nvcr.io/nvidia/pytorch:21.09-py3

RUN mkdir -p /workspace

WORKDIR /workspace

RUN git clone 'https://github.com/niklasdewally/EGI' /workspace/egi

ENV PATH=$CONDA_DIR/bin:$PATH

RUN python3 -m pip install --user --upgrade pip && \
    python3 -m pip install tqdm  dgl-cu110==0.5.3

