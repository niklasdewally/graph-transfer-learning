# devel:v2
# Development container for this project
# Automatically runs a jupyter notebook server

FROM nvcr.io/nvidia/pytorch:22.12-py3

ENTRYPOINT ["/bin/bash","-c"]

ENV PATH=$CONDA_DIR/bin:$PATH

RUN mkdir -p /workspace/mount

WORKDIR /workspace

RUN pip install --upgrade pip
RUN pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html && \
    pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

RUN pip install plotly scipy igraph networkx tqdm numpy jupyter_contrib_nbextensions

# Jupyter extensions
RUN jupyter contrib nbextension install --user

CMD ["jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password=''"]

