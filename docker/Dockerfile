# devel:v2
# Development container for this project
# Automatically runs a jupyter notebook server

FROM nvcr.io/nvidia/pytorch:21.09-py3

ENTRYPOINT ["/bin/bash","-c"]

ENV PATH=$CONDA_DIR/bin:$PATH

RUN mkdir -p /workspace/mount

WORKDIR /workspace

RUN pip install --upgrade pip

# dgi version
RUN python3 -m pip install --user --upgrade pip && \
    python3 -m pip install tqdm dgl-cu110==0.5.3

# Jupyter extensions
RUN pip install ipywidgets
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user

# Pip packages
RUN pip install plotly scipy igraph networkx pyvis

CMD ["jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password=''"]

