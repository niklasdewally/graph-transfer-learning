# devel:v1
# Development container for this project
# Automatically runs a jupyter notebook server

FROM graph-transfer-learning/egi:latest

ENTRYPOINT ["/bin/bash","-c"]
WORKDIR /workspace

RUN mkdir -p /workspace/mount

RUN pip install --upgrade pip

# Jupyter extensions
RUN pip install ipywidgets
RUN pip install jupyter_contrib_nbextensions
RUN pip install plotly scipy
RUN jupyter contrib nbextension install --user

CMD ["jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password=''"]

