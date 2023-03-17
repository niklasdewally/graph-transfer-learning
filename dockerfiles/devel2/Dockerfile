# devel:v1
# Development container for this project
# Automatically runs a jupyter notebook server

FROM graph-transfer-learning/egi:latest

ENTRYPOINT ["/bin/bash","-c"]
WORKDIR /workspace

RUN mkdir -p /workspace/mount

RUN pip3 install ipython ipykernel

CMD ["jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password=''"]

