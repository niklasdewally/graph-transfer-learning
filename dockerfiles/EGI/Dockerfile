FROM python:3.8

RUN git clone "https://github.com/GentleZhu/EGI" repository

RUN cd repository && \
    python -m pip install dgl==0.4.3   \
                          torch        \
                          IPython      \
                          networkx     \
                          tqdm         \
                          scikit-learn \
                          numpy

ENTRYPOINT ["/bin/bash"]

    



