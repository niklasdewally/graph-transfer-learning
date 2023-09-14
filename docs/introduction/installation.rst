============
Installation
============


Local Machine
--------------

`Poetry <https://python-poetry.org/>`_ is used to manage most packages.
However, it does not manage ``pytorch`` and ``dgl`` well, so these must be
installed seperately afterwards.

----


To create a working environment on your local machine:


1. Install `Poetry <https://python-poetry.org/>`_ and Python 3.11.

2. Install dependencies using ``poetry install``.

3. Enter the virtual environment using ``poetry shell``.

4. Install `DGL <https://www.dgl.ai/pages/start.html>`_ and `Pytorch
   <https://pytorch.org/get-started/locally/>`_ using ``pip`` as described for
   your system.

Docker
-------

A docker container has been provided for running the project on a GPU server.
NVIDIA Container Runtime is required for this container.

*This has been tested on gpu-serv-01 only*

.. highlight:: shell

::

  docker build -t graph-transfer-learning/devel .
  # e.g. use gpus 2 and 3. change as appropriate.
  docker run --gpus '"device=2,3"' --rm -it -v "$(pwd):/workspace" graph-transfer-learning/devel bash


Lab Machines 
------------

St Andrews CS systems ship an older version of ``python`` by default, but
provide alternate versions at ``/usr/local/python/bin``.

This project requires ``python3.11``, found at ``/usr/local/python/bin/python3.11``.

See :ref:`multiple-python-ver`.

.. _multiple-python-ver:

Systems with multiple versions of Python
----------------------------------------

To switch the version of python used by poetry, run ``poetry env <PATH>`` where ``<PATH>`` is the path to a working ``python3.11`` executable.
Then, follow the installation instructions above.

This is useful if you have multiple versions of Python on your system, which is often the case on MacOS installations with ``brew``.

