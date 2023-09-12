============
Installation
============


Local Machine
--------------

`Poetry <https://python-poetry.org/>`_ is used to manage most packages.
However, it does not manage ``pytorch`` and ``dgl`` well, so these must be
installed seperately.


To create a working environment on your local machine:


1. Install `Poetry <https://python-poetry.org/>`_ and Python 3.11.

2. Install dependencies using ``poetry install``.

3. Enter the virtual environment using ``poetry shell``.

4. Install `DGL <https://www.dgl.ai/pages/start.html>`_ and `Pytorch
   <https://pytorch.org/get-started/locally/>`_ using ``pip`` as described for
   your system.

Docker
-------

The docker container has been tested on gpu-serv only, and expects a CUDA 11.7
NVIDIA runtime to be installed.

.. highlight:: shell

::

  docker build -t graph-transfer-learning/devel
  # for example, use gpus 2,3
  docker run --gpus '"device=2,3"' --rm -it -v "$(pwd):/workspace" graph-transfer-learning/devel bash


Lab Machines 
------------

The St Andrews systems ship an older version of ``python`` by default.

Alternate versions of ``python`` are provided at ``/usr/local/python/bin``.

We want ``/usr/local/python/bin/python3.11``.

See :ref:`multiple-python-ver`.

.. _multiple-python-ver:

Systems with multiple versions of Python
----------------------------------------

To switch the version of python used by poetry, run ``poetry env <PATH>`` where ``<PATH>`` is the path to a working ``python3.11`` executable.

