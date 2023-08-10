# graph-transfer-learning

This repository contains ongoing research into the transfer learning of
Graph Neural Networks.


# Installation

This project uses `poetry` for dependency management.

The following system configurations are supported:

| System type | Install methods | 
|-------------|-----------------|
| Linux + CUDA 11.7 compatible card | Docker, Poetry|
| Apple Silcon Macs (GPU accelerated) | Poetry|
| Intel Macs (CPU) | Poetry|


Further systems could be added (if desired) by adding the systems' `dgl` wheels to the `pyproject.toml` file.

## Docker

Docker container running CUDA 11.7 has been provided. A working installation of NVIDIA container toolkit on the host machine is requried for this to work.

This has been tested on Ubuntu and a GTX 3060 card only.


**Run the following from this directory:**

First, build the container:

```
docker build --build-arg local_uid=$(id -u) --build-arg local_user=$USER -t graph-transfer-learning/devel .
```

Then, run the container:

```
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia --rm -it --user $(id -u):$(id -g) -v "$(pwd):/home/$(whoami)/workspace" graph-transfer-learning/devel bash
```

The local filesystem will be mounted to the `workspace/` folder in the container.


Note that the user inside the container has  the same user name and id as the user who launched the container on the host system. This is because of https://github.com/niklasdewally/graph-transfer-learning/issues/2. 

To access root commands inside the container, use `sudo`.

## Poetry

After installing `poetry`, 

* Run `poetry install --with cpu --without cuda`, or `poetry install --with cuda --without cpu`.
* Enter the project environment with `poetry shell`



# Usage

* Library code is contained within the `src` directory.

* Executable code, including experiments, is contained in the `scripts`
  directory. 

<!-- vim: tw=80 cc=80
-->



