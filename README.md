# graph-transfer-learning

This repository contains on-going research into the transfer learning of
Graph Neural Networks. 


# Installation

This project uses `poetry` for dependency management.

The following systems are supported:

| System type | Install methods | 
|-------------|-----------------|
| Linux with CUDA 11.7. | Docker, Poetry|
| Apple Silcon Macs (with GPU acceleration)| Poetry|



## Docker

An NVIDIA accelerated docker container running CUDA 11.7 has been provided.

This has been tested on Ubuntu and a GTX 3060 card only.


**Run the following from this directory:**

First, build the container:

```
docker build . -t graph-transfer-learning/devel
```

Then, run the container:

```
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia -v "$(pwd):/workspace graph-transfer-learning/devel
```

* * * 

* The repository on your local system will be mounted to the `/workspace/`
  folder in the container.

# Poetry

After installing `poetry`, run `poetry shell` to enter the project environment.




## M1 Macs (metal)

A `nix` file has been provided to run the project on modern macs that support
metal acceleration.


With nix installed, run `nix-shell` in the current directory.


# Usage

TODO

<!-- vim: tw=80 cc=80
-->



