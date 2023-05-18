# graph-transfer-learning

This repository contains on-going research into the transfer learning of
Graph Neural Networks. 

# Installation

## CUDA 

Currently, this code is ran through a NVIDIA accelerated docker container (as
mandated by the university systems). 

This has been tested on Ubuntu and a GTX 3060 card only.



**Run the following from this directory:**

First, build the container:

```
docker build . -t graph-transfer-learning/devel
```

Then, run the container:

```
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia -v "$(pwd):/workspace -p 8888:8888 --detach graph-transfer-learning/devel
```


* * * 

* The repository on your local system will be mounted to the `/workspace/`
  folder in the container.


## M1 Macs (metal)

A `nix` file has been provided to run the project on modern macs that support
metal acceleration.


With nix installed, run `nix-shell` in the current directory.


# Usage

TODO

<!-- vim: tw=80 cc=80
-->



