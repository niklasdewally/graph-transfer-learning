# graph-transfer-learning

This repository contains on-going research into the transfer learning of
Graph Neural Networks. 

# Installation

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
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia -v "$(pwd):/workspace/mount" -p 8888:8888 --detach graph-transfer-learning/devel
```


* * * 

* The repository on your local system will be mounted to the `/workspace/mount`
  folder in the container.

* This container also runs a `Jupyter` server, which is 
  accessible at `localhost:8888`.




# Usage

TODO

<!-- vim: tw=80 cc=80
-->



