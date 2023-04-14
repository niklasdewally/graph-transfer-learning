
A development container for this project. Automatically opens a jupyter notebook server on port 8080 on localhost.

Run from the root directory of this repository:

```
docker build docker/ -t graph-transfer-learning/devel:v2 -t graph-transfer-learning/devel:latest
```


```
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia -v "$(pwd):/workspace/mount" -p 8080:8888 --detach graph-transfer-learning/devel:latest
```
