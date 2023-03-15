
A development container for this project. Automatically opens a jupyter notebook server on port 8080 on localhost.


```
docker build . -t graph-transfer-learning/devel:v1 -t graph-transfer-learning/devel:latest
```

Run from the root directory of this repository:

```
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia -v "$(pwd):/workspace/mount" -p 8080:8888 --detach graph-transfer-learning/devel:latest
```
