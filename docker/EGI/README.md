```
docker build -t graph-transfer-learning/egi:latest -t graph-transfer-learning/egi:v2 .
```

```
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia -it graph-transfer-learning/egi:latest
```

