
docker run -it --rm \
  --name prioreye \
  --security-opt label=disable \
  --device nvidia.com/gpu=all \
  --shm-size=16g \
  "$@" \
  prioreye:v1 /bin/bash
