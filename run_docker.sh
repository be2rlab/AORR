#! /bin/bash

sudo docker run \
    --net host \
    --gpus all \
    --rm \
    -v /home/ivan/cv/ros_ws:/ws \
    -v /dev:/dev \
    -it \
    --privileged \
    ivan/iiwa_cv 