#! /bin/bash

sudo docker run \
    --net host \
    --gpus all \
    --rm \
    -v cv_volume:/ws \
    -v /home/iiwa/Nenakhov/ROS-object-recognition:/ws/src/ROS-object-recognition \
    -v /dev:/dev \
    -it \
    --privileged \
    ivan/ros_cv