#! /bin/bash

sudo docker run \
    --net host \
    --gpus all \
    --rm \
    -v cv_volume:/ws \
    -v /home/iiwa/Nenakhov/sandbox_ws/src/grasping_cell/ROS-object-recognition:/ws/src/ROS-object-recognition \
    -v /dev:/dev \
    -it \
    --privileged \
    ivan/iiwa_cv 
    # -v /home/ivan/cv/ros_ws:/ws \