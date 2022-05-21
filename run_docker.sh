#! /bin/bash

sudo docker run \
    --net host \
    --gpus all \
    --rm \
    -v cv_volume:/ws \
    -v /home/iiwa/Nenakhov/ROS-object-recognition:/ws/src/ROS-object-recognition \
    -v /dev:/dev \
    -v /home/iiwa/Nenakhov/MMDeploy/work_dir:/workspace/mmdeploy/work_dir \
    -v /media/iiwa/AAA1/UOAIS-Sim:/workspace/mmdeploy/work_dir/UOAIS-Sim \
    -it \
    --privileged \
    ivan/ros_cv_deploy_torch_trt 
    # -v /home/ivan/cv/ros_ws:/ws \