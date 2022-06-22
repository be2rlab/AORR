# ROS object recognition
This repo combines: 
1. class-agnostic segmentation with wrappers for Detectron (not tested), MMDetection, MMDeploy and TensorRT
2. classification based on transformer feature extractor and kNN classifier

# System requirements

This project was tested with:
- Ubuntu 20.04
- ROS noetic
- torch 1.10
- CUDA 11.3
- NVIDIA GTX 1050ti / RTX 3090

## Preparations:
1. clone this repo
2. (optionally) download model checkpoint and config from **[GDrive](https://drive.google.com/file/d/1GHeLyvsXV3rrEWwBA5H-omxduFUOOlH7/view?usp=sharing)** and extract it in scripts/checkpoints folder

## Environment setup with Anaconda (for Detectron2 and MMDetection usage)
1. Create anaconda environment: ```conda env create -n conda_environment.yml```
2. ```conda activate segmentation_ros```
3. Install MMdet ```pip install openmim; mim install mmdet```
4. (Optionally) install Detectron2

## Environment setup with Docker (for any framework)

1. build docker image ```sudo sh build_docker.sh```
2. In line 8 in ```run_docker.sh``` change first path to your workspace folder
3. run docker container ```sudo sh run_docker.sh```
4. ```catkin_make; source devel/setup.bash```

# Usage
## Main node
Run node:
```roslaunch computer_vision cv.launch```

By default, it runs publisher. Optionally you can pass an argument mode:=service to run in service mode.
Along with inference mode, this node has training mode to save new objects in classifier.

### Publisher mode 
#### Input data:
 - rgb image (/camera/color/image_raw)
 - aligned depth image (/camera/aligned_depth_to_color/image_raw)
#### Output data
As a result the node publishes a message [SegmentAndClassifyResult](https://github.com/be2rlab/ROS-object-recognition/blob/master/msg/SegmentAndClassifyResult.msg) to a topic ```/segm_results```.

A more deeper description can be found [here](https://github.com/be2rlab/ROS-object-recognition/blob/master/docs/Main_node.md).
## Learning of new objects
An algorithm for adding a new object:
1. place a new object in a field of view of camera so that it is the nearest detected object in a screen.
2. Call ```/segmentation_train_service``` to mask this object, get featues from feature extractor and save them
3. Repeat previous step with different angle of view
4. Call ```/segmentation_end_train_service``` to add all saved features to kNN.

### Realsense with point cloud publishing

```roslaunch computer_vision pc.launch``` - runs [Realsense node](https://github.com/IntelRealSense/realsense-ros) with point_cloud=true option.
