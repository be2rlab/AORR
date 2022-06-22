## segmentation_node.py
The main node is ```segmentation_node.py```. It performs segmentation, feature extraction and classification separately.

Organized in OOP paradigm it has the following components (Python-like pseudocode):
```Python
class VisionNode:
  def __init__(parameters):
    create ROS node, publisher, subscribers, service for inference
    Create services for training for new object
    initialize recognition_wrapper
  
  def service_inference_callback(request):
    read messages from RGB and Depth topics of camera
    run_segmentation()
    return ServiceResponse
    
  def callback_rgbd(rgb, depth):
    convert incoming rgb and depth messages to Numpy arrays
    run_segmentation()
    
  def run_segmentation():
    infer recognition model
    visualize predicted masks and classes
    convert results to ROS message
    send results
    
  def service_train_callback():
    segment current image and save the nearest object
    return trigger success response
    
  def service_end_train_callback():
    add saved images to classifier
    return trigger success response
```

## all_model.py
[Object recognition wrapper](https://github.com/be2rlab/ROS-object-recognition/blob/master/scripts/models/all_model.py) contains taks specific wrappers (for class-agnostic segmentation, feature extraction (not completely) and classification). Specifying segmentation model wrapper, we can change framework for segmentation. 

## Segmentation wrappers
Currently, four wrappers are supported:
 1. MMDetection
 2. Detectron2
 3. MMDeploy
 4. TensorRT

MMDetection and Detectron2 are the most common frameworks for object detection and instance segmentation tasks. MMDeploy and TensorRT are frameworks for faster inference of trained models. Instructions about them can be found [here](https://github.com/be2rlab/ROS-object-recognition/blob/master/docs/Model_deployment.md).

All wrappers have the same call intefrace and slightly different constructors.
