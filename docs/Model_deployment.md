## Converting a model with MMDeploy
To speed up inference of a trained model in MMDetection framework you can perform optimization with MMDeploy tool. Keep in mind that this optimization is platfrom specific. It means that for each Hardware setup (mostly for GPU) the optimized models will be different.
 1. [Make sure your model can be converted in ONNX and TensorRT format](https://mmdeploy.readthedocs.io/en/latest/supported_models.html)
 2. [Install MMDeploy framework](https://mmdeploy.readthedocs.io/en/latest/build.html)
 3. [Convert model](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_convert_model.html)
 4. Put in scripts/checkpoints following files:
  4.1. TensorRT .engine file 
  4.2. MMDetection config file
  4.3. MMDeploy config file
  4.4. tensorRT config file
    
## Inference 

For TensorRT framework the only file needed is .engine file. In a file ```models/all_model.py``` uncomment import of TensorRTWrapper and change ```self.segm_model``` wrapper to imported one.
For MMdeploy framework all 4 files are needed.

Comparing TensorRT and MMDeploy, the first one is slightly faster (on GTX 1050 ti). 

MMdeploy autimatically performs preprocessing of the result, so the output masks are much finer that in TensorRT, where you need to perform postprocessing step (I am sure it can be fixed with a proper processing). 
