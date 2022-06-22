import tensorrt as trt
import numpy as np
import os
import time
import pycuda.driver as cuda
import pycuda.autoinit

import albumentations as A
# from mmdeploy.apis import inference_model
from mmdeploy.apis.utils import build_task_processor
# from mmdeploy.utils import get_input_shape, load_config
# from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
import cv2 as cv

from models.segm_utils import resize_mask, _do_paste_mask

TRT_LOGGER = trt.Logger()

transforms = A.Compose([
    A.Normalize(),
])

# For torchvision models, input images are loaded in to a range of [0, 1] and
# normalized using mean = [0.485, 0.456, 0.406] and stddev = [0.229, 0.224, 0.225].
def preprocess(image):
    # Mean normalization
    # mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    # stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    # data = (np.asarray(image).astype('float32') / float(255.0) - mean) / stddev
    data = transforms(image=image)['image']
    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


class TRTWrapper:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32, segm_conf_thresh=0.9, **kwargs):

        self.conf_thresh = segm_conf_thresh

        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = load_engine(engine_path)
        self.max_batch_size = max_batch_size
        # width, height = 640, 480
        # self.inputs, self.outputs, self.bindings, self.stream, self.out_buffer_size = self.allocate_buffers(np.ones((1, 3, width, height)), dtype=np.float32)
        self.context = self.engine.create_execution_context()
        self.context.set_binding_shape(self.engine.get_binding_index("input"), (1, 3, 480, 640))

        self.allocate_buffers()


    def allocate_buffers(self):
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            shape = self.context.get_binding_shape(i)
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                'index': i,
                'name': name,
                'dtype': dtype,
                'shape': list(shape),
                'allocation': allocation,
                'host_allocation': host_allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            print("{} '{}' with shape {} and dtype {}".format(
                "Input" if is_input else "Output",
                binding['name'], binding['shape'], binding['dtype']))

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0
       
            
    def __call__(self,image):

        input_image = preprocess(image)

        input_image = input_image[..., np.newaxis]
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(input_image))
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])
        out = [o['host_allocation'] for o in self.outputs]

        keep_inds = out[0][0][:, -1] > self.conf_thresh
        boxes = out[0][0][keep_inds]
        masks = out[2][0][keep_inds]
        result = list(zip(boxes, masks))

        result = list(zip(boxes, masks))

        cropped_objects = []
        masks = []

        for box, mask in result:

            box, conf = box[:-1].astype(int), box[-1]

            if all(box) == 0.:
                continue
            if conf < self.conf_thresh:
                break

            # mask = (mask > 0.5).numpy().astype(np.uint8)
            mask = (mask > 0.5).astype(np.uint8)
            mask = resize_mask(mask, box, image.shape[:-1])

            if (box[3] - box[1]) * (box[2] - box[0]) / (image.shape[0] * image.shape[1]) > 0.6:
                continue

            tmp_im = image.copy()

            tmp_im = cv.bitwise_and(tmp_im, tmp_im, mask=mask)


            tmp_im = tmp_im[box[1]: box[3], box[0]:box[2]]

            mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_ERODE, np.ones(
                (3, 3), np.uint8)).astype(np.uint8)

            cntrs, _ = cv.findContours(
                mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if len(cntrs) > 1:
                areas = np.array([cv.contourArea(c) for c in cntrs])
                biggest_cntr = cntrs[np.argmax(areas)]
                new_mask = np.zeros_like(mask)
                cv.drawContours(new_mask, [biggest_cntr], -1, 255, cv.FILLED)
                mask = new_mask
            masks.append(mask)

            cropped_objects.append(tmp_im)

        return cropped_objects, np.array(masks)
      
        
if __name__ == "__main__":
    batch_size = 1
    trt_engine_path = os.path.join("checkpoints","trt_ckpts", "end2end.engine")
    print(trt_engine_path)
    model = TrtModel(trt_engine_path)
    shape = model.engine.get_binding_shape(0)

    
    data = np.random.randint(0,255,(batch_size,*shape[1:]))/255
    result = model(data,batch_size)
