
import numpy as np
# from mmdet.apis import inference_detector, init_detector

from mmdeploy.apis import inference_model
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config

import cv2 as cv

import torch
import time

class MMDeployWrapper:
    def __init__(self, model_cfg, deploy_cfg, backend_files, segm_conf_thresh=0.7, device='cuda', **kwargs):

        self.conf_thresh = segm_conf_thresh

        # initialize the detector
        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

        self.task_processor = build_task_processor(model_cfg, deploy_cfg, device)
        self.model = self.task_processor.init_backend_model(backend_files)
        self.input_shape = get_input_shape(deploy_cfg)



    def __call__(self, image):

        st = time.time()

        model_inputs, _ = self.task_processor.create_input(image, self.input_shape)
        with torch.no_grad():
            result = self.task_processor.run_inference(self.model, model_inputs)[0]
        dur = time.time() - st
        print(dur)
        
        cropped_objects = []
        masks = []
        for box, mask in zip(result[0][0], result[1][0]):

            mask = mask.cpu().numpy()
            box, conf = box[:-1], box[-1]
            # print(conf)

            if conf < self.conf_thresh:
                break
            if all(box == 0):
                box[0] = np.min(np.where(mask)[1])
                box[1] = np.min(np.where(mask)[0])
                box[2] = np.max(np.where(mask)[1])
                box[3] = np.max(np.where(mask)[0])
            box = box.astype(int)

            if (box[3] - box[1]) * (box[2] - box[0]) / (image.shape[0] * image.shape[1]) > 0.6:
                continue

            tmp_im = image.copy()

            tmp_im[~mask] = (0, 0, 0)

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
