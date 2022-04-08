
import numpy as np
from mmdet.apis import inference_detector, init_detector
import cv2 as cv
import os
import rospy
from models.utils import download_file_from_google_drive
import gdown
import shutil


class MMDetWrapper:
    def __init__(self, segm_conf_thresh=0.7, segm_config='models/mmdet_config.py', segm_checkpoint='models/latest.pth', device='cuda', **kwargs):

        self.conf_thresh = segm_conf_thresh
        if not os.path.exists(segm_checkpoint):
            print("Downloading data archive...", end=" ")

            save_dir = '/'.join(segm_checkpoint.split('/')[:-1])

            id = '1GHeLyvsXV3rrEWwBA5H-omxduFUOOlH7'
            gdown.download(f'https://drive.google.com/uc?id={id}', f"{save_dir}/mmdet_model.zip", quiet=False)

            shutil.unpack_archive(f"{save_dir}/mmdet_model.zip", save_dir)
            
            print("Done!")
        # initialize the detector
        self.model = init_detector(
            segm_config, segm_checkpoint, device=device)

    def __call__(self, image):

        result = inference_detector(self.model, image)

        cropped_objects = []
        masks = []
        for box, mask in zip(result[0][0], result[1][0]):

            box, conf = box[:-1], box[-1]

            if conf < self.conf_thresh:
                continue
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
