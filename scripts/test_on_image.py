# Copyright (C) 2022 ITMO University
# 
# This file is part of Adaptive Object Recognition For Robotics.
# 
# Adaptive Object Recognition For Robotics is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Adaptive Object Recognition For Robotics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Adaptive Object Recognition For Robotics.  If not, see <http://www.gnu.org/licenses/>.

from models.mmdet_wrapper import MMDetWrapper
from mmdet.apis import init_detector, inference_detector
import cv2 as cv
import numpy as np
import albumentations as A

tr = A.Compose([
    A.LongestMaxSize(max_size=100),
    A.PadIfNeeded(min_height=100, min_width=100, border_mode=cv.BORDER_CONSTANT, value=(0, 0, 0)),
])

if __name__ == '__main__':

    config_file = 'checkpoints/config_Mask_RCNN.py'
    checkpoint_file = 'checkpoints/ckpt_Mask_RCNN.pth'
    # model = MMDetWrapper(segm_config=config_file,
    #                         segm_checkpoint=checkpoint_file)

    
    im_file = '/home/iiwa/Nenakhov/Mask_RCNN_OCID_train/OCID-dataset/ARID20/floor/top/seq04/rgb/result_2018-08-20-10-26-39.png'

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    result = inference_detector(model, im_file)

    # or save the visualization results to image files
    model.show_result(im_file, result, thickness=1,
                    font_size=13,
                    
                    out_file='result.jpg')

    cropped_objects = []
    masks = []
    image = cv.imread(im_file)
    for idx, (box, mask) in enumerate(zip(result[0][0], result[1][0])):

        box, conf = box[:-1], box[-1]

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
        tmp_im = tr(image=tmp_im)['image']
        cropped_objects.append(tmp_im)
        cv.imwrite(f'crops/{idx}.png', tmp_im)

    cv.imwrite('crops/all_crop.png', np.vstack(cropped_objects))