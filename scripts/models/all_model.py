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


import time
import os

# import torch_tensorrt
import torch
import numpy as np
import cv2 as cv
import albumentations as A
from albumentations.pytorch import ToTensorV2
# local packages

from models.classifier import classifier
from utilities.utils import get_nearest_mask_id

class AllModel:
    # def __init__(self, fe=None, knn_file=None, segm_conf_thresh=None, fe_fp16=False) -> None:
    @torch.no_grad()
    def __init__(self, 
                fe=None, 
                fe_fp16=False, 
                n_augmented_crops=10, 
                dataset_save_folder=f'{os.path.dirname(os.path.realpath(__file__))}../dataset_segmentation', 
                framework='Detectron2', 
                **kwargs) -> None:
        self.fe_fp16 = fe_fp16
        self.n_augmented_crops = n_augmented_crops

        script_dir = os.path.dirname(os.path.realpath(__file__))

        self.dataset_save_folder = dataset_save_folder

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # prepare models for segmentation, feature extraction and classification

        if framework == 'MMDetection':
            # Import in the middle of the code is used to prevent conflicts when importing all frameworks at once
            from models.mmdet_wrapper import MMDetWrapper

            self.segm_model = MMDetWrapper(device=self.device, **kwargs)
        elif framework == 'MMDeploy':
            from models.mmdeploy_wrapper import MMDeployWrapper

            self.segm_model = MMDeployWrapper(f'{script_dir}/../checkpoints/config_Mask_RCNN.py', 
                                    # '/workspace/mmdeploy/configs/mmdet/instance-seg/instance-seg_tensorrt-fp16_dynamic-320x320-1344x1344.py', 
                                    f'{script_dir}/../checkpoints/trt_ckpts/config_int8_640x480.py',
                                    [f'{script_dir}/../checkpoints/trt_ckpts/int8.engine', 
                                    ], **kwargs)
        elif framework == 'Detectron2':
            from models.detectron2_wrapper import Detectron2Wrapper

            self.segm_model = Detectron2Wrapper()
        elif framework == 'TensorRT':
            from models.TRT_Wrapper import TRTWrapper

            self.segm_model = TRTWrapper(f'{script_dir}/../checkpoints/trt_ckpts/fp32.engine', **kwargs)
        else:
            raise NotImplementedError(f"Wrong framework flag: {framework}. Supported are: MMDetection, Detectron2, MMDeploy, TensorRT")

        if not fe:
            self.fe = torch.hub.load(
                'facebookresearch/dino:main', 'dino_vits16')
        else:
            self.fe = fe
        self.fe.eval()

        self.fe = torch.jit.trace(self.fe.to(self.device), torch.randn((1, 3, 224, 224), dtype=torch.float32, device=self.device))



        if fe_fp16:
            self.fe = self.fe.half()

        self.fe.to(self.device)


        self.fe_transforms = A.Compose([
            A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(min_height=224, min_width=224,
                          border_mode=cv.BORDER_CONSTANT, p=1),
            A.Normalize(),
            ToTensorV2()
        ])
        self.fe_augmentations = A.Compose([
            A.LongestMaxSize(max_size=300),
            A.PadIfNeeded(min_height=300, min_width=300,
                          border_mode=cv.BORDER_CONSTANT, p=1),
            A.RandomBrightnessContrast(
                brightness_limit=0.0, contrast_limit=0.4),
            A.Flip(),
            A.ShiftScaleRotate(shift_limit=0.125, scale_limit=0.2,
                               rotate_limit=90, border_mode=cv.BORDER_CONSTANT, p=1),
            A.CenterCrop(224, 224, p=0.3)
        ])

        self.features_to_save = []

        # self.classifier = knn(**kwargs)
        self.classifier = classifier(**kwargs)

        # check if saved knn file has same dimensionality as feature extractor
        if self.classifier.x_data is not None:
            sample_input = torch.ones(
                (1, 3, 224, 224), dtype=torch.float32, device=self.device)
            if fe_fp16:
                sample_input = sample_input.half()

                
        self.classes = self.classifier.classes.copy()

    @torch.no_grad()
    def __call__(self, color_im, depth_im, train=False):

        if not train:
            start = time.time()

            cropped_objs, masks = self.segm_model(color_im)

            segm_dur = time.time() - start
            if not cropped_objs:
                return (None, None, None), masks

            transformed_objs = torch.stack(
                [self.fe_transforms(image=i)['image'] for i in cropped_objs]).to(self.device)

            if self.fe_fp16:
                transformed_objs = transformed_objs.half()
            start = time.time()

            # torch.save(transformed_objs, 'f_tensor.pth')
            features = self.fe(transformed_objs).cpu()

            fe_dur = time.time() - start
            start = time.time()

            cls, confs, dists = self.classifier.classify(features)

            clf_dur = time.time() - start

            total = segm_dur + fe_dur + clf_dur
            print(f'Duration - segm: {segm_dur:.3f}, fe: {fe_dur:.3f}, clf: {clf_dur:.3f}, total: {total:.3f}')

            return (cls, confs, dists), masks
        else:
            cropped_objs, masks = self.segm_model(color_im)
            if not cropped_objs:
                return False

            nearest_mask_id = get_nearest_mask_id(depth_im, masks)

            if self.n_augmented_crops:
                augmented_crops = [self.fe_augmentations(image=cropped_objs[nearest_mask_id])[
                    'image'] for _ in range(self.n_augmented_crops)]
                # transformed_objs = torch.stack(transformed_objs).cuda()
            else:
                augmented_crops = [cropped_objs[nearest_mask_id]]
                # transformed_objs = torch.tensor(self.fe_transforms(
                #     image=cropped_objs[nearest_mask_id])['image']).unsqueeze(0).cuda()

            cl = f'{len(self.classifier.classes) + 1}'
            t_stamp = time.time()
            for idx, crop in enumerate(augmented_crops):
                fname = f'{self.dataset_save_folder}/{cl}/{(t_stamp):.2f}_{idx}.png'
                os.makedirs(f'{self.dataset_save_folder}/{cl}', exist_ok=True)
                cv.imwrite(fname, crop)

            transformed_objs = torch.stack([self.fe_transforms(
                image=cr)['image'] for cr in augmented_crops]).to(self.device)

            if self.fe_fp16:
                transformed_objs = transformed_objs.half()

            features = self.fe(transformed_objs).cpu().float()

            return self.save_feature(features)

    def save_feature(self, features):
        self.features_to_save += features
        return 'features saved'

    def add_to_dataset(self):
        feats = np.stack(self.features_to_save)

        self.classes.append(f'{len(self.classifier.classes) + 1}')

        self.classifier.add_points(
            feats, [f'{len(self.classifier.classes) + 1}'] * len(feats))
        self.classifier.print_info()
        self.features_to_save = []


# if __name__ == '__main__':
    # model = torch.hub.load(
    #             'facebookresearch/dino:main', 'dino_vits16')
    # model.cuda()

    # a = torch.ones((8, 3, 224, 224), dtype=torch.float32, device='cuda')

    # b = model(a)