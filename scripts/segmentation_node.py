#!/usr/bin/env python3
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
import threading
import os
import warnings

import numpy as np
import torch
import cv2 as cv

import rospy
import message_filters
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from computer_vision.msg import SegmentAndClassifyResult
from computer_vision.srv import SegmentAndClassifyService, SegmentAndClassifyServiceResponse

from models.all_model import AllModel
from utilities.utils import draw_masks, get_nearest_mask_id, get_ros_result

warnings.filterwarnings('ignore')


lock = threading.Lock()


class VisionNode:
    def __init__(self):
        # initialize a node
        rospy.init_node("vision_node", log_level=rospy.INFO)

        script_dir = os.path.dirname(os.path.realpath(__file__))
        # type 'topic' or 'service'
        self.type = rospy.get_param("'type'/node_type", "topic")
        self.dataset_save_folder = os.path.join(script_dir, rospy.get_param(
            'dataset_save_folder', f'segmentation_dataset'))
        self.segm_config = os.path.join(script_dir, rospy.get_param(
            'segm_config', f'checkpoints/config_Mask_RCNN.py'))
        self.segm_checkpoint = os.path.join(script_dir, rospy.get_param(
            'segm_checkpoint', f'checkpoints/ckpt_Mask_RCNN.pth'))
        self.segm_conf_thresh = rospy.get_param('segm_conf_thresh', 0.7)
        self.n_augmented_crops = rospy.get_param('n_augmented_crops', 10)
        self.fe_fp16 = rospy.get_param('fe_fp16', False)
        # self.knn_file = rospy.get_param(
        # 'knn_file', f'{script_dir}/checkpoints/sber_objects_20_augs.pth')
        self.knn_file = os.path.join(script_dir, rospy.get_param(
            'knn_file', f'm_platform_objects_10aug_no_crop.pth'))
        self.save_to_file = rospy.get_param('save_to_file', False)
        self.knn_size = rospy.get_param('knn_size', 15)

        rospy.logerr(self.dataset_save_folder)
        self.cv_bridge = CvBridge()

        self.im = None

        self.base_frame = 'measured/base_link'
        self.camera_frame = 'measured/camera_color_optical_frame'
        self.target_frame = self.base_frame

        self.features_to_save = None

        self.vis_pub = rospy.Publisher(
            '/rgb_segmented', Image, queue_size=10)

        if self.type == 'topic':

            rospy.logwarn('Publisher is setting up')
            self.results_pub = rospy.Publisher(
                '/segm_results', SegmentAndClassifyResult, queue_size=10)

            self.cropped_depth_pub = rospy.Publisher(
                '/depth_masked', Image, queue_size=10)

            self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw',
                                                      Image, queue_size=10)

            self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw',
                                                        Image, queue_size=10)

            ts = message_filters.ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub], 1, 0.1)
            ts.registerCallback(self.callback_rgbd)

        elif self.type == 'service':
            self.inference_srv = rospy.Service(
                'segmentation_inference_service', SegmentAndClassifyService, self.service_inference_callback)
            rospy.logwarn('Service is setting up')

            self.im = None
            self.depth = None
            self.rgb_msg_header = None

        self.train_srv = rospy.Service(
            'segmentation_train_service', Trigger, self.service_train_callback)

        self.end_train_srv = rospy.Service(
            'segmentation_end_train_service', Trigger, self.service_end_training_callback)

        self.model = AllModel(
            dataset_save_folder=self.dataset_save_folder,
            segm_config=self.segm_config,
            segm_checkpoint=self.segm_checkpoint,
            segm_conf_thresh=self.segm_conf_thresh,
            n_augmented_crops=self.n_augmented_crops,
            fe=torch.hub.load(
                'facebookresearch/dino:main', 'dino_vits16'),
            fe_fp16=self.fe_fp16,
            knn_file=self.knn_file,
            save_to_file=self.save_to_file,
            knn_size=self.knn_size
        )

        rospy.logwarn('Init complete!')

    def convert_msgs_to_images(self, rgb_msg, depth_msg):

        depth_im = self.cv_bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='32FC1')
        rgb_im = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        return rgb_im, depth_im

    def service_inference_callback(self, msg):
        rgb_msg = rospy.wait_for_message('/camera/color/image_raw', Image)
        depth_msg = rospy.wait_for_message(
            '/camera/aligned_depth_to_color/image_raw', Image)

        rgb_im, depth_im = self.convert_msgs_to_images(rgb_msg, depth_msg)

        return self.run_segmentation(rgb=rgb_im, depth=depth_im)

    def callback_rgbd(self, rgb, depth):

        rgb_im, depth_im = self.convert_msgs_to_images(rgb, depth)

        with lock:
            self.im = rgb_im.copy()
            self.depth = depth_im.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.rgb_msg_header = rgb.header

    def run_segmentation(self, rgb=None, depth=None):
        start = time.time()

        if rgb is None and depth is None:
            with lock:
                if self.im is None:
                    rospy.logerr_throttle(5, "No image received")
                    return
                rgb = self.im.copy()
                depth = self.depth.copy()

        (cl_names, cl_confs, cl_dists), masks = self.model(rgb, depth)

        nearest_mask = get_nearest_mask_id(depth, masks)

        depth_mask = cv.bitwise_and(
            depth, depth, mask=masks[nearest_mask])

        if len(masks) == 0:
            depth_mask = np.zeros_like(depth_mask)

        visualized_masks = draw_masks(
            rgb,
            depth,
            masks,
            cl_names,
            cl_confs,
            cl_dists,
            conf_thresh=0.6,
            dist_thresh=75,
            draw_only_nearest=False,
            # show_nearest=nearest_mask,
            show_nearest=-1,
            classes_list=self.model.classifier.classes,
            show_low_prob=True)

        if cl_names is not None:
            cl_names = [str(self.model.classes.index(cls) + 1)
                        for cls in cl_names]

        # prepare an answer
        results = get_ros_result(
            masks, depth_mask, cl_names, cl_confs, cl_dists, nearest_mask)

        vis_msg = self.cv_bridge.cv2_to_imgmsg(
            visualized_masks.astype(np.uint8), encoding='bgr8')
        vis_msg.header.stamp = rospy.get_rostime()
        self.vis_pub.publish(vis_msg)

        rospy.loginfo_throttle(10, f'FPS: {(1 /(time.time() - start)):.2f}')
        if self.model.classifier.classes == []:
            rospy.logwarn_throttle(5000, 'No trained classes found')
            if self.type == 'service':
                return SegmentAndClassifyServiceResponse(results)
            else:
                self.results_pub.publish(results)
                # rospy.logwarn(f'FPS: {(1 /(time.time() - start)):.2f}')
                rospy.logwarn(f'{(time.time() - start):.3f}')
                return

        # rospy.logwarn(f'FPS: {(1 /(time.time() - start)):.2f}')
        # send an answer
        if self.type == 'topic':
            self.results_pub.publish(results)
            depth_mask = cv.bitwise_and(
                depth, depth, mask=masks[nearest_mask])
            depth_mask_msg = self.cv_bridge.cv2_to_imgmsg(
                depth_mask.astype(np.uint16), encoding='16UC1')
            self.cropped_depth_pub.publish(depth_mask_msg)
        else:
            return SegmentAndClassifyServiceResponse(results)

    def service_train_callback(self, request):

        rgb_msg = rospy.wait_for_message('camera/color/image_raw', Image)
        depth_msg = rospy.wait_for_message(
            '/camera/aligned_depth_to_color/image_raw', Image)

        rgb, depth = self.convert_msgs_to_images(rgb_msg, depth_msg)

        resp = TriggerResponse()

        status = self.model(rgb, depth, train=True)

        if not status:
            resp.success = False
            resp.message = 'No masks found'
            return resp
        else:

            vis_msg = self.cv_bridge.cv2_to_imgmsg(
                rgb.astype(np.uint8), encoding='bgr8')
            vis_msg.header.stamp = rospy.get_rostime()
            self.vis_pub.publish(vis_msg)

            resp.success = True
            resp.message = status

            return resp

    def service_end_training_callback(self, request):
        self.model.add_to_dataset()
        resp = TriggerResponse()
        resp.success = True
        resp.message = ''
        return resp


if __name__ == '__main__':
    visionNone = VisionNode()
    rate = rospy.Rate(1000)

    if visionNone.type == 'topic':
        while not rospy.is_shutdown():

            visionNone.run_segmentation()
            rate.sleep()
    else:
        rospy.spin()
