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


import numpy as np
import cv2 as cv
from scipy.spatial.distance import euclidean
from cv_bridge import CvBridge
from std_msgs.msg import Header
from computer_vision.msg import SegmentAndClassifyResult
import rospy

bridge = CvBridge()
np.random.seed(0)
color_list = np.random.randint(0, 255, size=(100, 3))


def get_ros_result(masks, depth_cropped, cl_names, cl_confs, cl_dists, nearest_mask):
    results = SegmentAndClassifyResult()
    results.header = Header()
    results.header.stamp = rospy.get_rostime()
    results.mask = bridge.cv2_to_imgmsg(
        masks[nearest_mask].astype(np.uint8), encoding='8UC1')

    results.masked_depth = bridge.cv2_to_imgmsg(
        depth_cropped.astype(np.uint16), encoding='16UC1')

    if cl_names is not None:
        results.class_name = cl_names[nearest_mask]
        results.class_dist = cl_dists[nearest_mask]
        results.class_conf = cl_confs[nearest_mask]
        depth_cropped = np.zeros_like(masks[0])
    else:
        results.class_name = 'None'
        results.class_dist = 0.0
        results.class_conf = 0.0

    return results


def find_nearest_mask(depth, masks):
    dists = []
    if len(masks) == 0:
        return

    for mask in masks:

        cntrs, hierarchy = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # print(cntrs[0])
        M = cv.moments(cntrs[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        dists.append(depth[cY, cX])
    ret_im = depth.copy()
    ret_im[~masks[np.argmin(dists)]] = 0
    return ret_im


def draw_masks(inp_im, depth, masks, clss, confs, dists, show_low_prob=True, show_nearest=-1, conf_thresh=0.7, dist_thresh=70, draw_only_nearest=True, classes_list=None):
    # draw masks and nearest object
    image = inp_im.copy()

    if len(masks) == 0:
        return image

    if clss is None:
        clss = [None] * len(masks)
        confs = [np.nan] * len(masks)
        dists = [np.nan] * len(masks)

    depth_dists = []


    vis_cntrs_data = []
    for idx, (mask, cls, conf, dist) in enumerate(zip(masks.astype(np.uint8), clss, confs, dists)):

        contours, hierarchy = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        lengths = []
        for cnt in contours:
            lengths.append(len(cnt))
        cntr = contours[np.argmax(lengths)]

        M = cv.moments(cntr)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if depth[cY, cX] != 0:
            depth_dists.append(depth[cY, cX])
        else:
            depth_dists.append(1e4)

        xmin = np.min(contours[0][:, 0, 0])
        ymin = np.min(contours[0][:, 0, 1])

        if len(classes_list) == 0:
            color = (0, 0, 255)
        elif conf <= conf_thresh or dist >= dist_thresh or cls is None:
            color = (0, 0, 0)

        else:
            color = color_list[classes_list.index(cls)]

        # dist = 'Novel' if conf < conf_thresh else ''
        # text = f'{cls} {conf:.2f} {dist}'
        text = f'{cls} {conf:.2f} {dist:.2f}'
        # text = f'{cls} {conf:.2f}'# {dist:.2f}'

        vis_cntrs_data.append((contours,
                               text if cls is not None else None,
                               (xmin, ymin),
                               color)
                              )

    for idx, vis_cntr_data in enumerate(vis_cntrs_data):
        if idx != np.argmin(depth_dists) and draw_only_nearest:
            continue
        cntr, text, (xmin, ymin), color = vis_cntr_data
        color = list(map(int, color))
        if color == [0, 0, 0] and not show_low_prob:
            continue

        cv.drawContours(image, cntr, -1, color, 4)
        if text is not None:
            cv.putText(image, text,
                       (xmin, ymin - 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # # Find nearest to the camera contour and draw bold blue line
    if show_nearest != -1 and not draw_only_nearest:
        cntr = cv.findContours(
            masks[show_nearest], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        cv.drawContours(image, cntr, -1, (255, 0, 0), 4)
    return image


def get_padded_image(img):
    # Getting the bigger side of the image
    s = max(img.shape[0:2])

    # Creating a dark square with NUMPY
    f = np.zeros((s, s, 3), np.uint8)

    # Getting the centering position
    ax, ay = (s - img.shape[1])//2, (s - img.shape[0])//2

    # Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img

    return f


def get_nearest_to_center_box(im_shape, boxes):
    center = np.array(im_shape[:-1]) // 2
    min_dist = 1000000  # just a big number
    min_idx = -1
    for idx, box in enumerate(boxes):
        box_center = ((box[3] + box[1]) // 2, (box[2] + box[0]) // 2)
        dist = euclidean(box_center, center)
        if dist < min_dist:
            min_dist = dist
            min_idx = idx

    return min_idx


def eucl_dist(x, y): return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)


def get_nearest_mask_id(depth, masks):

    if len(masks) == 0:
        return None

    depth_dists = []
    dists = []

    center = np.array(depth.shape) // 2

    for mask in masks.astype(np.uint8):

        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue
        lengths = []
        for cnt in contours:
            lengths.append(len(cnt))
        cntr = contours[np.argmax(lengths)]

        M = cv.moments(cntr)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # center_depth = np.mean(depth[mask])
        center_depth = depth[cY, cX]
        dists.append(eucl_dist([cY, cX], center))
        if depth[cY, cX] != 0:
            depth_dists.append(center_depth)
        else:
            depth_dists.append(1e4)


    return np.argmin(dists)


