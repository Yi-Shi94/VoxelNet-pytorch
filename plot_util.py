# -*- cooing:UTF-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import with_statement
from __future__ import absolute_import

import cv2
import numpy as np
import math
import yaml
from box_overlaps import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

yamlPath = "configure.yaml"
f = open(yamlPath, 'r', encoding='utf-8')
conf = f.read()
conf_dict = yaml.safe_load(conf) 

range_x=conf_dict['range_x']
range_y=conf_dict['range_y']
range_z=conf_dict['range_z']
vox_depth = conf_dict['vox_d']
vox_width = conf_dict['vox_w']
vox_height = conf_dict['vox_h']
pt_thres_per_vox = conf_dict['pt_thres_per_vox'] 
anchors_per_vox = conf_dict['anchors_per_vox']

W = math.ceil((max(range_x)-min(range_x))/vox_width)
H = math.ceil((max(range_y)-min(range_y))/vox_height)
D = math.ceil((max(range_z)-min(range_z))/vox_depth)

ANCHOR_L = conf_dict['ANCHOR_L']
ANCHOR_W = conf_dict['ANCHOR_W']
ANCHOR_H = conf_dict['ANCHOR_H']
ANCHOR_Z =  -1.0 - ANCHOR_H/2
#-- util function to load calib matrices


def colorize(value, factor=1, vmin=None, vmax=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - factor: resize factor, scalar
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    value = np.sum(value, axis=-1)
    vmin = np.min(value) if vmin is None else vmin
    vmax = np.max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    value = (value * 255).astype(np.uint8)
    value = cv2.applyColorMap(value, cv2.COLORMAP_JET)
    value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    x, y, _ = value.shape
    value = cv2.resize(value, (y * factor, x * factor))

    return value


def load_calib(calib_dir):
    # P2 * R0_rect * Tr_velo_to_cam * y
    lines = open(calib_dir).readlines()
    lines = [ line.split()[1:] for line in lines ][:-1]
    #
    P = np.array(lines[CAM]).reshape(3,4)
    P = np.concatenate( (P, np.array( [[0,0,0,0]] )  ), 0  )
    #
    Tr_velo_to_cam = np.array(lines[5]).reshape(3,4)
    Tr_velo_to_cam = np.concatenate( [ Tr_velo_to_cam, np.array([0,0,0,1]).reshape(1,4)], 0 )
    #
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3,:3] = np.array(lines[4][:9]).reshape(3,3)
    #
    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')
    return P, Tr_velo_to_cam, R_cam_to_rect


def lidar_to_bird_view(x, y, factor=1):
    a = (x - min(range_x)) / vox_width * factor
    b = (y - min(range_y)) / vox_height * factor
    a = np.clip(a, a_max=(max(range_x)-min(range_x)) / vox_width * factor, a_min=0)
    b = np.clip(b, a_max=(max(range_y)-min(range_y)) / vox_height * factor, a_min=0)
    return a, b

def batch_lidar_to_bird_view(points, factor=1):
    # Input:
    #   points (N, 2)
    # Outputs:
    #   points (N, 2)
    
    a = (points[:, 0] - min(range_x)) / vox_width * factor
    b = (points[:, 1] - min(range_y)) / vox_height * factor
    a = np.clip(a, a_max=(max(range_x)-min(range_x)) / vox_width * factor, a_min=0)
    b = np.clip(b, a_max=(max(range_y)-min(range_y)) / vox_height * factor, a_min=0)
    return np.concatenate([a[:, np.newaxis], b[:, np.newaxis]], axis=-1)


def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2
    return angle


def camera_to_lidar(x, y, z, T_VELO_2_CAM=None, R_RECT_0=None):
    p = np.array([x, y, z, 1])
    p = np.matmul(np.linalg.inv(R_RECT_0), p)
    p = np.matmul(np.linalg.inv(T_VELO_2_CAM), p)
    p = p[0:3]
    return tuple(p)


def lidar_to_camera(x, y, z, T_VELO_2_CAM=None, R_RECT_0=None):
    p = np.array([x, y, z, 1])
    p = np.matmul(T_VELO_2_CAM, p)
    p = np.matmul(R_RECT_0, p)
    p = p[0:3]
    return tuple(p)


def camera_to_lidar_point(points, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 3) -> (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))]).T  # (N,4) -> (4,N)
    points = np.matmul(np.linalg.inv(R_RECT_0), points)
    points = np.matmul(np.linalg.inv(T_VELO_2_CAM), points).T  # (4, N) -> (N, 4)
    points = points[:, 0:3]
    return points.reshape(-1, 3)


def lidar_to_camera_point(points, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 3) -> (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))]).T
    points = np.matmul(T_VELO_2_CAM, points)
    points = np.matmul(R_RECT_0, points).T
    points = points[:, 0:3]
    return points.reshape(-1, 3)


def camera_to_lidar_box(boxes, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camera_to_lidar(
            x, y, z, T_VELO_2_CAM, R_RECT_0), h, w, l, -ry - np.pi / 2
        rz = angle_in_limit(rz)
        ret.append([x, y, z, h, w, l, rz])
    return np.array(ret).reshape(-1, 7)


def lidar_to_camera_box(boxes, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, rz = box
        (x, y, z), h, w, l, ry = lidar_to_camera(
            x, y, z, T_VELO_2_CAM, R_RECT_0), h, w, l, -rz - np.pi / 2
        ry = angle_in_limit(ry)
        ret.append([x, y, z, h, w, l, ry])
    return np.array(ret).reshape(-1, 7)


def center_to_corner_box2d(boxes_center, coordinate='lidar', T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 5) -> (N, 4, 2)
    N = boxes_center.shape[0]
    boxes3d_center = np.zeros((N, 7))
    boxes3d_center[:, [0, 1, 4, 5, 6]] = boxes_center
    boxes3d_corner = center_to_corner_box3d(
        boxes3d_center, coordinate=coordinate, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)

    return boxes3d_corner[:, 0:4, 0:2]


def center_to_corner_box3d(boxes_center, coordinate='lidar', T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    if coordinate == 'camera':
        boxes_center = camera_to_lidar_box(boxes_center, T_VELO_2_CAM, R_RECT_0)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
            np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    if coordinate == 'camera':
        for idx in range(len(ret)):
            ret[idx] = lidar_to_camera_point(ret[idx], T_VELO_2_CAM, R_RECT_0)

    return ret


def corner_to_center_box2d(boxes_corner, coordinate='lidar', T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 4, 2) -> (N, 5)  x,y,w,l,r
    N = boxes_corner.shape[0]
    boxes3d_corner = np.zeros((N, 8, 3))
    boxes3d_corner[:, 0:4, 0:2] = boxes_corner
    boxes3d_corner[:, 4:8, 0:2] = boxes_corner
    boxes3d_center = corner_to_center_box3d(
        boxes3d_corner, coordinate=coordinate, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)

    return boxes3d_center[:, [0, 1, 4, 5, 6]]


def corner_to_standup_box2d(boxes_corner):
    # (N, 4, 2) -> (N, 4) x1, y1, x2, y2
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)

    return standup_boxes2d


# TODO: 0/90 may be not correct
def anchor_to_standup_box2d(anchors):
    # (N, 4) -> (N, 4) x,y,w,l -> x1,y1,x2,y2
    anchor_standup = np.zeros_like(anchors)
    # r == 0
    anchor_standup[::2, 0] = anchors[::2, 0] - anchors[::2, 3] / 2
    anchor_standup[::2, 1] = anchors[::2, 1] - anchors[::2, 2] / 2
    anchor_standup[::2, 2] = anchors[::2, 0] + anchors[::2, 3] / 2
    anchor_standup[::2, 3] = anchors[::2, 1] + anchors[::2, 2] / 2
    # r == pi/2
    anchor_standup[1::2, 0] = anchors[1::2, 0] - anchors[1::2, 2] / 2
    anchor_standup[1::2, 1] = anchors[1::2, 1] - anchors[1::2, 3] / 2
    anchor_standup[1::2, 2] = anchors[1::2, 0] + anchors[1::2, 2] / 2
    anchor_standup[1::2, 3] = anchors[1::2, 1] + anchors[1::2, 3] / 2

    return anchor_standup


def corner_to_center_box3d(boxes_corner, coordinate='camera', T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 8, 3) -> (N, 7) x,y,z,h,w,l,ry/z
    if coordinate == 'lidar':
        for idx in range(len(boxes_corner)):
            boxes_corner[idx] = lidar_to_camera_point(boxes_corner[idx], T_VELO_2_CAM, R_RECT_0)
    ret = []
    for roi in boxes_corner:
        if True:  # average version or max
            roi = np.array(roi)
            h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4)
            w = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            ) / 4
            l = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            ) / 4
            x = np.sum(roi[:, 0], axis=0)/ 8
            y = np.sum(roi[0:4, 1], axis=0)/ 4
            z = np.sum(roi[:, 2], axis=0)/ 8
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
            if w > l:
                w, l = l, w
                ry = angle_in_limit(ry + np.pi / 2)
        else:  # max version
            h = max(abs(roi[:4, 1] - roi[4:, 1]))
            w = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            )
            l = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            )
            x = np.sum(roi[:, 0], axis=0)/ 8
            y = np.sum(roi[0:4, 1], axis=0)/ 4
            z = np.sum(roi[:, 2], axis=0)/ 8
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
            if w > l:
                w, l = l, w
                ry = angle_in_limit(ry + np.pi / 2)
        ret.append([x, y, z, h, w, l, ry])
    if coordinate == 'lidar':
        ret = camera_to_lidar_box(np.array(ret), T_VELO_2_CAM, R_RECT_0)

    return np.array(ret)


# this just for visulize and testing
def lidar_box3d_to_camera_box(boxes3d, cal_projection=False, P2 = None, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 7) -> (N, 4)/(N, 8, 2)  x,y,z,h,w,l,rz -> x1,y1,x2,y2/8*(x, y)
    num = len(boxes3d)
    boxes2d = np.zeros((num, 4), dtype=np.int32)
    projections = np.zeros((num, 8, 2), dtype=np.float32)

    lidar_boxes3d_corner = center_to_corner_box3d(boxes3d, coordinate='lidar', T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)
    for n in range(num):
        box3d = lidar_boxes3d_corner[n]
        box3d = lidar_to_camera_point(box3d, T_VELO_2_CAM, R_RECT_0)
        points = np.hstack((box3d, np.ones((8, 1)))).T  # (8, 4) -> (4, 8)
        points = np.matmul(P2, points).T
        points[:, 0] /= points[:, 2]
        points[:, 1] /= points[:, 2]

        projections[n] = points[:, 0:2]
        minx = int(np.min(points[:, 0]))
        maxx = int(np.max(points[:, 0]))
        miny = int(np.min(points[:, 1]))
        maxy = int(np.max(points[:, 1]))

        boxes2d[n, :] = minx, miny, maxx, maxy

    return projections if cal_projection else boxes2d


def lidar_to_bird_view_img(lidar, res_factor=4):
    # Input:
    #   lidar: (N', 4)
    # Output:
    #   birdview: (w, l, 3)
    birdview = np.zeros((H * res_factor, W * res_factor, 1))
    for point in lidar:
        x, y = point[0:2]
        if min(range_x) < x < max(range_x) and min(range_y) < y < max(range_y):
            x = int((x - min(range_x)) / vox_width * res_factor)
            y = int((y - min(range_y)) / vox_height * res_factor)
            birdview[y, x] += 1
    birdview = birdview - np.min(birdview)
    divisor = np.max(birdview) - np.min(birdview)
    # TODO: adjust this factor
    birdview = np.clip(5* (birdview / divisor * 255)  * res_factor, a_min=0, a_max=255)
    birdview = np.tile(birdview, 3).astype(np.uint8)
    return birdview


def draw_lidar_box3d_on_image(img, boxes3d, scores, gt_boxes3d=np.array([]),
                              color=(0, 255, 0), gt_color=(255, 0, 255), thickness=2, P2 = None, T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   img: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = img.copy()
    projections = lidar_box3d_to_camera_box(boxes3d, cal_projection=True, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)
    gt_projections = lidar_box3d_to_camera_box(gt_boxes3d, cal_projection=True, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)

    # draw projections
    for qs in projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)
    # draw gt projections
    for qs in gt_projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    


def draw_lidar_box3d_on_birdview(birdview, boxes3d, scores, gt_boxes3d=np.array([]),
                                 color=(0, 0, 255), gt_color=(255, 40, 100), thickness=2, factor=4, P2 = None, T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   birdview: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = birdview.copy()
    corner_boxes3d = center_to_corner_box3d(boxes3d, coordinate='lidar', T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)
    corner_gt_boxes3d = center_to_corner_box3d(gt_boxes3d, coordinate='lidar', T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)
    # draw gt
    for box in corner_gt_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 gt_color, thickness, cv2.LINE_AA)

    # draw detections
    for box_index in range(len(corner_boxes3d)):
        box = corner_boxes3d[box_index]
        score = scores[box_index]
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 color, thickness, cv2.LINE_AA)
        cv2.putText(img,str(score),(int(x0),int(y0)),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def box3d_to_label(batch_box3d, batch_cls, batch_score=[], coordinate='camera', P2 = None, T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   (N, N', 7) x y z h w l r
    #   (N, N')
    #   cls: (N, N') 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate(input): 'camera' or 'lidar'
    # Output:
    #   label: (N, N') N batches and N lines
    batch_label = []
    if batch_score:
        template = '{} ' + ' '.join(['{:.4f}' for i in range(15)]) + '\n'
        for boxes, scores, clses in zip(batch_box3d, batch_score, batch_cls):
            label = []
            for box, score, cls in zip(boxes, scores, clses):
                if coordinate == 'camera':
                    box3d = box
                    box2d = lidar_box3d_to_camera_box(
                        camera_to_lidar_box(box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0), cal_projection=False, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)[0]
                else:
                    box3d = lidar_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0)[0]
                    box2d = lidar_box3d_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), cal_projection=False, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)[0]
                x, y, z, h, w, l, r = box3d
                box3d = [h, w, l, x, y, z, r]
                label.append(template.format(
                    cls, 0, 0, 0, *box2d, *box3d, float(score)))
            batch_label.append(label)
    else:
        template = '{} ' + ' '.join(['{:.4f}' for i in range(14)]) + '\n'
        for boxes, clses in zip(batch_box3d, batch_cls):
            label = []
            for box, cls in zip(boxes, clses):
                if coordinate == 'camera':
                    box3d = box
                    box2d = lidar_box3d_to_camera_box(
                        camera_to_lidar_box(box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0), cal_projection=False,  P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)[0]
                else:
                    box3d = lidar_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0)[0]
                    box2d = lidar_box3d_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), cal_projection=False, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)[0]
                x, y, z, h, w, l, r = box3d
                box3d = [h, w, l, x, y, z, r]
                label.append(template.format(cls, 0, 0, 0, *box2d, *box3d))
            batch_label.append(label)

    return np.array(batch_label)


# BOTTLENECK
def delta_to_boxes3d(deltas, anchors, coordinate='lidar'):
    # Input:
    #   deltas: (N, w, l, 14)
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)

    # Ouput:
    #   boxes3d: (N, w*l*2, 7)
    #deltas = deltas.detach().numpy()
    anchors_reshaped = anchors.reshape(-1, 7)
    deltas = deltas.reshape(deltas.shape[0], -1, 7)
    anchors_d = np.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)
    boxes3d = np.zeros_like(deltas)
    boxes3d[..., [0, 1]] = deltas[..., [0, 1]] * anchors_d[:, np.newaxis] + anchors_reshaped[..., [0, 1]]
    boxes3d[..., [2]] = deltas[..., [2]] * ANCHOR_H + anchors_reshaped[..., [2]]
    boxes3d[..., [3, 4, 5]] = np.exp(deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
    boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

    return boxes3d

