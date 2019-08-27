from __future__ import division
import numpy as np
import math
import cv2
from box_overlaps import *
from data_aug import aug_data
import yaml

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
classes = conf_dict['classes'] 
pt_thres_per_vox = conf_dict['pt_thres_per_vox'] 
anchors_per_vox = conf_dict['anchors_per_vox']
pos_threshold = conf_dict['iou_pos_threshold']
neg_threshold = conf_dict['iou_neg_threshold']

W = math.ceil((max(range_x)-min(range_x))/vox_width)
H = math.ceil((max(range_y)-min(range_y))/vox_height)
D = math.ceil((max(range_z)-min(range_z))/vox_depth)
feature_map_shape = (int(H / 2), int(W / 2))
x = np.linspace(range_x[0]+vox_width, range_x[1]-vox_width, int(W/2))
y = np.linspace(range_y[0]+vox_height, range_x[1]-vox_height, int(H/2))
cx, cy = np.meshgrid(x, y)
cx = np.tile(cx[..., np.newaxis], 2)
cy = np.tile(cy[..., np.newaxis], 2)
cz = np.ones_like(cx) * (-1.0)
w = np.ones_like(cx) * 1.6
l = np.ones_like(cx) * 3.9
h = np.ones_like(cx) * 1.56
r = np.ones_like(cx)
r[..., 0] = 0
r[..., 1] = np.pi/2
anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1).reshape(-1,7)
        
def get_filtered_lidar(lidar, boxes3d=None):

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]

    filter_x = np.where((pxs >= range_x[0]) & (pxs < range_x[1]))[0]
    filter_y = np.where((pys >= range_y[0]) & (pys < range_y[1]))[0]
    filter_z = np.where((pzs >= range_z[0]) & (pzs < range_z[1]))[0]
    filter_xy = np.intersect1d(filter_x, filter_y)
    filter_xyz = np.intersect1d(filter_xy, filter_z)

    if boxes3d is not None:
        box_x = (boxes3d[:, :, 0] >= range_x[0]) & (boxes3d[:, :, 0] < range_x[1])
        box_y = (boxes3d[:, :, 1] >= range_y[0]) & (boxes3d[:, :, 1] < range_y[1])
        box_z = (boxes3d[:, :, 2] >= range_z[0]) & (boxes3d[:, :, 2] < range_z[1])
        box_xyz = np.sum(box_x & box_y & box_z,axis=1)

        return lidar[filter_xyz], boxes3d[box_xyz>0]

    return lidar[filter_xyz]

def lidar_to_bev(lidar):

    X0, Xn = 0, W
    Y0, Yn = 0, H
    Z0, Zn = 0, D

    width  = Yn - Y0
    height   = Xn - X0
    channel = Zn - Z0  + 2

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]

    qxs=((pxs-range_x[0])/vox_width).astype(np.int32)
    qys=((pys-range_y[0])/vox_height).astype(np.int32)
    qzs=((pzs-range_z[0])/vox_depth).astype(np.int32)

    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)
    mask = np.ones(shape=(height,width,channel-1), dtype=np.float32)* -5

    for i in range(len(pxs)):
        top[-qxs[i], -qys[i], -1]= 1+ top[-qxs[i], -qys[i], -1]
        if pzs[i]>mask[-qxs[i], -qys[i],qzs[i]]:
            top[-qxs[i], -qys[i], qzs[i]] = max(0,pzs[i]-range_z[0])
            mask[-qxs[i], -qys[i],qzs[i]]=pzs[i]
        if pzs[i]>mask[-qxs[i], -qys[i],-1]:
            mask[-qxs[i], -qys[i],-1]=pzs[i]
            top[-qxs[i], -qys[i], -2]=prs[i]


    top[:,:,-1] = np.log(top[:,:,-1]+1)/math.log(64)

    if 1:
        # top_image = np.sum(top[:,:,:-1],axis=2)
        density_image = top[:,:,-1]
        density_image = density_image-np.min(density_image)
        density_image = (density_image/np.max(density_image)*255).astype(np.uint8)
        # top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)

    return top, density_image

def project_velo2rgb(velo,calib):
    T=np.zeros([4,4],dtype=np.float32)
    T[:3,:]=calib['Tr_velo2cam']
    T[3,3]=1
    R=np.zeros([4,4],dtype=np.float32)
    R[:3,:3]=calib['R0']
    R[3,3]=1
    num=len(velo)
    projections = np.zeros((num,8,2),  dtype=np.int32)
    for i in range(len(velo)):
        box3d=np.ones([8,4],dtype=np.float32)
        box3d[:,:3]=velo[i]
        M=np.dot(calib['P2'],R)
        M=np.dot(M,T)
        box2d=np.dot(M,box3d.T)
        box2d=box2d[:2,:].T/box2d[2,:].reshape(8,1)
        projections[i] = box2d
    return projections

def _quantize_coords(x, y):
    xx = H - int((y - range_y[0]) / vox_height)
    yy = W - int((x - range_x[0]) / vox_width)
    return xx, yy

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

def box3d_cam_to_velo(box3d, Tr):

    def project_cam2velo(cam, Tr):
        T_inv = np.linalg.inv(Tr)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2

        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2*np.pi + angle

        return angle

    h,w,l,tx,ty,tz,ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)

    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])

    rz = ry_to_rz(ry)

    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])

    velo_box = np.dot(rotMat, Box)

    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T

    box3d_corner = cornerPosInVelo.transpose()

    return box3d_corner.astype(np.float32)

def anchors_center_to_corner(anchors):
    
    N = anchors.shape[0]
    anchor_corner = np.zeros((N, 4, 2))
    for i in range(N):
        anchor = anchors[i]
        translation = anchor[0:3]
        h, w, l = anchor[3:6]
        rz = anchor[-1]
        Box = np.array([
            [-l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2]])
        # re-create 3D bounding box in velodyne coordinate system
        rotMat = np.array([
            [np.cos(rz), -np.sin(rz)],
            [np.sin(rz), np.cos(rz)]])
        velo_box = np.dot(rotMat, Box)
        cornerPosInVelo = velo_box + np.tile(translation[:2], (4, 1)).T
        anchor_corner[i] = cornerPosInVelo.transpose()
    return anchor_corner


def corner_to_standup_box2d_batch(boxes_corner):
    # (N, 4, 2) -> (N, 4) x1, y1, x2, y2
    assert boxes_corner.ndim == 3
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)
    return standup_boxes2d

def box3d_corner_to_center_batch(box3d_corner):
    # (N, 8, 3) -> (N, 7)
    assert box3d_corner.ndim == 3
    batch_size = box3d_corner.shape[0]
    #mid
    xyz = np.mean(box3d_corner[:, :4, :], axis=1)
    # -down_z+up_z
    h = abs(np.mean(box3d_corner[:, 4:, 2] - box3d_corner[:, :4, 2], axis=1, keepdims=True))
    w = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 2, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 5, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 6, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    l = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 1, [0, 1]] - box3d_corner[:, 2, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 5, [0, 1]] - box3d_corner[:, 6, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    theta = (np.arctan2(box3d_corner[:, 2, 1] - box3d_corner[:, 1, 1],
                        box3d_corner[:, 2, 0] - box3d_corner[:, 1, 0]) +
             np.arctan2(box3d_corner[:, 3, 1] - box3d_corner[:, 0, 1],
                        box3d_corner[:, 3, 0] - box3d_corner[:, 0, 0]) +
             np.arctan2(box3d_corner[:, 2, 0] - box3d_corner[:, 3, 0],
                        box3d_corner[:, 3, 1] - box3d_corner[:, 2, 1]) +
             np.arctan2(box3d_corner[:, 1, 0] - box3d_corner[:, 0, 0],
                        box3d_corner[:, 0, 1] - box3d_corner[:, 1, 1]))[:, np.newaxis] / 4

    return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(batch_size, 7)




def cal_iou3d(box1, box2, T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   box1/2: x, y, z, h, w, l, r
    # Output:
    #   iou
    buf1 = np.zeros((cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 3))
    buf2 = np.zeros((cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 3))
    tmp = center_to_corner_box2d(np.array([box1[[0,1,4,5,6]], box2[[0,1,4,5,6]]]), coordinate='lidar', T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)
    box1_corner = batch_lidar_to_bird_view(tmp[0]).astype(np.int32)
    box2_corner = batch_lidar_to_bird_view(tmp[1]).astype(np.int32)
    buf1 = cv2.fillConvexPoly(buf1, box1_corner, color=(1,1,1))[..., 0]
    buf2 = cv2.fillConvexPoly(buf2, box2_corner, color=(1,1,1))[..., 0]
    share = np.sum((buf1 + buf2) == 2)
    area1 = np.sum(buf1)
    area2 = np.sum(buf2)
    
    z1, h1, z2, h2 = box1[2], box1[3], box2[2], box2[3]
    z_intersect = cal_z_intersect(z1, h1, z2, h2)

    return share * z_intersect / (area1 * h1 + area2 * h2 - share * z_intersect)


def cal_box3d_iou(boxes3d, gt_boxes3d, cal_3d=0, T_VELO_2_CAM=None, R_RECT_0=None):
    # Inputs:
    #   boxes3d: (N1, 7) x,y,z,h,w,l,r
    #   gt_boxed3d: (N2, 7) x,y,z,h,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    N1 = len(boxes3d)
    N2 = len(gt_boxes3d)
    output = np.zeros((N1, N2), dtype=np.float32)

    for idx in range(N1):
        for idy in range(N2):
            if cal_3d:
                output[idx, idy] = float(
                    cal_iou3d(boxes3d[idx], gt_boxes3d[idy]), T_VELO_2_CAM, R_RECT_0)
            else:
                output[idx, idy] = float(
                    cal_iou2d(boxes3d[idx, [0, 1, 4, 5, 6]], gt_boxes3d[idy, [0, 1, 4, 5, 6]], T_VELO_2_CAM, R_RECT_0))

    return output



'''
def get_anchor3d(anchors):
    num = anchors.shape[0]
    anchors3d = np.zeros((num,8,3))
    anchors3d[:, :4, :2] = anchors
    anchors3d[:, :, 2] = cfg.z_a
    anchors3d[:, 4:, :2] = anchors
    anchors3d[:, 4:, 2] = cfg.z_a + cfg.h_a
    return anchors3d
'''


