from __future__ import division
from __future__ import print_function
from __future__ import with_statement
from __future__ import absolute_import

import os
import sys
from utils.file_load import *
from utils.utils import box3d_corner_to_center_batch, anchors_center_to_corner, corner_to_standup_box2d_batch
from box_overlaps import bbox_overlaps
from data_aug import aug_data

import torch.utils.data as data
import numpy as np
import torch
import cv2
import yaml


class KITDataset(data.Dataset):
    def __init__(self, conf_dict, root_path='/home/screentest/dataset/voxelnet',setting='train',data_type='velodyne_train'):
        
        
        self.data_root_path = root_path
        self.data_type = data_type
        self.setting = setting
        self.record_path = os.path.join(root_path,setting+'.txt')
        with open(self.record_path) as f:
            lines = f.readlines()                            
            self.file_paths = list(map(lambda x:x.strip('\n'),lines))
                          
        self.range_x=conf_dict['range_x']
        self.range_y=conf_dict['range_y']
        self.range_z=conf_dict['range_z']
        self.vox_depth = conf_dict['vox_d']
        self.vox_width = conf_dict['vox_w']
        self.vox_height = conf_dict['vox_h']
        self.classes = conf_dict['classes'] 
        self.pt_thres_per_vox = conf_dict['pt_thres_per_vox'] 
        self.anchors_per_vox = conf_dict['anchors_per_vox']
        self.pos_threshold = conf_dict['iou_pos_threshold']
        self.neg_threshold = conf_dict['iou_neg_threshold']
                                        
        self.H = (max(self.range_x)-min(self.range_x))//self.vox_height
        self.W = (max(self.range_y)-min(self.range_y))//self.vox_width
        self.D = (max(self.range_z)-min(self.range_z))//self.vox_depth
       
        self.feature_map_shape = (int(self.H / 2), int(self.W / 2))
                                        
        x = np.linspace(self.range_x[0]+self.vox_width, self.range_x[1]-self.vox_width, self.W/2)
        y = np.linspace(self.range_y[0]+self.vox_height, self.range_x[1]-self.vox_height, self.H/2)
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
        self.anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1)
        self.anchors = self.anchors.reshape(-1,7)
                                
                                        
    def cal_target(self, gt_box3d):
        # Input:
        #   labels: (N,)
        #   feature_map_shape: (w, l)
        #   anchors: (w, l, 2, 7)
        # Output:
        #   pos_equal_one (w, l, 2)
        #   neg_equal_one (w, l, 2)
        #   targets (w, l, 14)
        # attention: cal IoU on birdview

        anchors_d = np.sqrt(self.anchors[:, 4] ** 2 + self.anchors[:, 5] ** 2)
        pos_equal_one = np.zeros((*self.feature_map_shape, 2))
        neg_equal_one = np.zeros((*self.feature_map_shape, 2))
        targets = np.zeros((*self.feature_map_shape, 14))

        gt_xyzhwlr = box3d_corner_to_center_batch(gt_box3d)

        anchors_corner = anchors_center_to_corner(self.anchors)

        anchors_standup_2d = corner_to_standup_box2d_batch(anchors_corner)
        # BOTTLENECK
        gt_standup_2d = corner_to_standup_box2d_batch(gt_box3d)

        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        id_highest = np.argmax(iou.T, axis=1)  # the maximum anchor's ID
        id_highest_gt = np.arange(iou.T.shape[0])
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]
        # find anchor iou > cfg.XXX_POS_IOU
        id_pos, id_pos_gt = np.where(iou > self.pos_threshold)
        # find anchor iou < cfg.XXX_NEG_IOU
        id_neg = np.where(np.sum(iou < self.neg_threshold,
                                 axis=1) == iou.shape[1])[0]

        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        # TODO: uniquify the array in a more scientific way
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()
        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*self.feature_map_shape, self.anchors_per_position))
        pos_equal_one[index_x, index_y, index_z] = 1
        # ATTENTION: index_z should be np.array

        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_xyzhwlr[id_pos_gt, 0] - self.anchors[id_pos, 0]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_xyzhwlr[id_pos_gt, 1] - self.anchors[id_pos, 1]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_xyzhwlr[id_pos_gt, 2] - self.anchors[id_pos, 2]) / self.anchors[id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_xyzhwlr[id_pos_gt, 3] / self.anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_xyzhwlr[id_pos_gt, 4] / self.anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_xyzhwlr[id_pos_gt, 5] / self.anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_xyzhwlr[id_pos_gt, 6] - self.anchors[id_pos, 6])
        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*self.feature_map_shape, self.anchors_per_position))
        neg_equal_one[index_x, index_y, index_z] = 1
        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*self.feature_map_shape, self.anchors_per_position))
        neg_equal_one[index_x, index_y, index_z] = 0
        return pos_equal_one, neg_equal_one, targets

    def voxelize(self, lidar): #preprocessing
        voxel_coords = ((lidar[:, :3] - np.array([self.range_X[0], self.range_y[0], self.range_z[0]])) / (
                        self.vox_width, self.vox_height, self.vox_depth)).astype(np.int32)
        # convert to  (D, H, W)
        voxel_coords = voxel_coords[:,[2,1,0]]                    
        # unique voxel coordinates, index in original array of each element in unique array
        voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0, \
                                                  return_inverse=True, return_counts=True)
        voxel_features = list()                 
        for i in range(len(voxel_coords)):
            voxel = np.zeros((self.pt_thres_per_vox,7), dtype=np.float32)
            #all pts belong to that voxel
            pts = lidar[inv_ind == i]  
            if voxel_counts[i] > self.pt_thres_per_vox:
                pts = pts[:self.pt_thres_per_vox, :]
                voxel_counts[i] = self.pt_thres_per_vox
            # normalize all pts in this voxel and append it colomnwise behind voxel coords
            voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
            voxel_features.append(voxel)
        voxel_features = np.array(voxel_features)
        return voxel_features, voxel_coord
    

    
                                        
    def __getitem__(self, index):
        image_file_path, lidar_file_path, calib_file_path, label_file_path = generate_file_path(index,self.data_root_path)
        calib = read_cal(calib_file_path)['Tr_velo2cam']
        lidars = read_velodyne_points(lidar_file_path)
        lidars,_ = prepare_velodyne_points(lidars, range_x = self.range_x,range_y = self.range_y, range_z = self.range_z)         
        image = cv2.imread(image_file_path)  
        gt_box3d = read_label(label_file_path,T)        
        
        if self.setting=='train':
            # online data augmentation
            lidar, gt_box3d = aug_data(lidar, gt_box3d)
            # specify a range
            lidar, gt_box3d = utils.get_filtered_lidar(lidar, gt_box3d)
            # voxelize
            voxel_features, voxel_coords = self.voxelize(vel)
            # bounding-box encoding
            pos_equal_one, neg_equal_one, targets = self.cal_target(gt_box3d)
            return voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, image, calib, self.file_paths[index]
        
       
        else:
            lidar, gt_box3d = utils.get_filtered_lidar(lidar, gt_box3d)
            voxel_features, voxel_coords = self.voxelize(vel)
            pos_equal_one, neg_equal_one, targets = self.cal_target(gt_box3d)
            
            return voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, image, calib, self.file_paths[index]
        

    def __len__(self):
        return len(self.file_paths)
