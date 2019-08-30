from __future__ import division
from __future__ import print_function
from __future__ import with_statement
from __future__ import absolute_import

import time
import math
import os
import sys
import yaml
from tqdm import tqdm
import numpy as np
from glob import glob
import torch.nn.functional as F
import cv2
from plot_util import corner_to_standup_box2d,center_to_corner_box2d,delta_to_boxes3d,draw_lidar_box3d_on_image,draw_lidar_box3d_on_birdview,lidar_to_bird_view_img
#from utils.utils import *
from utils.nms import nms
from data.data import KITDataset 

from VoxelNet import VoxelNet
from torch.autograd import Variable
import torch
import torch.utils.data as data
import torch.backends.cudnn
import torch.optim as optim
import torch.nn.init as init
import warnings
warnings.filterwarnings("ignore")

yamlPath = "configure.yaml"
f = open(yamlPath, 'r', encoding='utf-8')
conf = f.read()
conf_dict = yaml.safe_load(conf) 

#if_cuda = False
if_cuda = True if conf_dict["if_cuda"] == 1 and torch.cuda.is_available() else False
classes = conf_dict["classes"]
parent_path = './'
chk_pth = parent_path+ "checkpoints/chk_Car_122.pth"

if if_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("-"*20)
print("if_cuda:",if_cuda)
batch_size = conf_dict['batch_size']
range_x=conf_dict['range_x']
range_y=conf_dict['range_y']
range_z=conf_dict['range_z']
vox_depth = conf_dict['vox_d']
vox_width = conf_dict['vox_w']
vox_height = conf_dict['vox_h']
pt_thres_per_vox = conf_dict['pt_thres_per_vox'] 
anchors_per_vox = conf_dict['anchors_per_vox']
pos_threshold = conf_dict['iou_pos_threshold']
neg_threshold = conf_dict['iou_neg_threshold']

ANCHOR_L = conf_dict['ANCHOR_L']
ANCHOR_W = conf_dict['ANCHOR_W']
ANCHOR_H = conf_dict['ANCHOR_H']

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
w = np.ones_like(cx) * ANCHOR_W
l = np.ones_like(cx) * ANCHOR_L
h = np.ones_like(cx) * ANCHOR_H
r = np.ones_like(cx)
r[..., 0] = 0
r[..., 1] = np.pi/2
anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1).reshape(-1,7)

def detection_collate(batch):
    voxel_features = list()
    voxel_coords = list()
    gt_box3ds = list()
    lidars = list()
    images = list()
    calibs = list()
    ids = list()
    
    for i, item in enumerate(batch):
        voxel_features.append(item[0])
        voxel_coords.append(
            np.pad(item[1], ((0, 0), (1, 0)),
                mode='constant', constant_values=i))
       
        gt_box3ds.append(item[2])
        images.append(item[3])
        lidars.append(item[4])
        calibs.append(item[5])
        ids.append(item[6])
        
    return np.concatenate(voxel_features), np.concatenate(voxel_coords), gt_box3ds, \
           images, lidars, calibs, ids


def inference(setting="val"):#test,val
    kit_dataset= KITDataset(conf_dict=conf_dict,setting="val2",root_path='./data/dataset')#test,val
    kit_data_loader = data.DataLoader(kit_dataset, batch_size=batch_size, num_workers=4,collate_fn=detection_collate,pin_memory=True)

    print('Loading pre-trained weights...')
    #chk = glob(chk_pth+'/*')[-1]
    net = VoxelNet()
    if if_cuda:
        net.cuda()
    net.load_state_dict(torch.load(chk_pth,map_location='cpu'))
    net.eval()
    for batch_index, contents in enumerate(tqdm(kit_data_loader)):
      
        voxel_features, voxel_coords, gt_box3d, images, lidars, calibs, ids = contents
        print(ids)
        voxel_features = Variable(torch.FloatTensor(voxel_features))
        if if_cuda:
            voxel_features = voxel_features.cuda()
 
        psm, rm = net(voxel_features, voxel_coords)
        psm = F.sigmoid(psm.permute(0,2,3,1))
        
        psm = psm.reshape((batch_size, -1))
        rm = rm.permute(0,2,3,1).contiguous()
        rm = rm.view(rm.size(0),rm.size(1),rm.size(2),14)#([batch, 200, 176, 2, 7])
        batch_boxes3d = delta_to_boxes3d(rm, anchors, coordinate='lidar')
        batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
        print(np.shape(batch_boxes3d),np.shape(batch_boxes2d))
        #rm = rm.view(rm.size(0),-1,14)
        #rm = rm.detach().numpy()
        psm = psm.detach().numpy()
        torch.cuda.empty_cache()
        
        ret_box3d = []
        ret_score = []
        
        for sid in range(batch_size):
            # remove box with low score
            ind = np.where(psm[sid, :] >= 0.95)[0]
            tmp_boxes3d = batch_boxes3d[sid, ind, ...]
            tmp_boxes2d = batch_boxes2d[sid, ind, ...]
            tmp_scores = psm[sid, ind]
            boxes_2d = corner_to_standup_box2d(center_to_corner_box2d(tmp_boxes2d, coordinate='lidar'))
            
            tmp_scores = tmp_scores
            boxes2d_cat = np.concatenate((boxes_2d,tmp_scores[...,np.newaxis]),axis=1)
            
            ind_nms = nms(boxes2d_cat,thresh=0.1)
            #ind_nms = ind
            tmp_boxes2d = tmp_boxes2d[ind_nms, ...]
            tmp_boxes3d = tmp_boxes3d[ind_nms, ...]
            tmp_scores  = tmp_scores [ind_nms]
            ret_box3d.append(tmp_boxes3d)
            ret_score.append(tmp_scores)
            
        for sid in range(batch_size):
            Tr_velo_to_cam = calibs[sid]['Tr_velo_to_cam']
            R_cam_to_rect = calibs[sid]['R_cam_to_rect']
            P = calibs[sid]['P']
            
            front_image = draw_lidar_box3d_on_image(images[sid], ret_box3d[sid], 
                                                    ret_score[sid], gt_box3d[sid], 
                                                    P2=P, T_VELO_2_CAM=Tr_velo_to_cam, R_RECT_0=R_cam_to_rect)
            #img = point_cloud_2_birdseye(lidars[sid])
            #for b in ret_box3d[sid]:
            #    print (b)
             #   #b = list(map(lambda x:eval(x),b))
             #   b = box3d_cam_to_velo(b, Tr_velo_to_cam)
             #   bbox_bev = bbox3d_2_birdeye(b,mode ="top")
             #   cv2.rectangle(img,(bbox_bev[0], bbox_bev[1]), (bbox_bev[2], bbox_bev[3]), (255,255,0),1)    

            
            bird_view = lidar_to_bird_view_img(lidars[sid], factor=3)
            bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[sid], 
                                                     ret_score[sid],gt_box3d[sid], 
                                                     factor=3, P2=P, 
                                                     T_VELO_2_CAM=Tr_velo_to_cam, R_RECT_0=R_cam_to_rect)
            
            cv2.imwrite(parent_path+"predicts/"+ids[sid]+"_front.jpg",front_image)
            cv2.imwrite(parent_path+"predicts/"+ids[sid]+"_brids.jpg",bird_view)
            
            #heatmap = colorize(psm[0, ...], 4)
            #cv2.imwrite(parent_path+"predicts/"+ids[0]+"_heat.jpg",heatmap)
        
    
if __name__ == '__main__':
    
      inference()
    
    
    
