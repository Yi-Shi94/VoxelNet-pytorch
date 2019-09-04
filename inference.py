from __future__ import division
from __future__ import print_function
from __future__ import with_statement
from __future__ import absolute_import

import time
import math
import os
import sys
import yaml
import cv2
from tqdm import tqdm
import numpy as np
from glob import glob
import torch.nn.functional as F
from plot_util import corner_to_standup_box2d,center_to_corner_box2d,delta_to_boxes3d,draw_lidar_box3d_on_image,draw_lidar_box3d_on_birdview,lidar_to_bird_view_img,box3d_to_label
#from utils.utils import *

from data.data import KITDataset 
import VoxelNet
import VoxelNet1
from torch.autograd import Variable
import torch
import torch.utils.data as data
import torch.backends.cudnn
import torch.optim as optim
import torch.nn.init as init
import warnings
warnings.filterwarnings("ignore")

parent_path = './'
yaml_path = "configure.yaml"
f = open(yaml_path, 'r', encoding='utf-8')
conf = f.read()
conf_dict = yaml.safe_load(conf) 

#if_cuda = False
if_cuda = True if conf_dict["if_cuda"] == 1 and torch.cuda.is_available() else False
classes = conf_dict["classes"]

if if_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("-"*20)
print("if_cuda:",if_cuda)
chk_name = conf_dict['chk_name']
chk_pth = parent_path+ "checkpoints/"+ chk_name
net_type = conf_dict['net_type'] 
batch_size = conf_dict['batch_size_test']
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
    kit_dataset= KITDataset(conf_dict=conf_dict,setting="validation")#test,val
    kit_data_loader = data.DataLoader(kit_dataset, batch_size=batch_size, num_workers=4,collate_fn=detection_collate,pin_memory=True)

    print('Loading pre-trained weights...')
    #chk = glob(chk_pth+'/*')[-1]
    if net_type==0:
        net = VoxelNet.VoxelNet()
    else:
        net = VoxelNet1.VoxelNet()

    if if_cuda:
        net.cuda()
        net.load_state_dict(torch.load(chk_pth))
    else:
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
        if if_cuda:
            psm = psm.detach().cpu().numpy()
            rm = rm.detach().cpu().numpy()
        else:
            psm = psm.detach().numpy()
            rm = rm.detach().numpy()
        batch_boxes3d = delta_to_boxes3d(rm, anchors, coordinate='lidar')
        batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
        print(np.shape(batch_boxes3d),np.shape(batch_boxes2d))
        #rm = rm.view(rm.size(0),-1,14)
        #rm = rm.detach().numpy()
        
        torch.cuda.empty_cache()
        
        ret_box3d = []
        ret_score = []
        
        
        for sid in range(batch_size):
            # remove box with low score
            flag = 0
            ind = np.where(psm[sid, :] >= 0.96)[0]
            tmp_boxes3d = batch_boxes3d[sid, ind, ...]
            tmp_boxes2d = batch_boxes2d[sid, ind, ...]
            tmp_scores = psm[sid, ind]
            boxes_2d = corner_to_standup_box2d(center_to_corner_box2d(tmp_boxes2d, coordinate='lidar'))
            
            ind_nms = ind
            if len(tmp_scores)==0:
                flag = 1
                
            if 0 and flag==0:
                from nms import nms
                boxes_2d_c = torch.from_numpy(boxes_2d)
                boxes_2d_c = boxes_2d_c.cuda().contiguous().float()
                tmp_scores_c = torch.from_numpy(tmp_scores)
                tmp_scores_c = tmp_scores_c.cuda().contiguous().float()
                ind_nms,num_to_keep, parent_object_index = nms(boxes_2d_c,tmp_scores_c,overlap=.1,top_k=100)
            elif flag==0:
                from utils.nms import nms
                boxes2d_cat = np.concatenate((boxes_2d,tmp_scores[...,np.newaxis]),axis=1)
                ind_nms = nms(boxes2d_cat,thresh = 0.1)
            
            tmp_boxes2d = tmp_boxes2d[ind_nms, ...]
            tmp_boxes3d = tmp_boxes3d[ind_nms, ...]
            tmp_scores  = tmp_scores [ind_nms]
            ret_box3d.append(tmp_boxes3d)
            ret_score.append(tmp_scores)
            
        ret_box3d_score = list()
        
        for boxes3d, scores in zip(ret_box3d, ret_score):
            ret_box3d_score.append(np.concatenate([np.tile('Car', len(boxes3d))[:, np.newaxis],
                                                   boxes3d, scores[:, np.newaxis]], axis=-1))
        for index,item in enumerate(zip(ids,ret_box3d_score)):
            ida, result = item
            out_path = os.path.join(parent_path,'predicts/data',ida+'.txt')
            
            
            Tr_velo_to_cam = calibs[index]['Tr_velo_to_cam']
            R_cam_to_rect = calibs[index]['R_cam_to_rect']
            P = calibs[index]['P']
            cur_coord = [result[:, 1:8]]
            cur_cls = [result[:, 0]]
            cur_score = [result[:, -1]]
            labels = box3d_to_label(cur_coord, cur_cls, cur_score, coordinate='lidar',
                                    P2=P, T_VELO_2_CAM=Tr_velo_to_cam, R_RECT_0=R_cam_to_rect)[0]
            f = open(out_path,'w')
            for line in labels:
                #print(line)
                f.write(line)
            f.close()
            print('write out {} objects to {}'.format(len(labels),'Car'))
            print('score',result[:,-1])

        if batch_index % 20 != 0 and len(labels)<=1:
            continue    
            
        for sid in range(batch_size):
            Tr_velo_to_cam = calibs[sid]['Tr_velo_to_cam']
            R_cam_to_rect = calibs[sid]['R_cam_to_rect']
            P = calibs[sid]['P']
            
            front = draw_lidar_box3d_on_image(images[sid], 
                                                    ret_box3d[sid], 
                                                    ret_score[sid], 
                                                    gt_box3d[sid], 
                                                    P2=P, T_VELO_2_CAM=Tr_velo_to_cam, R_RECT_0=R_cam_to_rect)
            #img = point_cloud_2_birdseye(lidars[sid])
            
            bird_view = draw_lidar_box3d_on_birdview(lidar_to_bird_view_img(lidars[sid]), 
                                                     ret_box3d[sid], 
                                                     ret_score[sid],
                                                     gt_box3d[sid], 
                                                     P2=P, T_VELO_2_CAM=Tr_velo_to_cam, R_RECT_0=R_cam_to_rect)
            
            cv2.imwrite(parent_path+"predictions/img/"+ids[sid]+"_front.jpg",front)
            cv2.imwrite(parent_path+"predictions/img/"+ids[sid]+"_brids.jpg",bird_view)
            
            #heatmap = colorize(psm[0, ...], 4)
            #cv2.imwrite(parent_path+"predicts/"+ids[0]+"_heat.jpg",heatmap)
        
    
if __name__ == '__main__':
    
      inference()
    
    
    
