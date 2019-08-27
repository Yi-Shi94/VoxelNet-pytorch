from __future__ import division
from __future__ import print_function
from __future__ import with_statement
from __future__ import absolute_import

import time
import os
import sys
import yaml
from tqdm import tqdm
import numpy as np
from glob import glob
import torch.nn.functional as F
import cv2
from plot import *
#from utils.utils import *
#from utils.coord_transform import *
from utils.nms import nms
#from utils.file_load import read_cal
from data.data import KITDataset 
from box_overlaps import bbox_overlaps
from data_aug import aug_data

from VoxelNet import VoxelNet,weights_init

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
#parent_path = "/home/screentest/ys3237/VoxelNet-pytorch/"
parent_path = './'
chk_pth = parent_path+ "checkpoints/chk_Car_55.pth"

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
    voxel_features = []
    voxel_coords = []
    gt_box3ds = []
    lidars = []
    images = []
    calibs = []
    ids = []
    
    for i, item in enumerate(batch):
        voxel_features.append(item[0])
        voxel_coords.append(
            np.pad(item[1], ((0, 0), (1, 0)),
                mode='constant', constant_values=i))
       
        #pos_equal_one.append(sample[2])
        #neg_equal_one.append(sample[3])
        #targets.append(sample[4])
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
         # wrapper to variable
        print(ids)
        #print(np.shape(voxel_features),np.shape(voxel_coords))
        voxel_features = Variable(torch.FloatTensor(voxel_features))
        if if_cuda:
            voxel_features = voxel_features.cuda()
            #pos_equal_one = Variable(torch.cuda.FloatTensor(pos_equal_one))
            #neg_equal_one = Variable(torch.cuda.FloatTensor(neg_equal_one))
            #targets = Variable(torch.cuda.FloatTensor(targets))
            #pos_equal_one = Variable(torch.FloatTensor(pos_equal_one))
            #neg_equal_one = Variable(torch.FloatTensor(neg_equal_one))
            #targets = Variable(torch.FloatTensor(targets))
            # zero the parameter gradients
 
        psm, rm = net(voxel_features, voxel_coords)
        psm = psm.reshape((batch_size, -1))
        rm = rm.permute(0,2,3,1).contiguous()
        rm = rm.view(rm.size(0),rm.size(1),rm.size(2),14)#([batch, 200, 176, 2, 7])
        print(rm.shape,psm.shape)
        batch_boxes3d = delta_to_boxes3d(rm, anchors, coordinate='lidar')
        batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
        print(np.shape(batch_boxes3d),np.shape(batch_boxes2d))
        ret_box3d = []
        ret_score = []
        
        rm = rm.view(rm.size(0),-1,14)
        for sid in range(batch_size):
            # remove box with low score
            ind = np.where(psm[sid, :] >= 0.98)[0]
            tmp_boxes3d = batch_boxes3d[sid, ind, ...]
            tmp_boxes2d = batch_boxes2d[sid, ind, ...]
            print(sid,len(ind),rm.shape)
            tmp_scores = psm[sid, ind]
            boxes_2d = corner_to_standup_box2d(center_to_corner_box2d(tmp_boxes2d, coordinate='lidar'))
            
            tmp_scores = tmp_scores.detach().numpy()
            print("2dd",np.shape(boxes_2d),np.shape(tmp_scores))
            
            boxes2d_cat = np.concatenate((boxes_2d,tmp_scores[...,np.newaxis]),axis=1)
            ind_after_nms = nms(boxes2d_cat,thresh=0.1)
            
            print("2dd",np.shape(boxes2d_cat),np.shape(ind_after_nms))
            tmp_boxes2d = tmp_boxes2d[ind_after_nms, ...]
            tmp_boxes3d = tmp_boxes3d[ind_after_nms, ...]
            tmp_scores  = tmp_scores [ind_after_nms]
            ret_box3d.append(tmp_boxes3d)
            ret_score.append(tmp_scores)
        print(np.shape(ret_box3d[0]),np.shape(ret_score))
        
        ret_box3d_score = []
        
        for boxes3d, scores in zip(ret_box3d, ret_score):
            ret_box3d_score.append(np.concatenate([np.tile(classes, len(boxes3d))[:, np.newaxis],
                                                   boxes3d, scores[:, np.newaxis]], axis=-1))
        print(np.shape(ret_box3d_score))
        
        for sid in range(batch_size):
            Tr_velo_to_cam = calibs[sid]['Tr_velo_to_cam']
            R_cam_to_rect = calibs[sid]['R_cam_to_rect']
            P = calibs[sid]['P']
            front_image = draw_lidar_box3d_on_image(images[sid], ret_box3d[sid], 
                                                    ret_score[sid],gt_box3d[sid], 
                                                    P2=P, T_VELO_2_CAM=Tr_velo_to_cam, R_RECT_0=R_cam_to_rect)
           
            bird_view = lidar_to_bird_view_img(lidars[sid], factor=4)
            bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[sid], 
                                                     ret_score[sid],gt_box3d[sid], 
                                                     factor=4, P2=P, 
                                                     T_VELO_2_CAM=Tr_velo_to_cam, R_RECT_0=R_cam_to_rect)
            cv2.imwrite(parent_path+"predicts/"+ids[sid]+"_front.jpg",front_image)
            cv2.imwrite(parent_path+"predicts/"+ids[sid]+"_brids.jpg",bird_view)
            torch.cuda.empty_cache()
            #heatmap = colorize(psm[0, ...], 4)
            #cv2.imwrite(parent_path+"predicts/"+ids[0]+"_heat.jpg",heatmap)
        
            
            
        '''
        rm = rm.permute(0,2,3,1).contiguous()
        rm = rm.view(rm.size(0),rm.size(1),rm.size(2),-1,7)#([batch, 200, 176, 2, 7])
        rm_pos = rm.view(-1,7).detach().cpu().numpy()
        p_pos = F.sigmoid(psm.permute(0,2,3,1))#([batch, 200, 176, 2])
        p_pos = p_pos.view(-1,1).detach().cpu().numpy()
        p_index = p_pos.argsort(axis=0)[::-1]
        p = p_pos[p_index].squeeze(-1)
        rm = anchors_center_to_corner(rm_pos[p_index].squeeze(1))
        #rm = corner_to_standup_box2d_batch(rm)
        rm_bev = bbox3d_2_birdeye(rm)
        #print(np.shape(rm_bev),np.shape(p))
        
        bboxes_bev = np.concatenate((rm_bev,p),axis=1)
        
        bboxes_final = bboxes_bev[nms(bboxes_bev,0.4)]
        print(np.shape(bboxes_final))
        print(bboxes_final)
        print(gt_box3d)
        gt_box3d = bbox3d_2_birdeye(gt_box3d)
        print(gt_box3d)
       
        
        file_name = ids[0].split('/')[-1].split('.')[0]
        log_file = open("/home/screentest/ys3237/VoxelNet-pytorch/predicts/"+setting+'_'+file_name+'.txt','w+')
        for i in bboxes_final:
            items = list(map(lambda x:str(x),list(i)))
            log_file.write(','.join(items)+'\n')
        log_file.close()
         '''
    
if __name__ == '__main__':
    
      inference()
    
    
    
