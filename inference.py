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

import cv2
from utils.utils import box3d_corner_to_center_batch, anchors_center_to_corner, corner_to_standup_box2d_batch
from data.data import KITDataset 
from box_overlaps import bbox_overlaps
from data_aug import aug_data

from VoxelNet import VoxelNet,weights_init
from VoxelLoss import VoxelLoss

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

if_cuda = True if conf_dict["if_cuda"] == 1 else False
learning_rate = conf_dict["lr"]
a = conf_dict["alpha"]
b = conf_dict["beta"]
classes = '_'.join(conf["classes"])
epoch_num = conf_dict["epoch"]
chk_pth = conf_dict["chk_pth_infer"]

print("if_cuda: {}, learning_rate:{}, loss_param_alpha:{}, loss_param_beta:{}, classes:{}".format( if_cuda, learning_rate, a, b, classes))

if if_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def detection_collate(batch):
    voxel_features = []
    voxel_coords = []
    pos_equal_one = []
    neg_equal_one = []
    targets = []
    images = []
    calibs = []
    ids = []
    
    for i, sample in enumerate(batch):
        voxel_features.append(sample[0])
        voxel_coords.append(
            np.pad(sample[1], ((0, 0), (1, 0)),
                mode='constant', constant_values=i))

        pos_equal_one.append(sample[2])
        neg_equal_one.append(sample[3])
        targets.append(sample[4])

        images.append(sample[5])
        calibs.append(sample[6])
        ids.append(sample[7])
    return np.concatenate(voxel_features), np.concatenate(voxel_coords), \
           np.array(pos_equal_one),np.array(neg_equal_one),\
           np.array(targets), images, calibs, ids

print("------------------------------------------------------")
kit_dataset= KITDataset(conf_dict=conf_dict,setting="val")#test,val
kit_data_loader = data.DataLoader(kit_dataset, batch_size=1, num_workers=4, \
                              collate_fn=detection_collate, \
                              pin_memory=False)

net = VoxelNet()
if if_cuda:
    net.cuda()

print('Loading pre-trained weights...')
    #chk = glob(chk_pth+'/*')[-1]
net.load_state_dict(torch.load(chk_pth))
net.eval()
    
def inference(setting="val"):#test,val
    for batch_index, contents in enumerate(tqdm(kit_data_loader)):
        voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, images, calibs, ids = contents
            # wrapper to variable
        if if_cuda:
            voxel_features = Variable(torch.cuda.FloatTensor(voxel_features))
            pos_equal_one = Variable(torch.cuda.FloatTensor(pos_equal_one))
            neg_equal_one = Variable(torch.cuda.FloatTensor(neg_equal_one))
            targets = Variable(torch.cuda.FloatTensor(targets))
        else:
            voxel_features = Variable(torch.FloatTensor(voxel_features))
            pos_equal_one = Variable(torch.FloatTensor(pos_equal_one))
            neg_equal_one = Variable(torch.FloatTensor(neg_equal_one))
            targets = Variable(torch.FloatTensor(targets))

            # zero the parameter gradients
        psm, rm = net(voxel_features, voxel_coords)
        print(psm,rm)
            # calculate loss
            #conf_loss, reg_loss = criterion(rm, psm, pos_equal_one, neg_equal_one, targets)
            #loss = conf_loss + reg_loss
        #log_file = open("./predicts/"+setting+'/'+ids+'.txt')
        #log_file.close()
    
if __name__ == '__main__':
    mytrain()
      
    
    
    
