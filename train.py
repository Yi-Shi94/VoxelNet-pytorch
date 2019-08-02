from __future__ import division
from __future__ import print_function
from __future__ import with_statement
from __future__ import absolute_import

import os
import sys
import yaml
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cv2

#from utils import box3d_corner_to_center_batch, anchors_center_to_corner, corner_to_standup_box2d_batch
#from box_overlaps import bbox_overlaps
#from data_aug import aug_data

from utils.coord_transform import *
from utils.file_load import *
from data.data import KITDataset 
from VoxelNet import VoxelNet

import torch
import torch.utils.data as data
import torch.backends.cudnn
import torch.optim as optim
import torch.nn.init as init

yamlPath = "configure.yaml"
f = open(yamlPath, 'r', encoding='utf-8')
conf = f.read()
conf_dict = yaml.safe_load(conf) 

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
    return np.concatenate(voxel_features), \
           np.concatenate(voxel_coords), \
           np.array(pos_equal_one),\
           np.array(neg_equal_one),\
           np.array(targets),\
           images, calibs, ids



batch_size = conf_dict['batch_size']
cuda = True if conf_dict["cuda"] == "1" else False
leaning_rate = eval(conf_dict["lr"])
a = eval(conf_dict["alpha"])
b = eval(conf_dict["beta"])
epoch = eval(conf_dict["epoch"])


kit_dataset= KITDataset(conf_dict=conf_dict)
data_loader = data.DataLoader(kit_dataset, batch_size=batch_size, num_workers=4, \
                              collate_fn=detection_collate, shuffle=True, \
                              pin_memory=False)

# network
net = VoxelNet()
if cuda:
    net.cuda()

def train():
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform(m.weight.data)
            m.bias.data.zero_()
            
    net.train()
    # initialization
    print('Initializing weights...')
    net.apply(weights_init)
    # define optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    # define loss function
    criterion = VoxelLoss(alpha=a, beta=b)
    # training process
    batch_iterator = None
    epoch_size = len(dataset) // batch_size
    
    
    print('Epoch size', epoch_size)
    for iteration in range(10000):
            if (not batch_iterator) or (iteration % epoch_size == 0):
                # create batch iterator
                batch_iterator = iter(data_loader)

            voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, images, calibs, ids = next(batch_iterator)

            # wrapper to variable
            voxel_features = Variable(torch.cuda.FloatTensor(voxel_features))
            pos_equal_one = Variable(torch.cuda.FloatTensor(pos_equal_one))
            neg_equal_one = Variable(torch.cuda.FloatTensor(neg_equal_one))
            targets = Variable(torch.cuda.FloatTensor(targets))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            t0 = time.time()
            psm,rm = net(voxel_features, voxel_coords)

            # calculate loss
            conf_loss, reg_loss = criterion(rm, psm, pos_equal_one, neg_equal_one, targets)
            loss = conf_loss + reg_loss

            # backward
            loss.backward()
            optimizer.step()

            t1 = time.time()


            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f || Conf Loss: %.4f || Loc Loss: %.4f' % \
                  (loss.data[0], conf_loss.data[0], reg_loss.data[0]))

            # visualization
            #draw_boxes(rm, psm, ids, images, calibs, 'pred')
            draw_boxes(targets.data, pos_equal_one.data, images, calibs, ids,'true')



if __name__ == '__main__':
    train()
      
        



