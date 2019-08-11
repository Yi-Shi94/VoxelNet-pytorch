from __future__ import division
from __future__ import print_function
from __future__ import with_statement
from __future__ import absolute_import

import os
import sys
import yaml
import tqdm
import numpy as np
from glob import glob

import cv2

from utils.utils import box3d_corner_to_center_batch, anchors_center_to_corner, corner_to_standup_box2d_batch
from data.data import KITDataset 
from box_overlaps import bbox_overlaps
from data_aug import aug_data

from VoxelNet import VoxelNet
from VoxelLoss import VoxelLoss

import torch
import torch.utils.data as data
import torch.backends.cudnn
import torch.optim as optim
import torch.nn.init as init

yamlPath = "configure.yaml"
f = open(yamlPath, 'r', encoding='utf-8')
conf = f.read()
conf_dict = yaml.safe_load(conf) 
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()
        
            
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

if_continued = True if conf_dict['if_continued'] == "1" else False
if_cuda = True if conf_dict["if_cuda"] == "1" else False
batch_size = conf_dict['batch_size']
learning_rate = conf_dict["lr"]
a = conf_dict["alpha"]
b = conf_dict["beta"]
epoch_num = conf_dict["epoch"]
chk_pth = conf_dict["chk_pth"]
print("batch_size:{}, if_continued:{}, if_cuda: {} , epoch_num:{}, learning_rate:{}, loss_param_alpha:{}, loss_param_beta:{},".format(batch_size, if_continued, if_cuda, epoch_num, learning_rate, a, b))
print(type(chk_pth),type(epoch_num))
if if_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("----------------------------------------\n")
kit_dataset= KITDataset(conf_dict=conf_dict)
data_loader = data.DataLoader(kit_dataset, batch_size=batch_size, num_workers=4, \
                              collate_fn=detection_collate, shuffle=True, \
                              pin_memory=False)

# network
net = VoxelNet()
if if_cuda:
    net.cuda()
    
def train():
    log_file = open('./log.txt','w')
    net.train()
   
    if if_continued:
        print('Loading pre-trained weights...')
        chk = glob(chk_pth+'/*')[-1]
        net.load_state_dict(torch.load(chk))
        net.eval()
    else:       
        # initialization
        print('Initializing weights...')
        net.apply(weights_init)
    
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    criterion = VoxelLoss(alpha=a, beta=b)
    batch_per_epoch = len(data_loader)//batch_size
    # training process
    for epoch in range(epoch_num):
        scheduler.step()
        for batch_index,contents in tqdm(enumerate(data_loader)):
            voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, images, calibs, ids = contents
            # wrapper to variable
            voxel_features = Variable(torch.cuda.FloatTensor(voxel_features))
            pos_equal_one = Variable(torch.cuda.FloatTensor(pos_equal_one))
            neg_equal_one = Variable(torch.cuda.FloatTensor(neg_equal_one))
            targets = Variable(torch.cuda.FloatTensor(targets))

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            t0 = time.time()
            psm, rm = net(voxel_features, voxel_coords)
            # calculate loss
            conf_loss, reg_loss = criterion(rm, psm, pos_equal_one, neg_equal_one, targets)
            loss = conf_loss + reg_loss
            loss.backward()
            optimizer.step()
            if epoch % 4 ==0:
                 torch.save(model.state_dict(), chk_pth+'/chk_'+str(epoch)+'.pth')
            if batch_index % 10 == 0:
                res = ('Epoch %d, batch: %d / %d, Timer Taken: %.4f sec.\n' % \
                  (epoch,batch_index,batch_per_epoch,(time.time() - t0)))
                res += 'Total Loss: %.4f || Conf Loss: %.4f || Loc Loss: %.4f\n' % \
                  (loss.data[0], conf_loss.data[0], reg_loss.data[0])
                print(res)
                log_file.write(res)
    
if __name__ == '__main__':
    train()
      