from __future__ import division
from __future__ import print_function
from __future__ import with_statement
from __future__ import absolute_import

import os
from glob import glob
import torch
import yaml
import utils
import tdqm
from utils.mAp import *

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


yamlPath = "configure.yaml"
f = open(yamlPath, 'r', encoding='utf-8')

conf = f.read()
conf_dict = yaml.safe_load(conf) 
chk_pth = conf_dict['chk_pth']

if conf_dict['cuda']==1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
kit_dataset= KITDataset(conf_dict=conf_dict,setting="val",data_type="velodyne_test")
data_loader = data.DataLoader(kit_dataset, batch_size=4, num_workers=4, \
                              collate_fn=detection_collate, shuffle=False, \
                              pin_memory=True)
net = VoxelNet()
if if_cuda:
    net.cuda()
    
def test(mode="testing"):
    if mode=="testing":
        list_path = os.path.join('./data/val.txt')
    else:
        list_path = os.path.join('./data/train.txt')
    lidar_train_path = os.path.join('./data/'+mode, "crop/")
    image_train_path = os.path.join('./data/'+mode, "image_2/")
    calib_train_path = os.path.join('./data/'+mode, "calib/")
    label_train_path = os.path.join('./data/'+mode, "label_2/")
    
    f = open(list_path,'r')
    list_files = f.readlines()
    files=[i.strip().split('/')[-1].split('.')[0] for i in list_files]
    net.load_state_dict(torch.load(sorted(glob(chk_pth))[-1]))
    net.eval()
    
    total_loss = 0
    total_conf_loss = 0
    total_reg_loss = 0
    
    for item in tqdm(data_loader):
        voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, images, calibs, ids = item
        # wrapper to variable
        voxel_features = Variable(torch.cuda.FloatTensor(voxel_features))
        pos_equal_one = Variable(torch.cuda.FloatTensor(pos_equal_one))
        neg_equal_one = Variable(torch.cuda.FloatTensor(neg_equal_one))
        targets = Variable(torch.cuda.FloatTensor(targets))
        # filtering
        psm, rm = net(voxel_features, voxel_coords)
        conf_loss, reg_loss = criterion(rm, psm, pos_equal_one, neg_equal_one, targets)
        loss = conf_loss + reg_loss
        total_loss += loss
        total_conf_loss += conf_loss
        total_reg_loss += reg_loss
        
    print("=========================================================\n")
    res = 'Total Loss: %.4f || Conf Loss: %.4f || Loc Loss: %.4f' %(loss.data[0], conf_loss.data[0], reg_loss.data[0])
    print(res)
    
if __name__ == '__main__':
    test()
