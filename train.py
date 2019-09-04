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
from data.data import KITDataset
import VoxelNet1
import VoxelNet
from VoxelLoss import VoxelLoss

from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.utils.data as data
import torch.backends.cudnn
import torch.optim as optim
import torch.nn.init as init
import warnings
warnings.filterwarnings("ignore")


        
yaml_path = "configure.yaml"
f = open(yaml_path, 'r', encoding='utf-8')
conf = f.read()
conf_dict = yaml.safe_load(conf) 

net_type = conf_dict['net_type'] 
chk_name = conf_dict['chk_name'] 
#chk_name_num = eval(chk_name.split('_')[-1].split('.')[0])
chk_save_interval = conf_dict['chk_save_interval']
if_continued = True if conf_dict['if_continued'] == 1 else False
if_cuda = True if conf_dict["if_cuda"] == 1 and torch.cuda.is_available() else False
batch_size = conf_dict['batch_size']
learning_rate = conf_dict["lr"]
a = conf_dict["alpha"]
b = conf_dict["beta"]
r = conf_dict["lambda"] #regression weight
clip_grad_thres = conf_dict["clip_grad_thres"]

lr_ds_interval_epoch = conf_dict["lr_ds_interval_epoch"]
lr_ds_coeff = conf_dict["lr_ds_coeff"]

classes = '_'.join(conf_dict["classes"])
epoch_num = conf_dict["epoch"]
chk_pth = conf_dict["chk_pth"]

def detection_collate(batch):
    voxel_features = []
    voxel_coords = []
    pos_map = []
    neg_map = []
    targets = []
    images = []
    calibs = []
    ids = []
    pad = 1
    for i, sample in enumerate(batch):
        
        if pad:
            sample_tmp_1 = np.pad(sample[1], ((0, 0), (1, 0)),
                mode='constant', constant_values=i)
        else:
            sample_tmp_1 = sample[1]
            
        voxel_features.append(sample[0])
        voxel_coords.append(sample_tmp_1)
        pos_map.append(sample[2])
        neg_map.append(sample[3])
        targets.append(sample[4])
        images.append(sample[5])
        calibs.append(sample[6])
        ids.append(sample[7])
    return np.concatenate(voxel_features), np.concatenate(voxel_coords), \
           np.array(pos_map),np.array(neg_map), np.array(targets), images, calibs, ids


#print(chk_name,chk_name_num)
print("batch_size:{}, if_continued:{}, if_cuda: {} , epoch_num:{}, learning_rate:{}, loss_param_alpha:{}, loss_param_beta:{}, classes: {}".format(batch_size, if_continued, if_cuda, epoch_num, learning_rate, a, b,classes))

if if_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
print("----------------------------------------")
kit_dataset= KITDataset(conf_dict=conf_dict, setting='training',root_path='/root/data/syd/voxelnet')
kit_data_loader = data.DataLoader(kit_dataset, batch_size=batch_size, num_workers=6, \
                              collate_fn=detection_collate, \
                              shuffle=True, \
                              pin_memory=False)

# network
if net_type==0:
    net = VoxelNet.VoxelNet()
else:
    net = VoxelNet1.VoxelNet()

if if_cuda:
    net.cuda()
net.train()

if if_continued:
    print('Loading pre-trained weights...')
    chk = os.path.join(chk_pth,chk_name)
    net.load_state_dict(torch.load(chk))
    net.eval()
else:       
    # initialization
    print('Initializing weights...')
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            m.bias.data.zero_()
    net.apply(weights_init)
    
    
def mytrain():
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_ds_interval_epoch, gamma=lr_ds_coeff)
    criterion = VoxelLoss(alpha=a, beta=b, reg = r )
    batch_per_epoch = len(kit_data_loader)
    
    # training process
    loss_epoch = .0
    for epoch in range(epoch_num):
        loss_epoch = .0
        #scheduler.step()
        #for batch_index, contents in enumerate(tqdm(kit_data_loader)):
        for batch_index, contents in enumerate(kit_data_loader):
            voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, images, calibs, ids = contents
         
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
            optimizer.zero_grad()
            # forward
            
            psm, rm = net(voxel_features, voxel_coords)
            #print (psm.shape,rm.shape)
            # calculate loss
            
            conf_loss, reg_loss = criterion(rm, psm, pos_equal_one, neg_equal_one, targets)
            total_loss = conf_loss + reg_loss
            total_loss.backward()
            clip_grad_norm_(net.parameters(), clip_grad_thres)

            loss_epoch += total_loss.item()  
            optimizer.step()
            
            if batch_index % 10  == 0 or batch_index == batch_per_epoch-1:
                if batch_index == 0:
                    t0 = time.time()
                res = ('Epoch %d, batch: %d / %d, Timer Taken: %.4f sec.\n' % \
                  (epoch,batch_index,batch_per_epoch,(time.time() - t0)))
                res += 'Total Loss: %.4f ===== Confidence Loss: %.4f ===== BBox Loss: %.4f\n' % \
                  (total_loss.item(), conf_loss.item(), reg_loss.item())
                print(res)
                t0 = time.time()
                #log_file.write(res)
        print("total loss in epoch %d: %f" % (epoch,loss_epoch/(batch_size*len(kit_data_loader))))
        
        print("Saving pth:", chk_pth+'/chk_CV_latest_final.pth')
        torch.save(net.state_dict(), chk_pth+'/chk_CV_latest_final.pth')
        
        if epoch % chk_save_interval ==0 and epoch != 0:
            if if_continued == False:
                chk_name_num = 0
            print("Saving pth:", chk_pth+'/chk_CV_'+str(epoch+29)+'.pth')
            torch.save(net.state_dict(), chk_pth+'/chk_CV_'+str(epoch+29)+'.pth')
    
    
if __name__ == '__main__':
    mytrain()
      
