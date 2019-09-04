from __future__ import division
from __future__ import print_function
from __future__ import with_statement
from __future__ import absolute_import
import math
import os
from glob import glob
import torch
import yaml
import utils
import tqdm
from utils.plot import box3d_to_label

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
        
        psm = F.sigmoid(psm.permute(0,2,3,1))
        psm = psm.reshape((batch_size, -1))
        rm = rm.permute(0,2,3,1).contiguous()
        rm = rm.view(rm.size(0),rm.size(1),rm.size(2),14)#([batch, 200, 176, 2, 7])
        batch_boxes3d = delta_to_boxes3d(rm, anchors, coordinate='lidar')
        batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]

        psm = psm.detach().numpy()
        torch.cuda.empty_cache()
        
        ret_box3d = []
        ret_score = []
        
        for sid in range(batch_size):
            # remove box with low score
            ind = np.where(psm[sid, :] >= 0.97)[0]
            tmp_boxes3d = batch_boxes3d[sid, ind, ...]
            tmp_boxes2d = batch_boxes2d[sid, ind, ...]
            print(sid,len(ind),rm.shape)
            tmp_scores = psm[sid, ind]
            boxes_2d = corner_to_standup_box2d(center_to_corner_box2d(tmp_boxes2d, coordinate='lidar'))
            
            tmp_scores = tmp_scores
            print("2dd",np.shape(boxes_2d),np.shape(tmp_scores))
            boxes2d_cat = np.concatenate((boxes_2d,tmp_scores[...,np.newaxis]),axis=1)
            
            ind_nms = nms(boxes2d_cat,thresh=0.1)
            #ind_nms = ind
            print("2dd",np.shape(boxes2d_cat),np.shape(ind_nms))
            tmp_boxes2d = tmp_boxes2d[ind_nms, ...]
            tmp_boxes3d = tmp_boxes3d[ind_nms, ...]
            tmp_scores  = tmp_scores [ind_nms]
            ret_box3d.append(tmp_boxes3d)
            ret_score.append(tmp_scores)
            
        print(np.shape(ret_box3d[0]),np.shape(ret_score))
        
        ret_box3d_score = []
        num_classes = len(self.classes)
        
        for boxes3d, scores in zip(ret_box3d, ret_score):
            ret_box3d_score.append(np.concatenate([np.tile(num_classes, len(boxes3d))[:, np.newaxis],
                                                   boxes3d, scores[:, np.newaxis]], axis=-1))
        
        for result in ret_box3d_score:
            out_path = os.path.join('./prediction',   '.txt')                      
            labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
            for line in labels:
                f.write(line)
            print('write out {} objects to {}'.format(len(labels), tag))
                        
                        
if __name__ == '__main__':
    test()
