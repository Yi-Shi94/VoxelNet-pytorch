from __future__ import division
from __future__ import print_function
from __future__ import with_statement
from __future__ import absolute_import

import os
from glob import glob
import torch
import yaml
import utils

yamlPath = "configure.yaml"
f = open(yamlPath, 'r', encoding='utf-8')

conf = f.read()
conf_dict = yaml.safe_load(conf) 
chk_pth = conf_dict['chk_pth']

if conf_dict['cuda']==1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
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
    
    for i in files:
        lidar_file = lidar_path + '/' + file[i] + '.bin'
        calib_file = calib_path + '/' + file[i] + '.txt'
        label_file = label_path + '/' + file[i] + '.txt'
        image_file = image_path + '/' + file[i] + '.png'
        
        print("Processing: ", lidar_file)
        lidar = np.fromfile(lidar_file, dtype=np.float32)
        lidar = lidar.reshape((-1, 4))
        
        calib = load_kitti_calib(calib_file)
        gt_box3d = load_kitti_label(label_file, calib['Tr_velo2cam'])
        # filtering
        lidar, gt_box3d = get_filtered_lidar(lidar, gt_box3d)
        
        
if __name__ == '__main__':
    test()
