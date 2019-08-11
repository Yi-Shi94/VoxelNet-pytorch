import cv2
import os
from utils.nms import nms
import numpy as np
from utils.utils import box3d_cam_to_velo
def cvt(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def eval_lst(lst):
    return list(map(lambda x: eval(x),lst))

def read_img(file_path):
    return cvt(cv2.imread(file_path))

def read_velodyne_points(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points   

def prepare_velodyne_points(pts3d_raw,range_x = None,range_y = None, range_z = None):
    '''Replaces the reflectance value by 1, and tranposes the array, so
        points can be directly multiplied by the camera projection matrix'''
    pts3d = pts3d_raw
    indices = pts3d[:, 3] >= 0
    
    #print("range",range_x,range_y,range_z,len(indices),len(pts3d))
    if range_x!=None:
        indices_x = np.logical_and((pts3d[:,0]>=min(range_x)),(pts3d[:,0]<=max(range_x)))
        indices = np.logical_and(indices,indices_x)
    if range_y!=None:
        indices_y = np.logical_and((pts3d[:,1]>=min(range_y)),(pts3d[:,1]<=max(range_y)))
        indices = np.logical_and(indices,indices_y)
    if range_z!=None:
        indices_z = np.logical_and((pts3d[:,2]>=min(range_z)),(pts3d[:,2]<=max(range_z)))
        indices = np.logical_and(indices,indices_z)
    pts3d = pts3d[indices ,:]   
    #pts3d[:,3] = 1 
    return pts3d, indices

def read_cal(file_path):
    info_dict = {}
    with open(file_path,mode='r') as f:
        
        lines = f.readlines()[:-1]
        items = list(map(lambda x:(x.split()[0][:-1],eval_lst(x.split()[1:])),lines))
        for i in range(0,4):
            info_dict[items[i][0]]= np.array(items[i][1]).reshape(4,3).astype('float32')
        Tr_velo_to_cam = np.array(items[5][1]).reshape(3,4)
        Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([0,0,0,1]).reshape(1,4)],0).astype('float32')
        R_cam_to_rect = np.eye(4)
        R_cam_to_rect[:3,:3] = np.array(items[4][1]).reshape(3,3).astype('float32')
        info_dict['R_cam_to_rect'] = R_cam_to_rect
        info_dict['Tr_velo_to_cam'] = Tr_velo_to_cam
        
        return info_dict
    
def read_label_for_vis(file_path):
    with open(file_path,mode='r') as f:
        lines = f.readlines()
        lines = np.array(list(map(lambda x:x.split(),lines)))
        #print (bbox_2d)
        coords_2d = np.concatenate([lines[:,:1],lines[:,4:8],lines[:,-1:]],1)
        coords_keep_indices = nms(coords_2d[:,1:].astype('float32'),0.5)
        bbox_2d = coords_2d[coords_keep_indices,:]
        coords_3d = np.concatenate([lines[:,:1],lines[:,8:-1],lines[:,-1:]],1)
        bbox_3d = coords_3d[coords_keep_indices,:]
        return bbox_2d,bbox_3d

def read_label(file_path,Tr,classes):
    with open(file_path,'r') as f:
        lines = f.readlines()
        gt_boxes3d_corner = []

        num_obj = len(lines)

        for j in range(num_obj):
            obj = lines[j].strip().split(' ')
            obj_class = obj[0].strip()
            if obj_class not in classes:
                continue
            box3d_corner = box3d_cam_to_velo(obj[8:], Tr)

            gt_boxes3d_corner.append(box3d_corner)
        gt_boxes3d_corner = np.array(gt_boxes3d_corner).reshape(-1,8,3)
        return gt_boxes3d_corner
    
def generate_file_path(file_index,root_path,mode="training"):
    parent_pth = root_path
    file_name = "%06d"%(file_index)
    img = os.path.join(parent_pth, mode,'image_2',  file_name+'.png')
    vel = os.path.join(parent_pth, mode,'velodyne', file_name+'.bin')
    cal = os.path.join(parent_pth, mode,'calib',    file_name+'.txt')
    label = os.path.join(parent_pth,mode,'label_2', file_name+'.txt')
    return [img,vel,cal,label]