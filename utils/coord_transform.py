from __future__ import division
import numpy as np
import math
import yaml

yamlPath = "configure.yaml"
f = open(yamlPath, 'r', encoding='utf-8')
conf = f.read()
conf_dict = yaml.safe_load(conf) 
range_x=conf_dict['range_x']
range_y=conf_dict['range_y']
range_z=conf_dict['range_z']
vox_depth = conf_dict['vox_d']
vox_width = conf_dict['vox_w']
vox_height = conf_dict['vox_h']

def anchors_center_to_corner(anchors):
    N = anchors.shape[0]
    anchor_corner = np.zeros((N, 4, 2))
    for i in range(N):
        anchor = anchors[i]
        translation = anchor[0:3]
        h, w, l = anchor[3:6]
        rz = anchor[-1]
        Box = np.array([
            [-l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2]])
        # re-create 3D bounding box in velodyne coordinate system
        rotMat = np.array([
            [np.cos(rz), -np.sin(rz)],
            [np.sin(rz), np.cos(rz)]])
        velo_box = np.dot(rotMat, Box)
        cornerPosInVelo = velo_box + np.tile(translation[:2], (4, 1)).T
        anchor_corner[i] = cornerPosInVelo.transpose()
    return anchor_corner


def project_velo_to_cam(lidar, P_intr, Tr, R):
    coord_in_cam0 = np.dot(lidar,Tr)
    coord_in_cam2 = np.dot(coord_in_cam0,R)
    return np.dot(coord_in_cam2,P_intr)
    
def project_cam2velo(cam, T):
    T_inv = np.linalg.inv(T)
    lidar_loc_ = np.dot(T_inv,cam)
    lidar_loc = lidar_loc_[:3]
    return lidar_loc.reshape(1, 3)

def box3d_cam_to_velo(box3d, Tr):
    def ry_to_rz(ry):
        angle = -ry - np.pi / 2
        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2*np.pi + angle
        return angle
    
    h,w,l,x,y,z,ry = box3d
    cam = np.ones([4, 1])
    cam[0] = x
    cam[1] = y
    cam[2] = z
    t_lidar = project_cam2velo(cam, Tr)
    
    #in origin parallel to xy
    Box = np.array([[-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
                    [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
                    [0, 0, 0, 0, h, h, h, h]])

    rz = ry_to_rz(ry)
    # rotate matrix along z axis
    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])
    # perform rotation
    velo_box = np.dot(rotMat, Box)
    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T
    box3d_corner = cornerPosInVelo.transpose()
    return box3d_corner.astype(np.float32)

def point_cloud_2_birdseye(points,
                           res=0.1,
                           side_range=range_y,  
                           fwd_range =range_x, 
                           height_range=range_z, 
                           ):
  
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()
    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    s0 = np.ones([len(y_points),])*np.abs(side_range[0]-0)
    f0 = np.ones([len(x_points),])*np.abs(fwd_range[1]-fwd_range[0])
    x_img = ((-side_range[0]-y_points)/res).astype(np.int32) 
    y_img = ((fwd_range[1]-fwd_range[0]-x_points)/res).astype(np.int32) 
    
    
    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values
    return im

def bbox3d_2_birdeye(points,
                     res=0.1,
                     fwd_range =range_x, # back-most to forward-most
                     side_range=range_y,  # left-most to right-most
                     height_range=range_z):  # bottom-most to upper-most
    if len(points)==0:
        return np.array([])
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    x_img = ((-side_range[0]-y_points)/res).astype(np.int32) 
    y_img = ((fwd_range[1]-fwd_range[0]-x_points)/res).astype(np.int32) 
    #print(x_img,np.shape(x_img))
    #print(y_img,np.shape(y_img))
    
    x_min = x_img.min(axis=1)
    x_max = x_img.max(axis=1)
    y_min = y_img.min(axis=1)
    y_max = y_img.max(axis=1)
    return np.array(list(zip(x_min,y_min,x_max,y_max)))

