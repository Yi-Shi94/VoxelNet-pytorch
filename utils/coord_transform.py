import numpy as np

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
