import numpy as np
import cv2
import yaml
import math

yamlPath = "configure.yaml"
f = open(yamlPath, 'r', encoding='utf-8')
conf = f.read()
conf_dict = yaml.safe_load(conf) 

range_x=conf_dict['range_x']
range_y=conf_dict['range_y']
range_z=conf_dict['range_z']
voxel_depth = conf_dict['vox_d']
voxel_width = conf_dict['vox_w']
voxel_height = conf_dict['vox_h']

H = math.ceil((max(range_x)-min(range_x))/voxel_height)
W = math.ceil((max(range_y)-min(range_y))/voxel_width)
D = math.ceil((max(range_z)-min(range_z))/voxel_depth)
print(H,W,D)
def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):
    # Input:
    #   points: (N, 3)
    #   rx/y/z: in radians
    # Output:
    #   points: (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))])
    mat1 = np.eye(4)
    mat1[3, 0:3] = tx, ty, tz
    points = np.matmul(points, mat1)
    if rx != 0:
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        mat[3, 3] = 1
        mat[1, 1] = np.cos(rx)
        mat[1, 2] = -np.sin(rx)
        mat[2, 1] = np.sin(rx)
        mat[2, 2] = np.cos(rx)
        points = np.matmul(points, mat)
    if ry != 0:
        mat = np.zeros((4, 4))
        mat[1, 1] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(ry)
        mat[0, 2] = np.sin(ry)
        mat[2, 0] = -np.sin(ry)
        mat[2, 2] = np.cos(ry)
        points = np.matmul(points, mat)
    if rz != 0:
        mat = np.zeros((4, 4))
        mat[2, 2] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(rz)
        mat[0, 1] = -np.sin(rz)
        mat[1, 0] = np.sin(rz)
        mat[1, 1] = np.cos(rz)
        points = np.matmul(points, mat)
    return points[:, 0:3]

def box_transform(boxes_corner, tx, ty, tz, r=0):
    # boxes_corner (N, 8, 3)
    for idx in range(len(boxes_corner)):
        boxes_corner[idx] = point_transform(boxes_corner[idx], tx, ty, tz, rz=r)
    return boxes_corner

def cal_iou2d(box1_corner, box2_corner):
    box1_corner = np.reshape(box1_corner, [4, 2])
    box2_corner = np.reshape(box2_corner, [4, 2])
    box1_corner = ((W, H)-(box1_corner - (range_x[0], range_y[0])) / (voxel_width, voxel_height)).astype(np.int32)
    box2_corner = ((W, H)-(box2_corner - (range_x[0], range_y[0])) / (voxel_width, voxel_height)).astype(np.int32)

    buf1 = np.zeros((H, W, 3))
    buf2 = np.zeros((H, W, 3))
    buf1 = cv2.fillConvexPoly(buf1, box1_corner, color=(1,1,1))[..., 0]
    buf2 = cv2.fillConvexPoly(buf2, box2_corner, color=(1,1,1))[..., 0]

    indiv = np.sum(np.absolute(buf1-buf2))
    share = np.sum((buf1 + buf2) == 2)
    if indiv == 0:
        return 0.0 # when target is out of bound
    return share / (indiv + share)

def aug_data(lidar, gt_box3d_corner):
    np.random.seed()
    choice = np.random.randint(1, 10)
    if choice >= 7:
        for idx in range(len(gt_box3d_corner)):
            # TODO: precisely gather the point
            is_collision = True
            _count = 0
            while is_collision and _count < 100:
                t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
                t_x = np.random.normal()
                t_y = np.random.normal()
                t_z = np.random.normal()

                # check collision
                tmp = box_transform(
                    gt_box3d_corner[[idx]], t_x, t_y, t_z, t_rz)
                is_collision = False
                for idy in range(idx):
                    iou = cal_iou2d(tmp[0,:4,:2],gt_box3d_corner[idy,:4,:2])
                    if iou > 0:
                        is_collision = True
                        _count += 1
                        break
            if not is_collision:
                box_corner = gt_box3d_corner[idx]
                minx = np.min(box_corner[:, 0])
                miny = np.min(box_corner[:, 1])
                minz = np.min(box_corner[:, 2])
                maxx = np.max(box_corner[:, 0])
                maxy = np.max(box_corner[:, 1])
                maxz = np.max(box_corner[:, 2])
                bound_x = np.logical_and(
                    lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
                bound_y = np.logical_and(
                    lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
                bound_z = np.logical_and(
                    lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
                bound_box = np.logical_and(
                    np.logical_and(bound_x, bound_y), bound_z)
                lidar[bound_box, 0:3] = point_transform(
                    lidar[bound_box, 0:3], t_x, t_y, t_z, rz=t_rz)
                gt_box3d_corner[idx] = box_transform(
                    gt_box3d_corner[[idx]], t_x, t_y, t_z, t_rz)

        gt_box3d = gt_box3d_corner

    elif choice < 7 and choice >= 4:
        # global rotation
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
        gt_box3d = box_transform(gt_box3d_corner, 0, 0, 0, r=angle)

    else:
        # global scaling
        factor = np.random.uniform(0.95, 1.05)
        lidar[:, 0:3] = lidar[:, 0:3] * factor
        gt_box3d = gt_box3d_corner * factor

    return lidar, gt_box3d
