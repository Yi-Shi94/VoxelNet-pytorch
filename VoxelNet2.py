import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import yaml
import math

yamlPath = "configure.yaml"
f = open(yamlPath, 'r', encoding='utf-8')
conf = f.read()
conf_dict = yaml.safe_load(conf) 

if_cuda = True if conf_dict['if_cuda'] == 1 and torch.cuda.is_available() else False
batch_size = conf_dict['batch_size']
range_x=conf_dict['range_x']
range_y=conf_dict['range_y']
range_z=conf_dict['range_z']
vox_depth = conf_dict['vox_d']
vox_width = conf_dict['vox_w']
vox_height = conf_dict['vox_h']
anchor_per_pos = conf_dict['anchors_per_vox']
pt_thres_per_vox = conf_dict['pt_thres_per_vox']

W = math.ceil((max(range_x)-min(range_x))/vox_width)
H = math.ceil((max(range_y)-min(range_y))/vox_height)
D = math.ceil((max(range_z)-min(range_z))/vox_depth)
       
small_addon_for_BCE = 1e-6

class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VFELayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.units = int(out_channels / 2)
        self.dense = nn.Sequential(nn.Linear(self.in_channels, self.units), nn.ReLU())
        self.batch_norm = nn.BatchNorm1d(self.units)

    def forward(self, inputs, mask):
        # [ΣK, T, in_ch] -> [ΣK, T, units] -> [ΣK, units, T]
        tmp = self.dense(inputs).transpose(1, 2)
        # [ΣK, units, T] -> [ΣK, T, units]
        pointwise = self.batch_norm(tmp).transpose(1, 2)
        aggregated, _ = torch.max(pointwise, dim = 1, keepdim = True)
        repeated = aggregated.expand(-1, pt_thres_per_vox, -1)
        concatenated = torch.cat([pointwise, repeated], dim = 2)
        # [ΣK, T, 1] -> [ΣK, T, 2 * units]
        mask = mask.expand(-1, -1, 2 * self.units)
        concatenated = concatenated * mask.float()
        return concatenated

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.vfe1 = VFELayer(7, 32)
        self.vfe2 = VFELayer(32, 128)

    def forward(self, feature, coordinate):
        batch_size = len(feature)
        feature = torch.cat(feature, dim = 0)   
        coordinate = torch.cat(coordinate, dim = 0)     
        vmax, _ = torch.max(feature, dim = 2, keepdim = True)
        mask = (vmax != 0)  # [ΣK, T, 1]
        x = self.vfe1(feature, mask)
        x = self.vfe2(x, mask)
        voxelwise, _ = torch.max(x, dim = 1)
        outputs = torch.sparse.FloatTensor(coordinate.t(), voxelwise, torch.Size([batch_size, D, H, W, 128]))
        outputs = outputs.to_dense()
        return outputs

class ConvMD(nn.Module):
    def __init__(self, M, cin, cout, k, s, p, bn = True, activation = True):
        super(ConvMD, self).__init__()
        self.M = M  # Dimension of input
        self.cin = cin
        self.cout = cout
        self.k = k
        self.s = s
        self.p = p
        self.bn = bn
        self.activation = activation
        if self.M == 2:     # 2D input
            self.conv = nn.Conv2d(self.cin, self.cout, self.k, self.s, self.p)
            if self.bn:
                self.batch_norm = nn.BatchNorm2d(self.cout)
        elif self.M == 3:   # 3D input
            self.conv = nn.Conv3d(self.cin, self.cout, self.k, self.s, self.p)
            if self.bn:
                self.batch_norm = nn.BatchNorm3d(self.cout)
        else:
            raise Exception('No such mode!')

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.bn:
            out = self.batch_norm(out)
        if self.activation:
            return F.relu(out)
        else:
            return out

class Deconv2D(nn.Module):
    def __init__(self, cin, cout, k, s, p, bn = True):
        super(Deconv2D, self).__init__()
        self.cin = cin
        self.cout = cout
        self.k = k
        self.s = s
        self.p = p
        self.bn = bn
        self.deconv = nn.ConvTranspose2d(self.cin, self.cout, self.k, self.s, self.p)
        if self.bn:
            self.batch_norm = nn.BatchNorm2d(self.cout)

    def forward(self, inputs):
        out = self.deconv(inputs)
        if self.bn == True:
            out = self.batch_norm(out)
        return F.relu(out)


class MiddleAndRPN(nn.Module):
    def __init__(self, alpha = 1.5, beta = 1, sigma = 3, training = True, name = ''):
        super(MiddleAndRPN, self).__init__()

        self.middle_layer = nn.Sequential(ConvMD(3, 128, 64, 3, (2, 1, 1,), (1, 1, 1)),
                                          ConvMD(3, 64, 64, 3, (1, 1, 1), (0, 1, 1)),
                                          ConvMD(3, 64, 64, 3, (2, 1, 1), (1, 1, 1)))

        if 1:
            self.block1 = nn.Sequential(ConvMD(2, 128, 128, 3, (2, 2), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)))
        
        self.deconv1 = Deconv2D(128, 256, 3, (1, 1), (1, 1))

        self.block2 = nn.Sequential(ConvMD(2, 128, 128, 3, (2, 2), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)))

        self.deconv2 = Deconv2D(128, 256, 2, (2, 2), (0, 0))

        self.block3 = nn.Sequential(ConvMD(2, 128, 256, 3, (2, 2), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)))

        self.deconv3 = Deconv2D(256, 256, 4, (4, 4), (0, 0))

        self.prob_conv = ConvMD(2, 768, 2, 1, (1, 1), (0, 0), bn = False, activation = False)

        self.reg_conv = ConvMD(2, 768, 14, 1, (1, 1), (0, 0), bn = False, activation = False)

        self.output_shape = [H, W]


    def forward(self, inputs):

        batch_size, DEPTH, HEIGHT, WIDTH, C = inputs.shape  # [batch_size, 10, 400/200, 352/240, 128]

        inputs = inputs.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)

        temp_conv = self.middle_layer(inputs)   # [batch, 64, 2, 400, 352]
        temp_conv = temp_conv.view(batch_size, -1, HEIGHT, WIDTH)   # [batch, 128, 400, 352]

        temp_conv = self.block1(temp_conv)      # [batch, 128, 200, 176]
        temp_deconv1 = self.deconv1(temp_conv)

        temp_conv = self.block2(temp_conv)      # [batch, 128, 100, 88]
        temp_deconv2 = self.deconv2(temp_conv)

        temp_conv = self.block3(temp_conv)      # [batch, 256, 50, 44]
        temp_deconv3 = self.deconv3(temp_conv)
        temp_conv = torch.cat([temp_deconv3, temp_deconv2, temp_deconv1], dim = 1)
        # Probability score map, [batch, 2, 200/100, 176/120]
        p_map = self.prob_conv(temp_conv)
        # Regression map, [batch, 14, 200/100, 176/120]
        r_map = self.reg_conv(temp_conv)
        return torch.sigmoid(p_map), r_map
    
class VoxelNet(nn.Module):
    def __init__(self):
        super(VoxelNet, self).__init__()
        self.fn = FeatureNet()
        self.mr = MiddleAndRPN()

    def forward(self, features, coords):
        out = self.fn(features, coords)
        psm,rm = self.mr(out)
        return psm, rm

def smooth_l1(deltas, targets, sigma = 3.0):
    # Reference: https://mohitjainweb.files.wordpress.com/2018/03/smoothl1loss.pdf
    sigma2 = sigma * sigma
    diffs = deltas - targets
    smooth_l1_signs = torch.lt(torch.abs(diffs), 1.0 / sigma2).float()
    smooth_l1_option1 = torch.mul(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = torch.mul(smooth_l1_option1, smooth_l1_signs) + torch.mul(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add
    return smooth_l1