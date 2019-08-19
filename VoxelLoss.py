import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(size_average=False)
        self.alpha = alpha
        self.beta = beta

    def forward(self, rm, psm, pos_equal_one, neg_equal_one, targets):
        #regression map,possibility score map: h/2,w/2,14,
        p_pos = F.sigmoid(psm.permute(0,2,3,1))
        print("psm",psm.shape)
        print("p_pos",p_pos.shape)
        cls_pos_loss = -pos_equal_one * torch.log(p_pos + 1e-6)
        print("cls_pos_loss_map",cls_pos_loss.shape)
        cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + 1e-6)
        print("cls_pos_loss",cls_pos_loss.data)
        cls_neg_loss = -neg_equal_one * torch.log(1 - p_pos + 1e-6)
        print("cls_neg_loss_map",cls_neg_loss.shape)
        cls_neg_loss = cls_neg_loss.sum() / (neg_equal_one.sum() + 1e-6)
        print("cls_neg_loss",cls_neg_loss.data)
        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss
        
        print("rm",rm.shape)
        rm = rm.permute(0,2,3,1).contiguous()
        rm = rm.view(rm.size(0),rm.size(1),rm.size(2),-1,7)
        targets = targets.view(targets.size(0),targets.size(1),targets.size(2),-1,7)
        print("peo",pos_equal_one.shape)
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(pos_equal_one.dim()).expand(-1,-1,-1,-1,7)
        print("pos_equal_one_for_reg",pos_equal_one_for_reg.shape)
        rm_pos = rm * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg
        print("targets_pos",targets_pos.shape)
        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = reg_loss / (pos_equal_one.sum() + 1e-6)
        return conf_loss, reg_loss

