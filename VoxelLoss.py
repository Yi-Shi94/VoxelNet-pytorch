import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
import numpy as np

class VoxelLoss(nn.Module):
    def __init__(self, alpha, beta, reg):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(size_average=False, reduction=True)
        self.alpha = alpha
        self.beta = beta
        self.reg = reg
        self.small = 1e-6
    
    def SmoothL1Loss_custom(input, target,sigma = 3.0,size_average=None, reduce=True, reduction='mean'):
        # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
        def _smooth_l1_loss(input, target,sigma = 3.0):
        # type: (Tensor, Tensor) -> Tensor
            t = torch.abs(input - target)
            return torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
    
        if not (target.size() == input.size()):
            warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
        if size_average is not None or reduce is not None:
            reduction = _Reduction.legacy_get_string(size_average, reduce)
        if target.requires_grad:
            ret = _smooth_l1_loss(input, target)
            if reduction != 'none':
                ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        else:
            expanded_input, expanded_target = torch.broadcast_tensors(input, target)
            ret = torch._C._nn.smooth_l1_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
        return ret
    

    def forward(self, rm, psm, pos_equal_one, neg_equal_one, targets):
        #regression map,possibility score map: h/2,w/2,14,
        p_pos = F.sigmoid(psm.permute(0,2,3,1))
        d_peo = pos_equal_one.dim()
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(d_peo).expand(-1,-1,-1,-1,7)
        rm = rm.permute(0,2,3,1).contiguous()
        rm = rm.view(rm.size(0),rm.size(1),rm.size(2),-1,7)
        targets = targets.view(targets.size(0),targets.size(1),targets.size(2),-1,7)
        
        rm_pos = rm * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg
        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = self.reg * reg_loss / (pos_equal_one.sum() + self.small)
        
        #print("p_pos",p_pos.shape)
        cls_pos_loss = (-pos_equal_one * torch.log(p_pos + self.small)).sum()/(pos_equal_one.sum()+self.small)
        cls_neg_loss = (-neg_equal_one * torch.log(1 - p_pos + self.small)).sum()/(neg_equal_one.sum()+self.small)
        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss
        
        return conf_loss, reg_loss
    

