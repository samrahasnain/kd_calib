import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy

import math
def cal_score(img, gt):
    img = np.float32(img)
    gt = np.float32(gt)
    gt *= 1/255.0
    img *= 1/255.0
    gt[gt >= 0.5] =1.
    gt[gt < 0.5] = 0.
    img[img >= 0.5] = 1.
    img[img < 0.5] = 0.

    over = (img*gt).sum()
    union = ((img+gt)>=1).sum()
    
    iou = over / (1e-7 + union);
    return iou
    
def distillation_loss(source, target):
    # Calculate L2 loss
    criterion = nn.MSELoss()
    loss = criterion(source, target)
    return loss.item()

  
class build_model_kd(nn.Module):
    def __init__(self, t_net_rgb, s_net_rgb , t_net_depth , s_net_depth):
        super(build_model_kd, self).__init__()
        self.t_net_rgb = t_net_rgb
        self.s_net_rgb = s_net_rgb
        self.t_net_depth = t_net_depth
        self.s_net_depth = s_net_depth

    def forward(self, x,y,gt):
        att_rgb_t,det_rgb_t,xt3,xt4,xt5 = self.t_net_rgb(x)
        att_rgb_s,det_rgb_s,xs3,xs4,xs5 = self.s_net_rgb(x)
        att_depth_t,det_depth_t,yt3,yt4,yt5 = self.t_net_depth(y)
        att_depth_s,det_depth_s,ys3,ys4,ys5 = self.s_net_depth(y)
        
        det_corr_depth_t = cal_score(det_depth_t, gt)
        loss_distill_depth = distillation_loss(det_depth_s, det_corr_depth_t)
        loss_distill_rgb = distillation_loss(det_rgb_s, det_rgb_t.detach())

        return final, loss_distill_rgb , loss_distill_depth
