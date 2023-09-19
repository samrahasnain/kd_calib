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

class ShuffleChannelAttention(nn.Module):
    def __init__(self, channel=64,reduction=16,kernel_size=3,groups=8):
        super(ShuffleChannelAttention, self).__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.g=groups
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,3,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        
    
    def forward(self, x) :
        b,c,h,w=x.shape
        max_result=self.maxpool(x)
        print('CSA max',max_result.shape)
        shuffled_in=max_result.view(b,self.g,c//self.g,1,1).permute(0,2,1,3,4).reshape(b,c,1,1)
        print('CSA shuffled_in',shuffled_in.shape)
        max_out=self.se(shuffled_in)
        print('CSA max_out',max_out.shape)
        output=self.sigmoid(max_out)
        print('CSA',output.shape)
        output=output.view(b,c,1,1)
        print('CSA',output.shape)
        return output
        
def adapter(xt3,xt4,xt5,yt3,yt4,yt5,s3,s4,s5):
    m = nn.Softmax(dim=1)
    print('distillation',xt3.shape,xt4.shape,xt5.shape,yt3.shape,yt4.shape,yt5.shape,s3.shape,s4.shape,s5.shape)
    s3 = ShuffleChannelAttention(s3)
    f3 = (xt3 * s3) + (yt3 * s3)
    f3_ = m(f3)
    
    s4 = ShuffleChannelAttention(s4)
    f4 = (xt4 * s4) + (yt4 * s4)
    f4_ = m(f4)

    s5 = ShuffleChannelAttention(s5)
    f5 = (xt5 * s5) + (yt5 * s5)
    f5_ = m(f5)

    final = f3_ + f4_ + f5_
    return final
    

    
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

        rgb_final = adapter(xt3,xt4,xt5,yt3,yt4,yt5,xs3,xs4,xs5)
        depth_final = adapter(xt3,xt4,xt5,yt3,yt4,yt5,ys3,ys4,ys5)
        det_corr_depth_t = cal_score(det_depth_t, gt)
        loss_distill_depth = distillation_loss(det_depth_s, det_corr_depth_t)
        loss_distill_rgb = distillation_loss(det_rgb_s, det_rgb_t.detach())
        final = (rgb_final + depth_final) + (rgb_final * depth_final)
        return final, loss_distill_rgb , loss_distill_depth
