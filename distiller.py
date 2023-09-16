import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy

import math
def cal_score(img, gt, beta=0.3):
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

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)
  
class build_model_kd(nn.Module):
    def __init__(self, t_net, s_net):
        super(build_model_kd, self).__init__()
        t_channels=[512,256,64,64,1]
        s_channels=[128,64,64,64,1]
        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])
        self.t_net = t_net
        self.s_net = s_net

    def forward(self, x,y):
        t1,t2,t3,t4,t5 = self.t_net(x)
        s1,s2,s3,s4,s5 = self.s_net(y)
        t_feats=[t5,t4,t3,t2,t1]
        s_feats=[s5,s4,s3,s2,s1]
        feat_num = len(t_feats)
        loss_distill = 0
        for i in range(feat_num):
            s_feats[i] = self.Connectors[i](s_feats[i])
            #print(t_feats[i].shape,s_feats[i].shape)
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach())

        return s1, loss_distill
