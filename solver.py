import torch
from torch.nn import functional as F
from teacher_rgb.DCF_ResNet_models import DCF_ResNet_rgb_t
from teacher_depth.DCF_ResNet_models import DCF_ResNet_depth_t
from student_rgb.DCF_ResNet_models import DCF_ResNet_rgb_s
from student_depth.DCF_ResNet_models import DCF_ResNet_depth_s
from distiller import build_model_kd
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
import torch.nn as nn
from tqdm import trange, tqdm

class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
      
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.net_rgb_t = DCF_ResNet_rgb_t()
        self.net_depth_t = DCF_ResNet_depth_t()
        self.net_rgb_s = DCF_ResNet_rgb_s()
        self.net_depth_s = DCF_ResNet_depth_s()
        self.net_kd=build_model_kd(self.net_rgb_t,self.net_rgb_s,self.net_depth_t,self.net_depth_s)

        print('Loading pre-trained teacher models for kd from %s and %s...' % (self.config.model_rgb_t,self.config.model_depth_t))
        self.net_rgb_t.load_state_dict(torch.load(self.config.model_rgb_t))
        self.net_depth_t.load_state_dict(torch.load(self.config.model_depth_t))
        if config.mode == 'test':
            print('Loading pre-trained model for testing from %s...' % self.config.model)
            self.net_kd.load_state_dict(torch.load(self.config.model, map_location=torch.device('cpu')))
        if config.mode == 'train':
            if self.config.load_rgb != '':
                print('Loading pretrained model to resume training from %s and %s...' % (self.config.load_rgb,self.config.load_depth))
                self.net_rgb_s.load_state_dict(torch.load(self.config.load_rgb))  # load pretrained model
                self.net_depth_s.load_state_dict(torch.load(self.config.load_depth))  # load pretrained model
        
        if self.config.cuda:
            self.net_rgb_t = self.net_rgb_t.cuda()
            self.net_depth_t = self.net_depth_t.cuda()
            self.net_rgb_s = self.net_rgb_s.cuda()
            self.net_depth_s = self.net_depth_s.cuda()
            self.net_kd = self.net_kd.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        
        self.optimizer = torch.optim.Adam(self.net_kd.parameters() , lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net_kd, 'Incomplete modality RGBD SOD Structure')

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params_t = 0
        num_params=0
        for p in model.parameters():
            if p.requires_grad:
                num_params_t += p.numel()
            else:
                num_params += p.numel()
        print(name)
        #print(model)
        print("The number of trainable parameters: {}".format(num_params_t))
        print("The number of parameters: {}".format(num_params))


    def test(self):
        print('Testing...')
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)

                #input = torch.cat((images, depth), dim=0)
                preds,r,d = self.net_kd(images,depth,depth)
                #print(preds.shape)
                #preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()

                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + '_rgbonly.png')
                cv2.imwrite(filename, multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')
    
  
    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        
        loss_vals=  []

        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss_item=0
            for i, data_batch in tqdm(enumerate(self.train_loader)):
                sal_image, sal_depth,sal_label= data_batch['sal_image'], data_batch['sal_depth'], data_batch['sal_label']

             
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label= sal_image.to(device),sal_depth.to(device),sal_label.to(device)
              
                self.optimizer.zero_grad()
               
                sal_out,dist_rgb,dist_depth = self.net_kd(sal_image,sal_depth,sal_label)
                

                sal_loss_final =  F.binary_cross_entropy_with_logits(sal_out, sal_label, reduction='sum')


                sal_rgb_only_loss = sal_loss_final + dist_rgb + dist_depth
                r_sal_loss += sal_rgb_only_loss.data
                r_sal_loss_item+=sal_rgb_only_loss.item() * sal_image.size(0)
                sal_rgb_only_loss.backward()
                self.optimizer.step()

 
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_kd.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
            train_loss=r_sal_loss_item/len(self.train_loader.dataset)
            writer.add_scalar('training loss', train_loss,epoch)
            loss_vals.append(train_loss)
            
            print('Epoch:[%2d/%2d] | Train Loss : %.3f | Learning rate : %0.7f' % (epoch, self.config.epoch,train_loss,self.optimizer.param_groups[0]['lr']))
   
            
        # save model
        torch.save(self.net_kd.state_dict(), '%s/final.pth' % self.config.save_folder)
        
