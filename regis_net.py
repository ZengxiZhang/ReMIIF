import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchgeometry as tgm
import scipy.io as io
from utils.utils import coords_grid, upflow8, print0
from utils_ import *
from decoder import GMA_update
import warnings
from transfer import *
from ATT.attention_layer import Correlation, DAM
import time
import kornia
from torch.nn.functional import interpolate
from layers import  predict_flow, conv2D
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass



class Get_Flow(nn.Module):
    def __init__(self, sz):
        super().__init__()
        self.sz = sz

    def forward(self, four_point, a):
        four_point = four_point/ torch.Tensor([a]).cuda()

        four_point_org = torch.zeros((2, 2, 2)).cuda()

        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3]-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2]-1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3]-1, self.sz[2]-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)

        # four_point_new = four_point_org + four_point
        four_point_new = four_point_org + four_point

        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        # H = tgm.get_perspective_transform(four_point_org, four_point_new)
        H = tgm.get_perspective_transform(four_point_new, four_point_org)
        gridy, gridx = torch.meshgrid(torch.linspace(0, self.sz[3]-1, steps=self.sz[3]), torch.linspace(0, self.sz[2]-1, steps=self.sz[2]))
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, self.sz[3] * self.sz[2]))),
                           dim=0).unsqueeze(0).repeat(self.sz[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                          points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        return flow


class Initialize_Flow(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, b):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//b, W//b).cuda()
        coords1 = coords_grid(N, H//b, W//b).cuda()

        return coords0, coords1
    

class Conv1(nn.Module):
    def __init__(self, input_dim = 145):
        super(Conv1, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(input_dim, 128, 1, padding=0, stride=1), nn.ReLU(), 
        )

    def forward(self, x):
        x = self.layer0(x)
        return x


class Conv3(nn.Module):
    def __init__(self, input_dim = 130):
        super(Conv3, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(input_dim, 128, 3, padding=1, stride=1), nn.ReLU(), 
        )

    def forward(self, x):
        x = self.layer0(x)
        return x
        

class ReMIF(nn.Module):
    # def __init__(self, args):
    def __init__(self,args):
        super().__init__()

        self.conv3 = Conv3(input_dim=130)
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1, stride=1), nn.ReLU())
        self.dc_conv1_0 = conv2D(2, 48, kernel_size=3, stride=1, padding=1, dilation=1) # [48, 48, 32]
        self.dc_conv1_1 = conv2D(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv1_2 = conv2D(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.predict_flow1 = predict_flow(32, 2)

        self.transformer_0 = DAM(128, 1, 128, 128)
        self.kernel_list_0 = [0, 9, 5, 3, 3, 3] #此处0表示GM全局
        self.pad_list_0    = [0, 4, 2, 1, 1, 1]
        self.transformer_1 = DAM(96, 1, 96, 96)
        self.kernel_list_1 = [5, 5, 3, 3, 3, 3]
        self.pad_list_1    = [2, 2, 1, 1, 1, 1]
        self.kernel_0 = 17
        self.pad_0 = 8
        self.kernel_1 = 9
        self.pad_1 = 4
        self.conv1_0 = Conv1(input_dim = 145)
        self.conv1_1 = Conv1(input_dim=81)

        self.initialize_flow_4 = Initialize_Flow()
        self.update_block_4 = GMA_update(32)
        self.initialize_flow_2 = Initialize_Flow()
        # self.update_block_2 = GMA_update(16)
        self.update_block_2 = GMA_update(64)
        self.ShareFeature = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(4),nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(8),nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(1),nn.ReLU(inplace=True),
        )

        self.bridge = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=4, padding=0)
        self.genMask_decoder = Mask_Decoder(output_dim=1)
        self.args = args


        self.predict_flow = predict_flow(64, 32, 2)
  
    def test(self, ir1, vis2, ir_orisize, vis_orisize, vis_color, ir_encoder,vis_encoder,iters_lev0 = 4, iters_lev1= 2):
        _, _, h_ori, w_ori = ir_orisize.shape
        image1 = ir1
        image2 = vis2
        
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        image2_org = image2

        img1_en1, img1_en2 = ir_encoder(image1)
        img2_en1, img2_en2 = vis_encoder(image2)

        mask_I1_full = self.genMask_decoder(img1_en2)#  ir1
        mask_I2_full = self.genMask_decoder(img2_en2)# vis2
        mask_I1_full_32 = interpolate(mask_I1_full, size=(32, 32), 
                                      mode='bilinear', align_corners=False)
        mask_I2_full_32 = interpolate(mask_I2_full, size=(32, 32), 
                                      mode='bilinear', align_corners=False)


        fmap1_32 = img1_en2 * mask_I1_full_32
        fmap1_64 = img1_en1
        fmap2_32 = img2_en2 * mask_I2_full_32

        four_point_disp = torch.zeros((image1.shape[0], 2, 2, 2)).cuda()
        four_point_predictions = []
    
        flow_field_pred_list = torch.zeros((image1.shape[0], 2, 128, 128)).cuda()
        if iters_lev0>0:
            coords0, coords1 = self.initialize_flow_4(image1, 4)
            coords0 = coords0.detach()
            sz = fmap1_32.shape
            self.sz = sz
            self.get_flow_now_4 = Get_Flow(sz)
            for itr in range(iters_lev0):
                fmap1, fmap2 = self.transformer_0(fmap1_32, fmap2_32, self.kernel_list_0[itr], self.pad_list_0[itr])
            
                corr = F.relu(Correlation.apply(fmap1.contiguous(), fmap2.contiguous(), self.kernel_0, self.pad_0)) 
                b, h, w, _ = corr.shape
                corr_1 = F.avg_pool2d(corr.view(b, h*w, self.kernel_0, self.kernel_0), 2).view(b, h, w, 64).permute(0, 3, 1, 2)
                corr_2 = corr.view(b, h*w, self.kernel_0, self.kernel_0)
                corr_2 = corr_2[:,:,4:13,4:13].contiguous().view(b, h, w, 81).permute(0, 3, 1, 2)                                                  
                corr = torch.cat([corr_1, corr_2], dim=1)

                corr = self.conv1_0(corr)                                         
                flow = coords1 - coords0
                corr_flow = torch.cat((corr, flow), dim=1)
                corr_flow = self.conv3(corr_flow)             
                
                delta_four_point = self.update_block_4(corr_flow)
                four_point_disp =  four_point_disp + delta_four_point
                four_point_predictions.append(four_point_disp)
                coords1 = self.get_flow_now_4(four_point_disp, 4) # 除以4，为了适配32大小的图像
                
                if itr < (iters_lev0-1):
                    flow_med = coords1 - coords0
                    flow_med = F.upsample_bilinear(flow_med, None, [4, 4]) * 4 # 32 -> 128         
                    image2_warp = warp(image2_org, flow_med)
                   
                    _, fmap2_32_warp = vis_encoder(image2_warp)
                    warp_mask_I2_full = warp(mask_I2_full, flow_med)
                    warp_mask_I2_full_32 = interpolate(warp_mask_I2_full, size=(32, 32), mode='bilinear', align_corners=False)
                    fmap2_32_warp = fmap2_32_warp * warp_mask_I2_full_32
                    fmap2_32 = fmap2_32_warp.float()    
        
        if iters_lev1 > 0:
            flow_med = coords1 - coords0
            flow_med = F.upsample_bilinear(flow_med, None, [4, 4]) * 4   # 32 -> 128          
            flow_field_pred_list = torch.cat((flow_field_pred_list, flow_med), 1)
            image2_warp_homo = warp(image2_org, flow_med)

            fmap2_64_warp_homo, _ = vis_encoder(image2_warp_homo)
            fmap2_64 = fmap2_64_warp_homo.float()

            sz = fmap1_64.shape
            self.sz = sz
            self.get_flow_now_2 = Get_Flow(sz)
            
            coords0, coords1 = self.initialize_flow_2(image1, 2)
            coords0 = coords0.detach()
            coords1 = self.get_flow_now_2(four_point_disp, 2)

            flow_field_pred_final = flow_med
            for itr in range(iters_lev1):
                fmap1, fmap2 = self.transformer_1(fmap1_64, fmap2_64, self.kernel_list_1[itr], self.pad_list_1[itr])
                
                corr = F.relu(Correlation.apply(fmap1.contiguous(), fmap2.contiguous(), self.kernel_1, self.pad_1)).permute(0, 3, 1, 2)    
                b, _, h, w = corr.shape

                corr = self.conv1_1(corr)   
                corr_flow = corr
                corr = corr_flow
                corr_flow = self.conv4(corr_flow)  # 8,130,64,64
                corr_flow_disp = self.predict_flow(corr_flow) # 8,2,64,64

                x = self.dc_conv1_2(self.dc_conv1_1(self.dc_conv1_0(corr_flow_disp)))
                corr_flow_disp = self.predict_flow1(x) + corr_flow_disp # flow refine  #for 64 size

                if itr < (iters_lev1 - 1):
                    corr_flow_disp_up = F.upsample_bilinear(corr_flow_disp, None, [2, 2]) * 2#.detach()
                    flow_field_pred_final = flow_field_pred_final + corr_flow_disp_up# * scale
            
                    image2_warp_field = warp(image2_org, flow_field_pred_final)
                    
                    fmap2_64_warp, _ = vis_encoder(image2_warp_field)
                    fmap2_64_warp = fmap2_64_warp
                    fmap2_64 = fmap2_64_warp.float()
                
                else:
                    corr_flow_disp_up = F.upsample_bilinear(corr_flow_disp, None, [2, 2]) * 2#.detach()
                    flow_field_pred_final = flow_field_pred_final + corr_flow_disp_up #* scale
                    
                    flow_field_pred_final = interpolate(flow_field_pred_final, size=(h_ori, w_ori), mode='bilinear', align_corners=False)
                    flow_field_pred_final[:,0]=flow_field_pred_final[:,0]*(w_ori/128.)
                    flow_field_pred_final[:,1]=flow_field_pred_final[:,1]*(h_ori/128.)
                    
                    image2_ori_warp_field = warp(vis_orisize, flow_field_pred_final)
                    image2_ori_color_warp_field = warp(vis_color, flow_field_pred_final)

                   
        return flow_field_pred_final, image2_ori_warp_field, image2_ori_color_warp_field