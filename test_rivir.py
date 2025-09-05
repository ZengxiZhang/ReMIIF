
    
# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
import cv2
from datetime import datetime
from dataset_rivir import TestDataset_registration
from transfer import Trans_Encoder, Fusion_Decoder
from regis_net import *
from utils_ import *

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
train_log_dir = 'train_log_Oneline-FastDLT'
import warnings


def test(args):

    net = ReMIF(args)
    ir_encoder = Trans_Encoder(input_dim=1)
    vis_encoder = Trans_Encoder(input_dim=1)
    fus_decoder = Fusion_Decoder(output_dim=1)

    state_dict = torch.load('snapshot/RIVIR/ir_e.pkl', map_location='cpu')
    ir_encoder.load_state_dict(state_dict, strict=False)
    state_dict = torch.load('snapshot/RIVIR/vis_e.pkl', map_location='cpu')
    vis_encoder.load_state_dict(state_dict, strict=False)
    state_dict = torch.load('snapshot/RIVIR/regis.pkl.pkl', map_location='cpu')
    net.load_state_dict(state_dict)
    save_dir = "results/rivir_fake_10"

    if torch.cuda.is_available():
        net = net.cuda()
        ir_encoder = ir_encoder.cuda()
        vis_encoder = vis_encoder.cuda()
        fus_decoder = fus_decoder.cuda()
 

    test_data_path = 'RIVIR'
    test_data = TestDataset_registration(data_path = test_data_path)
    test_loader = DataLoader(dataset=test_data, batch_size = 1, num_workers = args.cpus, shuffle=False, drop_last=True)

    
    print("start testing")

    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则创建
    os.makedirs(os.path.join(save_dir,'registration'), exist_ok=True)  # 如果目录不存在，则创建
    os.makedirs(os.path.join(save_dir,'fusion'), exist_ok=True)  # 如果目录不存在，则创建

    net.eval()
    fus_decoder.eval()
    ir_encoder.eval()
    vis_encoder.eval()
    for iter, batch_value in enumerate(test_loader):
        ir1 = batch_value[0].float()
        vis2 = batch_value[1].float()
        
        ir1_orisize = batch_value[2].float()
        vis2_orisize = batch_value[3].float()
        vis2_orisize_color = batch_value[4].float()
        name = batch_value[5][0]

        print(name)
        if torch.cuda.is_available():
            ir1 = ir1.cuda()
            vis2 = vis2.cuda()
            ir1_orisize = ir1_orisize.cuda()
            vis2_orisize = vis2_orisize.cuda()
            vis2_orisize_color = vis2_orisize_color.cuda()

        iters_lev0 = 4
        iters_lev1 = 2
    
        with torch.no_grad():
            flow_pre, warp_vis2_orisize, vis2_orisize_color\
                    = net.test(ir1, vis2, ir1_orisize, vis2_orisize, vis2_orisize_color,\
                        ir_encoder,vis_encoder, iters_lev0 = iters_lev0, iters_lev1= iters_lev1)
            _, irf = vis_encoder(ir1_orisize)
            _, visf = ir_encoder(warp_vis2_orisize)
            _, irf = vis_encoder(ir1_orisize)
            _, visf = ir_encoder(warp_vis2_orisize)
    
            # print(ir1_orisize.shape)
            mask1 = net.genMask_decoder(ir_encoder(ir1_orisize)[-1])#  ir1
            warp_mask2 = net.genMask_decoder(vis_encoder(warp_vis2_orisize)[-1])# vis2
            mask1_resize = interpolate(mask1, size=(irf.shape[2], irf.shape[3]), mode='bilinear', align_corners=False)
            warp_mask2_resize = interpolate(warp_mask2, size=(visf.shape[2], visf.shape[3]), mode='bilinear', align_corners=False)
            fus = fus_decoder(irf, visf, mask1_resize, warp_mask2_resize)
        fus_ = fus[0,0].cpu().detach().numpy()*255
        vis2_orisize_color_ = vis2_orisize_color[0].permute(1,2,0).cpu().detach().numpy()*255
        
        ###########################################  visualization #######################################################
        cv2.imwrite(os.path.join(save_dir,'registration',name), vis2_orisize_color_)
        cv2.imwrite(os.path.join(save_dir,'fusion',name), fus_)

    print('Finished Testing')

if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--cpus', type=int, default=8, help='Number of cpus')
    parser.add_argument('--img_w', type=int, default=640)
    parser.add_argument('--img_h', type=int, default=360)
    parser.add_argument('--patch_size_h', type=int, default=315)
    parser.add_argument('--patch_size_w', type=int, default=560)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--model_name', type=str, default='resnet34')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained waights?')
    # parser.add_argument('--finetune', type=bool, default=False, help='Use pretrained waights?')

    parser.add_argument('--finetune', type=bool, default=True, help='Use pretrained waights?')

    print('<==================== Loading data ===================>\n')
    # default: not to freeze bn.
    parser.add_argument('--freeze_bn', action='store_true')
    
    parser.add_argument('--iters', type=int, default=12)
    
    parser.add_argument('--val_freq', type=int, default=10000,
                        help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='printing frequency')

    # parser.add_argument('--model_name', default='', help='specify model name')

    parser.add_argument('--position_only', default=False, action='store_true',
                        help='(GMA) only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='(GMA) use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='(GMA) number of heads in attention and aggregation')
    parser.add_argument('--posr', dest='pos_bias_radius', type=int, default=7, 
                        help='The radius of positional biases')

    parser.add_argument('--f1', dest='f1trans', type=str, 
                        choices=['none', 'shared', 'private'], default='none',
                        help='Whether to use transformer on frame 1 features. '
                             'shared:  use the same self-attention as f2trans. '
                             'private: use a private self-attention.')
    parser.add_argument('--f2', dest='f2trans', type=str, 
                        choices=['none', 'full'], default='full',
                        help='Whether to use transformer on frame 2 features.')                        

    parser.add_argument('--f2posw', dest='f2_pos_code_weight', type=float, default=0.5)
    parser.add_argument('--f2radius', dest='f2_attn_mask_radius', type=int, default=-1)
 
    parser.add_argument('--intermodes', dest='inter_num_modes', type=int, default=4, 
                        help='Number of modes in inter-frame attention')
    parser.add_argument('--intramodes', dest='intra_num_modes', type=int, default=4, 
                        help='Number of modes in intra-frame attention')
    parser.add_argument('--f2modes', dest='f2_num_modes',       type=int, default=4, 
                        help='Number of modes in F2 Transformer')
    # In inter-frame attention, having QK biases performs slightly better.
    parser.add_argument('--interqknobias', dest='inter_qk_have_bias', action='store_false', 
                        help='Do not use biases in the QK projections in the inter-frame attention')
                        
    parser.add_argument('--interpos', dest='inter_pos_code_type', type=str, 
                        choices=['lsinu', 'bias'], default='bias')
    parser.add_argument('--interposw', dest='inter_pos_code_weight', type=float, default=0.5)
    parser.add_argument('--intrapos', dest='intra_pos_code_type', type=str, 
                        choices=['lsinu', 'bias'], default='bias')
    parser.add_argument('--intraposw', dest='intra_pos_code_weight', type=float, default=1.0)
    args = parser.parse_args()
    print(args)
    test(args)


