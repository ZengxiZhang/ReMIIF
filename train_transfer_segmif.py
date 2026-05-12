# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from loss import ssim
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import cv2
# from torch_homography_model import build_model
from datetime import datetime
# from dataset_ours import TrainDataset
from dataset_rhwf_syn_fusion import TrainDataset_transfer
# from dataset_ca import TrainDataset
from utils_ import display_using_tensorboard
import util
from transfer import Trans_Encoder,Trans_Decoder
import time
from rhwf import *
# from rhwf_ours_copy import *
# from rhwf import *
from utils_ import *
from vgg import Vgg16
# name of log
import itertools
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
train_log_dir = 'train_log_Oneline-FastDLT'
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


CONTENT_WEIGHT = 1e0
TRANS_WEIGHT = 5*1e6


l1_loss = torch.nn.L1Loss()
l2_loss = torch.nn.MSELoss(reduce = True, reduction = 'mean')

def train(args):
    ir_encoder = Trans_Encoder(input_dim=1)
    vis_encoder = Trans_Encoder(input_dim=1)
    vis_decoder = Trans_Decoder(output_dim=1)
    ir_decoder = Trans_Decoder(output_dim=1)
    vgg = Vgg16()
    if torch.cuda.is_available():
        vgg = vgg.cuda()
        ir_encoder = ir_encoder.cuda()
        vis_encoder = vis_encoder.cuda()
        ir_decoder = ir_decoder.cuda()
        vis_decoder = vis_decoder.cuda()
    data_path = 'E:/ZZX/Alignment_jzy/forLJY/registration_Segmif_bigfield/train'
    train_data = TrainDataset_transfer(data_path = data_path)
    train_loader = DataLoader(dataset = train_data, batch_size = 8, num_workers = args.cpus, shuffle=True, drop_last=True)

    optimizer_ie = optim.Adam(ir_encoder.parameters(), lr=0.0005)
    optimizer_ve = optim.Adam(vis_encoder.parameters(), lr=0.0005)
    optimizer_id = optim.Adam(ir_decoder.parameters(), lr=0.0005)
    optimizer_vd = optim.Adam(vis_decoder.parameters(), lr=0.0005)
    
    scheduler_ie = optim.lr_scheduler.ExponentialLR(optimizer_ie, gamma=0.8)
    scheduler_ve = optim.lr_scheduler.ExponentialLR(optimizer_ve, gamma=0.8)
    scheduler_id = optim.lr_scheduler.ExponentialLR(optimizer_id, gamma=0.8)
    scheduler_vd = optim.lr_scheduler.ExponentialLR(optimizer_vd, gamma=0.8)

    print("start training")
    loss=[]
    loss_trans=[]
    loss_recon=[]
    loss_content=[]

    # ir_encoder.train()
    # vis_encoder.train()
    # ir_decoder.train()
    # vis_decoder.train()
    print("abc")
    state_dict = torch.load('trans_checkpoint_m3fdsegmif/ir_encoder_48.pkl', map_location='cpu')
    ir_encoder.load_state_dict(state_dict, strict=True)
    state_dict = torch.load('trans_checkpoint_m3fdsegmif/ir_decoder_48.pkl', map_location='cpu')
    ir_decoder.load_state_dict(state_dict, strict=True)
    state_dict = torch.load('trans_checkpoint_m3fdsegmif/vis_encoder_48.pkl', map_location='cpu')
    vis_encoder.load_state_dict(state_dict, strict=True)
    state_dict = torch.load('trans_checkpoint_m3fdsegmif/vis_decoder_48.pkl', map_location='cpu')
    vis_decoder.load_state_dict(state_dict, strict=True)
    for epoch in range(args.max_epoch):
        
        start = time.time()
        for iter, (ir1, vis1) in enumerate(train_loader):
            optimizer_ie.zero_grad()
            optimizer_id.zero_grad()
            optimizer_ve.zero_grad()
            optimizer_vd.zero_grad()
            # print(iter)
            ir1=ir1.float().cuda()
            vis1=vis1.float().cuda()

            _, ir1_en = ir_encoder(ir1)
            _, vis1_en = vis_encoder(vis1)
            fakevis1  = vis_decoder(ir1_en) # ir1 -> vis1  风格迁移
            fakeir1   = ir_decoder(vis1_en) # vis1 -> ir2  风格迁移
            reconir1  = ir_decoder(ir1_en) # ir1 -> ir1  重构
            reconvis1 = vis_decoder(vis1_en) # vis1 -> vis1  重构
           
            vis1_style_features = vgg(torch.cat((vis1, vis1, vis1),1))
            vis1_style_gram = [util.gram(fmap) for fmap in vis1_style_features]

            ##########这里取消了注释
            ir1_style_features = vgg(torch.cat((ir1, ir1, ir1),1))
            ir1_style_gram = [util.gram(fmap) for fmap in ir1_style_features]
            ######################


            fakevis1_hat_features = vgg(torch.cat((fakevis1, fakevis1, fakevis1),1))
            fakevis1_hat_gram = [util.gram(fmap) for fmap in fakevis1_hat_features]
            
            fakeir1_hat_features = vgg(torch.cat((fakeir1, fakeir1, fakeir1),1))
            fakeir1_hat_gram = [util.gram(fmap) for fmap in fakeir1_hat_features]

            ####加上content_loss
            vis1_vgg_features = vgg(torch.cat((vis1, vis1, vis1), 1))
            ir1_vgg_features = vgg(torch.cat((ir1, ir1, ir1), 1))
            # reconvis1_vgg_features = vgg(torch.cat((reconvis1, reconvis1, reconvis1), 1))
            # reconir1_vgg_features = vgg(torch.cat((reconir1, reconir1, reconir1), 1))
            content_loss = 0.0
            # for k in range(4):
            #     content_loss_vis2ir = l2_loss(vis1_vgg_features[k], reconir1_vgg_features[k])
            #     content_loss_ir2vis = l2_loss(ir1_vgg_features[k], reconvis1_vgg_features[k])
            #     content_loss += content_loss_vis2ir
            #     content_loss += content_loss_ir2vis
            content_loss_vis2ir = CONTENT_WEIGHT*l2_loss(vis1_vgg_features[1], fakevis1_hat_features[1])
            content_loss_ir2vis = CONTENT_WEIGHT*l2_loss(ir1_vgg_features[1], fakeir1_hat_features[1])
            content_loss += content_loss_vis2ir
            content_loss += content_loss_ir2vis

            trans_loss = 0.0
            for j in range(4):
                trans_loss += TRANS_WEIGHT*l2_loss(fakevis1_hat_gram[j], vis1_style_gram[j])
                trans_loss += TRANS_WEIGHT*l2_loss(fakeir1_hat_gram[j], ir1_style_gram[j])
            # for i in range(len(pre_4)):
                # homo_loss +=  l_lambda[i] * (pre_4[i]- gt).abs().mean()
            # mask_loss = 10 * (l1_loss(mask_forloss[:,:1], mask_forloss[:,2:3].detach()) + l1_loss(mask_forloss[:,1:2],mask_forloss[:,3:4].detach()))
            # trans_loss = 10 * (l1_loss(fakevis1, vis1) + l1_loss(fakeir2, ir2)) + 500 * ((1-ssim(fakevis1, vis1)) + (1-ssim(fakeir2, ir2)))
            recon_loss = 1 * (l1_loss(reconir1, ir1) + l1_loss(reconvis1, vis1)) + 50 * ((1 - ssim(reconir1, ir1)) + (1-ssim(reconvis1, vis1)))
            # total_loss = homo_loss + mask_loss + trans_loss + recon_los
            total_loss = trans_loss + recon_loss + content_loss

            total_loss.backward()
            optimizer_ie.step()
            optimizer_id.step()
            optimizer_ve.step()
            optimizer_vd.step()

            loss.append(total_loss.item())
            loss_trans.append(trans_loss.item())
            loss_recon.append(recon_loss.item())
            loss_content.append(content_loss.item())
            
            if iter % 40 == 0 and iter!=0:
                end = time.time()
                print('Epoch: ',str(epoch),' Iter: ',str(iter),' / ', len(train_loader),' Times: ',str(end-start), ' Total: ', str(np.mean(np.array(loss))),\
                  ' Recon: ', str(np.mean(np.array(loss_recon))),\
                            'Trans: ', str(np.mean(np.array(loss_trans))),\
                      'Content:',str(np.mean(np.array(loss_content))))
                start = time.time()
                loss = []
                loss_trans = []
                loss_recon = []
                loss_content = []
                vis1_ = vis1[0].permute(1,2,0).cpu().detach().numpy()*255
                ir1_ = ir1[0].permute(1,2,0).cpu().detach().numpy()*255
                fakeir1_ = fakeir1[0].permute(1,2,0).cpu().detach().numpy()*255
                fakevis1_ = fakevis1[0].permute(1,2,0).cpu().detach().numpy()*255
                reconir1_ = reconir1[0].permute(1,2,0).cpu().detach().numpy()*255
                reconvis1_ = reconvis1[0].permute(1,2,0).cpu().detach().numpy()*255
                fakeir1_ = np.clip(fakeir1_, 0, 255)
                fakevis1_ = np.clip(fakevis1_, 0, 255)
                reconir1_ = np.clip(reconir1_, 0, 255)
                reconvis1_ = np.clip(reconvis1_, 0, 255)
                row1 = np.concatenate((vis1_,fakevis1_,reconvis1_ ), 1)
                row2 = np.concatenate((ir1_,fakeir1_,reconir1_, ), 1)
                middle = np.concatenate((row1, row2), 0)
                cv2.imwrite('trans_middle.png', middle)
        if epoch % 4 == 0:
            torch.save(ir_encoder.state_dict(), os.path.join('trans_checkpoint_m3fdsegmif', 'ir_encoder_' + str(epoch) + '.pkl'))
            torch.save(vis_encoder.state_dict(), os.path.join('trans_checkpoint_m3fdsegmif', 'vis_encoder_' + str(epoch) + '.pkl'))
            torch.save(ir_decoder.state_dict(), os.path.join('trans_checkpoint_m3fdsegmif', 'ir_decoder_' + str(epoch) + '.pkl'))
            torch.save(vis_decoder.state_dict(), os.path.join('trans_checkpoint_m3fdsegmif', 'vis_decoder_' + str(epoch) + '.pkl'))
    print('Finished Training')


if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1, help='Number of splits')
    parser.add_argument('--cpus', type=int, default=8, help='Number of cpus')

    parser.add_argument('--img_w', type=int, default=640)
    parser.add_argument('--img_h', type=int, default=360)
    parser.add_argument('--patch_size_h', type=int, default=315)
    parser.add_argument('--patch_size_w', type=int, default=560)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--model_name', type=str, default='resnet34')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights?')
    # parser.add_argument('--finetune', type=bool, default=False, help='Use pretrained weights?')

    parser.add_argument('--finetune', type=bool, default=True, help='Use pretrained weights?')

    print('<==================== Loading data ===================>\n')
    ######## just for craft optiocal flow ############################
    # parser.add_argument('--stage', help="determines which dataset to use for training")

    # parser.add_argument('--craft', dest='craft', action='store_true', 
    #                     help='use craft (Cross-Attentional Flow Transformer)')
    # parser.add_argument('--setrans', dest='use_setrans', action='store_true', 
    #                     help='use setrans (Squeeze-Expansion Transformer) as the intra-frame attention')
    # parser.add_argument('--raft', action='store_true', help='use raft')
    # parser.add_argument('--nogma', action='store_true', help='(ablation) Do not use GMA')

    # parser.add_argument('--validation', type=str, nargs='+')
    # parser.add_argument('--restore_ckpt', help="restore checkpoint")
    # parser.add_argument('--loadopt',   dest='load_optimizer_state', action='store_true', 
    #                     help='Do not load optimizer state from checkpoint (default: not load)')
    # parser.add_argument('--loadsched', dest='load_scheduler_state', action='store_true', 
    #                     help='Load scheduler state from checkpoint (default: not load)')
    
    # parser.add_argument('--output', type=str, default='checkpoints', 
    #                     help='output directory to save checkpoints and plots')
    # parser.add_argument('--radius', dest='corr_radius', type=int, default=4)    
    # parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    # parser.add_argument('--wdecay', type=float, default=.00005)
    # parser.add_argument('--epsilon', type=float, default=1e-8)
    # parser.add_argument('--clip', type=float, default=1.0)
    # parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for fnet and cnet')
    # parser.add_argument('--upsample-learn', action='store_true', default=False,
    #                     help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    # parser.add_argument('--gamma', type=float, default=0.8, help='exponential loss weighting of the sequential predictions')
    # parser.add_argument('--add_noise', action='store_true')
    # parser.add_argument('--shiftprob', dest='shift_aug_prob', type=float,
    #                     default=0.0, help='Probability of shifting augmentation')
    # parser.add_argument('--shiftsigmas', dest='shift_sigmas', default="16,10", type=str,
    #                     help='Stds of shifts for shifting consistency loss')
                            
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
    train(args)


