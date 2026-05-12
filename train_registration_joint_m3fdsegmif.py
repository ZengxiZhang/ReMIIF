
    
# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from loss import ssim
import itertools
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import cv2
from datetime import datetime
from dataset_rhwf_syn_fusion_joint import TrainDataset_registration
from utils_ import display_using_tensorboard
import util
from kornia.losses import SSIMLoss
from transfer import Trans_Encoder, Trans_Decoder, Fusion_Decoder
import time
from rhwf import *
from utils_ import *
from vgg import Vgg16
from loss import Gradient_loss
# from SSIM_regis_ave import *
# from com_regis_others import *
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
train_log_dir = 'train_log_Oneline-FastDLT'
import warnings
criterion_ssim = SSIMLoss(window_size=11, reduction='none').cuda()
criterion_l2 = nn.MSELoss(reduce = True, reduction = 'mean')
criterion_l1 = nn.L1Loss(reduce = True, reduction = 'mean')
triplet_loss = nn.TripletMarginLoss(margin = 1.0, p = 1, reduce = False, reduction = 'none')
criterion_grad = Gradient_loss()

warnings.filterwarnings("ignore", category=UserWarning)


now_time = datetime.now()



l1_loss=torch.nn.L1Loss()

def train(args):

    net = RHWF(args)
    ir_encoder = Trans_Encoder(input_dim=1)
    vis_encoder = Trans_Encoder(input_dim=1)
    vis_decoder = Trans_Decoder(output_dim=1)
    ir_decoder = Trans_Decoder(output_dim=1)
    fus_decoder = Fusion_Decoder(output_dim=1)

    # state_dict = torch.load('trans_checkpoint_Roadscene/ir_encoder_1.pkl', map_location='cpu')
    state_dict = torch.load('trans_checkpoint_m3fdsegmif/ir_encoder_48.pkl', map_location='cpu')
    ir_encoder.load_state_dict(state_dict, strict=True)
    for param in ir_encoder.parameters():
        param.requires_grad = False

    # state_dict = torch.load('trans_checkpoint_Roadscene/ir_decoder_1.pkl', map_location='cpu')
    state_dict = torch.load('trans_checkpoint_m3fdsegmif/ir_decoder_48.pkl', map_location='cpu')
    ir_decoder.load_state_dict(state_dict, strict=True)
    for param in ir_decoder.parameters():
        param.requires_grad = False

    # state_dict = torch.load('trans_checkpoint_Roadscene/vis_encoder_1.pkl', map_location='cpu')
    state_dict = torch.load('trans_checkpoint_m3fdsegmif/vis_encoder_48.pkl', map_location='cpu')
    vis_encoder.load_state_dict(state_dict, strict=True)
    for param in vis_encoder.parameters():
        param.requires_grad = False

    # state_dict = torch.load('trans_checkpoint_Roadscene/vis_decoder_1.pkl', map_location='cpu')
    state_dict = torch.load('trans_checkpoint_m3fdsegmif/vis_decoder_48.pkl', map_location='cpu')
    vis_decoder.load_state_dict(state_dict, strict=True)
    for param in vis_decoder.parameters():
        param.requires_grad = False

    # state_dict = torch.load('field_checkpoint_essa_roadscene_joint/140_fus_d_0.7708044648170471.pkl', map_location='cpu')
    # state_dict = torch.load('field_checkpoint_essa_m3fdsegmif_joint/62_fus_d_0.5657703280448914.pkl', map_location='cpu')
    # state_dict = torch.load('field_checkpoint_essa_m3fdsegmif_joint_fusion/2_fus_d_0.5879331827163696.pkl', map_location='cpu')
    state_dict = torch.load('field_checkpoint_essa_m3fdsegmif_joint_0321/0_fus_d_0.5470868945121765.pkl', map_location='cpu')

    fus_decoder.load_state_dict(state_dict, strict=False)

    # state_dict = torch.load('field_checkpoint_essa_roadscene_joint_newloss2/20_regis_0.7733055353164673.pkl', map_location='cpu')
    # state_dict = torch.load('field_checkpoint_essa_m3fdsegmif_joint/62_regis_0.5657703280448914.pkl', map_location='cpu')
    # state_dict = torch.load('field_checkpoint_essa_m3fdsegmif_joint/62_regis_0.5657703280448914.pkl', map_location='cpu')
    # state_dict = torch.load('field_checkpoint_essa_m3fdsegmif_joint_fusion/2_regis_0.5879331827163696.pkl', map_location='cpu')
    # state_dict = torch.load('field_checkpoint_essa_nirscene2/5_regis_0.7350937724113464.pkl', map_location='cpu')
    state_dict = torch.load('field_checkpoint_essa_m3fdsegmif_joint_0321/0_regis_0.5470868945121765.pkl', map_location='cpu')
    net.load_state_dict(state_dict, strict=True)

    # net.load_state_dict(state_dict, strict=False)
# 
    # vgg = Vgg16()#.type(torch.cuda.FloatTensor)
    if torch.cuda.is_available():
        net = net.cuda()
        # vgg = vgg.cuda()
        ir_encoder = ir_encoder.cuda()
        vis_encoder = vis_encoder.cuda()
        ir_decoder = ir_decoder.cuda()
        vis_decoder = vis_decoder.cuda()
        fus_decoder = fus_decoder.cuda()
        
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 原来是1e-4
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 原来是1e-4
    # optimizer = optim.Adam(itertools.chain(ir_encoder.parameters(), vis_encoder.parameters() ,\
    #                         ir_decoder.parameters(), vis_decoder.parameters(), fus_decoder.parameters(), net.parameters()), lr=0.001)  # 原来是1e-4
    optimizer = optim.Adam(itertools.chain(fus_decoder.parameters(), net.parameters()), lr=0.0001)  # 原来是1e-4
    # optimizer = optim.Adam(itertools.chain(net.parameters(),fus_decoder.parameters()), lr=0.0001)  # 原来是1e-4
    # optimizer = optim.Adam(itertools.chain(fus_decoder.parameters()), lr=0.0001)  # 原来是1e-4
    # optimizer = optim.Adam(itertools.chain(fus_decoder.parameters()), lr=0.0001)  # 原来是1e-4
    # optimizer = optim.Adam(itertools.chain(net.parameters()), lr=0.0001)  # 原来是1e-4
    # optimizer = optim.Adam(itertools.chain(ir_encoder.parameters(), vis_encoder.parameters() ,\
    #                         ir_decoder.parameters(), vis_decoder.parameters(), net.parameters()), lr=0.001)  # 原来是1e-4
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=1)  # 本来是0.8
    # state_dict = torch.load('field_checkpoint_essa_nirroadscene/10_regis_opt_0.8084038496017456.pkl', map_location='cpu')
    # state_dict = torch.load('field_checkpoint_essa_m3fdsegmif_joint/62_regis_opt_0.5657703280448914.pkl', map_location='cpu')
    # state_dict = torch.load('field_checkpoint_essa_m3fdsegmif_joint_fusion/2_regis_opt_0.5879331827163696.pkl', map_location='cpu')
    # state_dict = torch.load('field_checkpoint_essa_nirscene2/5_regis_opt_0.7350937724113464.pkl', map_location='cpu')
    state_dict = torch.load('field_checkpoint_essa_m3fdsegmif_joint_0321/0_regis_opt_0.5470868945121765.pkl', map_location='cpu')
    optimizer.load_state_dict(state_dict)

    # data_path = '../ragistration_roadscene/train'
    # test_data_path = '../ragistration_roadscene/test'
    # data_path = '../ragistration_roadscene_bigfield/train'
    # test_data_path = '../ragistration_roadscene_bigfield/test'
    
    data_path = 'E:/ZZX/Alignment_jzy/forLJY/registration_M3FDSegmif_bigfield/train'
    test_data_path = 'E:/ZZX/Alignment_jzy/forLJY/registration_M3FDSegmif_bigfield/test'
    # train_data = TrainDataset_registration(data_path = data_path, exp_path = exp_name, patch_w = args.patch_size_w, patch_h = args.patch_size_h, rho=16)
    train_data = TrainDataset_registration(data_path = data_path, patch_w = args.patch_size_w, patch_h = args.patch_size_h, rho=16)
    train_loader = DataLoader(dataset=train_data, batch_size = 8, num_workers = args.cpus, shuffle=True, drop_last=True)
    # load vgg network
    
    # train_loader = DataLoader(dataset=train_data, batch_size=4, num_workers=args.cpus, shuffle=True, drop_last=True)
    
    print("start training")

    loss_total = []
    loss_trans = []
    loss_recon = []
    loss_mask = []
    loss_homo = []
    loss_field = []
    loss_feature = []
    loss_grad = []
    loss_align = []
    loss_l1 = []
    loss_ssim = []
    loss_fus = []
    # save_dir = "field_checkpoint_essa_roadscene_joint_newloss2"
    save_dir = "field_checkpoint_essa_m3fdsegmif_joint_0321"
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则创建
    # ssim_best = 0
    for epoch in range(args.max_epoch):
        # if (epoch % 5 == 0 and epoch != 0) or epoch == 149:
        # if ((epoch % 2 == 0) and epoch!=0) or epoch == 149:
        if ((epoch % 2 == 0)) or epoch == 149:
        # if epoch % 5 == 0  or epoch == 149:
            ssim_ave = test(args, test_data_path, net, ir_encoder, vis_encoder, ir_decoder, vis_decoder)
            print('Epoch:', str(epoch),' Mean SSIM: ', str(ssim_ave))
            # torch.save(ir_encoder.state_dict(), os.path.join(save_dir, f"{epoch}_ir_e_{ssim_ave}.pkl"))
            # torch.save(ir_decoder.state_dict(), os.path.join(save_dir, f"{epoch}_ir_d_{ssim_ave}.pkl"))
            # torch.save(vis_encoder.state_dict(), os.path.join(save_dir, f"{epoch}_vis_e_{ssim_ave}.pkl"))
            # torch.save(vis_decoder.state_dict(), os.path.join(save_dir, f"{epoch}_vis_d_{ssim_ave}.pkl"))
            torch.save(fus_decoder.state_dict(), os.path.join(save_dir, f"{epoch}_fus_d_{ssim_ave}.pkl"))
            torch.save(net.state_dict(), os.path.join(save_dir, f"{epoch}_regis_{ssim_ave}.pkl"))
            # torch.save(fus_decoder.state_dict(), os.path.join(save_dir, f"{epoch}_fus_d_{ssim_ave}.pkl"))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, f"{epoch}_regis_opt_{ssim_ave}.pkl"))

        net.train()
        fus_decoder.train()
        ir_encoder.eval()
        ir_decoder.eval()
        vis_encoder.eval()
        vis_decoder.eval()
        # ir_encoder.train()
        # ir_decoder.train()
        # vis_encoder.train()
        # vis_decoder.train()
        start = time.time()
        scheduler.step()  # Note: The initial learning rate should be 1e-4. torch_version==1.0.1 ->init lr == 0.0001; torch_version>=1.2.0 ->init lr == 0.0001*1.25?
        print(epoch, 'lr={:.6f}'.format(scheduler.get_lr()[0]))
        for iter, batch_value in enumerate(train_loader):
            ir1 = batch_value[0].float()
            vis2 = batch_value[1].float()
            vis1 = batch_value[2].float()
            ir2 = batch_value[3].float()
            gt_shift = batch_value[4].float()
            gt_disp = batch_value[5].float()
            ir_vsm = batch_value[6].float()
            vis_vsm = batch_value[7].float()
        
            if torch.cuda.is_available():
                ir1 = ir1.cuda()
                vis2 = vis2.cuda()
                vis1 = vis1.cuda()
                ir2 = ir2.cuda()
                gt_shift = gt_shift.cuda()
                gt_disp = gt_disp.cuda()
                ir_vsm = ir_vsm.cuda()#.detach()
                vis_vsm = vis_vsm.cuda()#.detach()

            optimizer.zero_grad()

            iters_lev0 = 4
            iters_lev1 = 2
        
            pre_4, vis_trans_all, ir_trans_all, allonemask_trans_all, mask_all, mask_forloss, \
            fakevis1, fakeir2, reconir1, reconvis2, mask_I1, mask_I2, \
            warp_I2_norm, img1_en2, img2_en2, warp_ir2, disp_list,flow_homo_gt,flow_field_gt,warp_vis_vsm\
                  = net(ir1, vis2, vis1, ir2, ir_encoder,vis_encoder,ir_decoder,vis_decoder, gt_shift, gt_disp, 
                        iters_lev0 = iters_lev0, iters_lev1= iters_lev1,vis_vsm=vis_vsm)
            
            warp_vis2 = vis_trans_all[:, -2:-1, :, :]
            warp_ir2 = ir_trans_all[:, -2:-1, :, :]
            warp_all_one_mask = allonemask_trans_all[:, -2:-1, :, :]
            warp_vis2_m = warp_I2_norm
            _, irf = vis_encoder(ir1)
            _, visf = ir_encoder(warp_vis2)
            # irf1, irf2 = ir_encoder(ir1)
            # visf1, visf2 = vis_encoder(warp_vis2)
            mask_I1_32 = interpolate(mask_I1, size=(32, 32), mode='bilinear', align_corners=False)
            warp_vis2_m_32 = interpolate(warp_vis2_m, size=(32, 32), mode='bilinear', align_corners=False)

            # fus = fus_decoder(torch.cat((irf2 * mask_I1_32, visf2 * warp_vis2_m_32),1))
            fus = fus_decoder(irf, visf, mask_I1_32, warp_vis2_m_32)
            # fus = ir1
            # fus = fus_decoder(torch.cat((irf2 , visf2 ),1))
            
            homo_loss = 0.0
            field_loss = 0.0
            total_loss = 0.0
            weight = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
            l_lambda = [16. / pow(2, i) for i in weight[:iters_lev0 + iters_lev1]]
            

            feature_loss = criterion_l1(fakevis1 * mask_I1 * warp_vis2_m,warp_vis2 * warp_vis2_m * mask_I1)/  \
            (0.01 * criterion_l1(fakevis1 * mask_I1 * warp_vis2_m, vis2 * mask_I2)) + \
            (criterion_l1(fakeir2 * mask_I1 * warp_vis2_m, warp_ir2 * warp_vis2_m * mask_I1) /  \
             (0.01 * criterion_l1(ir1 * mask_I1 * warp_vis2_m, fakeir2 * mask_I2)))
            # feature_loss = criterion_l1(fakevis1 * mask_I1 * warp_vis2_m,warp_vis2 * warp_vis2_m * mask_I1)  + \
            # criterion_l1(fakeir2 * mask_I1 * warp_vis2_m,  warp_ir2 * warp_vis2_m * mask_I1) 
            # feature_loss = criterion_l1(fakevis1 * mask_I1 * warp_vis2_m, warp_vis2 * warp_vis2_m * mask_I1)+\
            #                             criterion_l1(fakeir2 * mask_I1 * warp_vis2_m,warp_ir2 * warp_vis2_m * mask_I1)
            # feature_loss = criterion_l1(fakevis1 * mask_I1 * warp_vis2_m,
            #                                 warp_vis2 * warp_vis2_m * mask_I1) 
            ######################################################################################################################
            ########### calculate loss (supervise) ###############################################################################
            # vis1_style_features = vgg(torch.cat((vis1, vis1, vis1),1))
            # vis1_style_gram = [util.gram(fmap) for fmap in vis1_style_features]
            # ir2_style_features = vgg(torch.cat((ir2, ir2, ir2),1))
            # ir2_style_gram = [util.gram(fmap) for fmap in ir2_style_features]
            # fakevis1_hat_features = vgg(torch.cat((fakevis1, fakevis1, fakevis1),1))
            # fakevis1_hat_gram = [util.gram(fmap) for fmap in fakevis1_hat_features]
            # fakeir2_hat_features = vgg(torch.cat((fakeir2, fakeir2, fakeir2),1))
            # fakeir2_hat_gram = [util.gram(fmap) for fmap in fakeir2_hat_features]
            trans_loss = 0.0
            # for j in range(4):
                # trans_loss += 10*l1_loss(fakevis1_hat_gram[j], vis1_style_gram[j])
                # trans_loss += 10*l1_loss(fakeir2_hat_gram[j], ir2_style_gram[j])
            # for i in range(len(pre_4)):
            #     homo_loss +=  l_lambda[i] * (pre_4[i] - gt_shift).abs().mean()
            # print(len(pre_4))
            # print(pre_4[0][0])
            # print(pre_4[3][0])
            # print(gt_shift[0])
            # print(a)
            homo_loss =  l_lambda[-1] * (pre_4[-1] - gt_shift).abs().mean()
            # homo_loss = homo_loss
            l_a_w = 10*[16, 16, 8, 8, 4, 4, 2, 2]
            # print(vis_trans_all.shape,ir_trans_all.shape,allonemask_trans_all.shape)
            align_loss =  0
            for i in range(2, 2+iters_lev0+iters_lev1):
                # print(i)
                # align_loss = l_a_w[i-1] * l1_loss(fakevis1*allonemask_trans_all[:, i, :, :], vis_trans_all[:, i, :, :])+\
                #     l_a_w[i-1] * l1_loss(ir1*allonemask_trans_all[:, i, :, :], ir_trans_all[:, i, :, :])
                align_loss += l_a_w[i-1] * l1_loss(vis1*allonemask_trans_all[:, i, :, :], vis_trans_all[:, i, :, :])#+\
                    # l_a_w[i-1] * l1_loss(ir1*allonemask_trans_all[:, i, :, :], ir_trans_all[:, i, :, :])
            # scale = ir2.shape[-1]/2
            # disp_list = disp_list * scale
            # gt_disp = gt_disp * scale     
            grad_loss = 16 * criterion_grad(disp_list[:,10:12]-disp_list[:,8:10]) + 4 * criterion_grad(disp_list[:,12:14]-disp_list[:,8:10])
            # for i in range(iters_lev0):
            #     pred_homo_flow = disp_list[:,(i+1)*2:(i+1)*2+2]
            #     field_loss += 16./(2 **i) * l1_loss(pred_homo_flow, flow_homo_gt) 
            for i in range(iters_lev0, iters_lev0 + iters_lev1):
                pred_field_flow = disp_list[:,(i+1)*2:(i+1)*2+2]
                field_loss += 5* 16./(2 **i) * l1_loss(pred_field_flow, flow_field_gt) 
            # field_loss +=  2 * l1_loss(disp_list[:,2:4], flow_field_gt) + 0.5 * l1_loss(disp_list[:,4:6], flow_field_gt)
            # grad_loss = 0
            # field_loss = 0
            # trans_loss = 10 * (l1_loss(fakevis1, vis1) + l1_loss(fakeir2, ir2)) + 500 * ((1-ssim(fakevis1, vis1)) + (1-ssim(fakeir2, ir2)))
            recon_loss = 1 * (l1_loss(reconir1, ir1) + l1_loss(reconvis2, vis2)) + 50 * ((1-ssim(reconir1, ir1)) + (1-ssim(reconvis2, vis2)))
            # recon_loss = 0
            # total_loss = homo_loss + mask_loss + trans_loss + recon_loss
            # total_loss = trans_loss + recon_loss + feature_loss + homo_loss # + mask_loss
            # l_l1 = 1 * (criterion_l1(fus, ir1) + criterion_l1(fus, warp_vis2))
            # l_ssim = 100 * ((1-ssim(fus, ir1)) + (1-ssim(fus, warp_vis2)))

            
            b1 = 10
            b2 = 1000
            b3 = 10
            l_ir =  b1 * criterion_l1(fus, ir1)+b2 * criterion_ssim(fus, ir1) + b3 * criterion_grad(fus, ir1)
            l_vis = b1 * criterion_l1(fus, warp_vis2)+ b2* criterion_ssim(fus, warp_vis2) + b3 * criterion_grad(fus, warp_vis2)
            ### w1, w2 = 0.5 + 0.5 * (ir_vsm - warp_vis_vsm), 0.5 + 0.5 * (warp_vis_vsm - ir_vsm)  # data driven loss weights
            # 计算 softmax
            exp_ir = torch.exp(ir_vsm)  # 对 img1 指数化
            exp_vis = torch.exp(vis_vsm)  # 对 img2 指数化
            w1 = exp_ir / (exp_ir + exp_vis + 1e-2)
            w2 = exp_vis / (exp_ir + exp_vis + 1e-2)
            l_fus = (w1 * l_ir + w2 * l_vis).mean() #+ b3 * criterion_grad(fus, ir, vis) # fus <- ssim + l1 -> (ir, vi)
  
            # total_loss = homo_loss  + grad_loss #+ field_loss + l_fus
            total_loss = homo_loss  + grad_loss +feature_loss#+ field_loss# + l_fus
            #######################################################################################################################

            total_loss = total_loss.cuda()
            total_loss.backward()
            optimizer.step()

            loss_total.append(total_loss.item())
            # loss_trans.append(trans_loss.item())
            loss_trans.append(trans_loss)
            loss_recon.append(recon_loss.item())
            # loss_recon.append(recon_loss)
            # loss_mask.append(mask_loss.item())
            loss_homo.append(homo_loss.item())
            loss_field.append(field_loss.item())
            loss_feature.append(feature_loss.item())
            loss_grad.append(grad_loss.item())
            loss_align.append(align_loss.item())
            loss_fus.append(l_fus.item())
            # loss_fus.append(0)
            # loss_sigma_feature += loss_feature.item()
            # if it
            if iter %150 == 0:
                end = time.time()
                # print(torch.min(disp_list[:,4:6]-disp_list[:,0:2]).item(),torch.max(disp_list[:,4:6]-disp_list[:,0:2]).item(),\
                #     torch.min(flow_field_gt-disp_list[:,0:2]).item(),torch.max(flow_field_gt-disp_list[:,0:2]).item())
                print('Epoch:',str(epoch),' Itr:',str(iter),'/', len(train_loader),' Times:',"{:.3f}".format(end-start), 
                      ' Total:', "{:.3f}".format(np.mean(np.array(loss_total))),' Homo:', "{:.3f}".format(np.mean(np.array(loss_homo))), 
                      ' Field:', "{:.3f}".format(np.mean(np.array(loss_field))), ' Grad:', "{:.3f}".format(np.mean(np.array(loss_grad))), 
                      ' Recon:', "{:.3f}".format(np.mean(np.array(loss_recon))), 
                      ' Trans:', "{:.3f}".format(np.mean(np.array(loss_trans))), 'Fea:', "{:.3f}".format(np.mean(np.array(loss_feature))),
                      ' Align:', "{:.3f}".format(np.mean(np.array(loss_align))), 'Fus:', "{:.3f}".format(np.mean(np.array(loss_fus))))
                    #   ' SSIM:', "{:.3f}".format(np.mean(np.array(loss_ssim))), 'L1:', "{:.3f}".format(np.mean(np.array(loss_l1))))
                start = time.time()
                loss_total = []
                loss_trans = []
                loss_recon = []
                loss_mask = []
                loss_homo = []
                loss_feature = []
                loss_grad = []
                loss_field = []
                loss_align = []
                loss_fus = []
                vis_trans_all = vis_trans_all[0].permute(1,2,0).cpu().detach().numpy()*255

                ir2_ = ir2[0].permute(1,2,0).cpu().detach().numpy()*255
                vis1_ = vis1[0].permute(1,2,0).cpu().detach().numpy()*255
                ir1_ = ir1[0].permute(1,2,0).cpu().detach().numpy()*255
                vis2_ = vis2[0].permute(1,2,0).cpu().detach().numpy()*255
                fakeir2_ = fakeir2[0].permute(1,2,0).cpu().detach().numpy()*255
                fakevis1_ = fakevis1[0].permute(1,2,0).cpu().detach().numpy()*255
                reconir1_ = reconir1[0].permute(1,2,0).cpu().detach().numpy()*255
                reconvis2_ = reconvis2[0].permute(1,2,0).cpu().detach().numpy()*255
                fus_ = fus[0].permute(1,2,0).cpu().detach().numpy()*255
                mask_forloss=mask_forloss[0].permute(1,2,0).cpu().detach().numpy()*255
                mask1 = mask_forloss[...,:1]
                mask2 = mask_forloss[...,1:2]
                
                ###########################################  visualization #######################################################
                input1 = vis_trans_all[..., 0: 1]
                warp_gt = vis_trans_all[...,8:9]
                img_stitch1 = warp_gt.copy()*1.
                img_stitch1 = (warp_gt+input1)/2
                row1 = np.concatenate((vis_trans_all[..., 0: 1], vis_trans_all[..., 1: 2], vis_trans_all[..., 2: 3], vis_trans_all[..., 3:4]), 1)
                row2 = np.concatenate((vis_trans_all[...,4:5], vis_trans_all[...,5:6], vis_trans_all[...,6:7], vis_trans_all[...,7:8]), 1)
                row3 = np.concatenate((mask1, mask2, vis_trans_all[...,8:9], img_stitch1), 1)
                row4 = np.concatenate((fakevis1_, vis1_, fakeir2_, ir2_), 1)
                row5 = np.concatenate((reconir1_, ir1_, reconvis2_, vis2_), 1)
                ave_heatmap_ir = np.mean(img1_en2.cpu().detach().numpy(), axis=(0, 1))
                ave_heatmap_vis = np.mean(img2_en2.cpu().detach().numpy(), axis=(0, 1))
                heatmap_normalized_ir = cv2.normalize(ave_heatmap_ir, None, 0, 255, cv2.NORM_MINMAX).astype(
                    np.uint8)
                heatmap_normalized_vis = cv2.normalize(ave_heatmap_vis, None, 0, 255, cv2.NORM_MINMAX).astype(
                    np.uint8)
                heatmap_colored_ir = cv2.applyColorMap(heatmap_normalized_ir, cv2.COLORMAP_JET)
                heatmap_colored_vis = cv2.applyColorMap(heatmap_normalized_vis, cv2.COLORMAP_JET)
                heatmap_colored_ir = cv2.resize(heatmap_colored_ir,(128,128))
                heatmap_colored_vis = cv2.resize(heatmap_colored_vis,(128,128))
                black = np.zeros_like(heatmap_colored_ir)
                fus_ = cv2.cvtColor(np.uint8(fus_), cv2.COLOR_GRAY2RGB)
                a = (vis1[0]*allonemask_trans_all[0, 7, :, :])[0].cpu().detach().numpy()*255
                b = vis_trans_all[...,7]
                # print(a.shape, b.shape)
                a = cv2.cvtColor(np.uint8(a), cv2.COLOR_GRAY2BGR)
                b = cv2.cvtColor(np.uint8(b), cv2.COLOR_GRAY2BGR)
                # print(heatmap_colored_ir.shape, a.shape, b.shape)
                row6 = np.concatenate((heatmap_colored_ir, heatmap_colored_vis, fus_, b), 1)
                middle = np.concatenate((row1, row2, row3, row4, row5), 0)
                middle = cv2.cvtColor(np.uint8(middle), cv2.COLOR_GRAY2BGR)
                middle = np.concatenate((middle,row6), 0)
                cv2.imwrite('middle.png', middle)

    print('Finished Training')

def test(args, test_data_path, net, ir_encoder, vis_encoder, ir_decoder, vis_decoder):
    test_data = TrainDataset_registration(data_path=test_data_path, patch_w=args.patch_size_w,
                                           patch_h=args.patch_size_h, rho=16,use_vsm=False)
    test_loader = DataLoader(dataset=test_data, batch_size=8, num_workers= args.cpus, shuffle=False, drop_last=False)
    # print(len(test_loader))
    ssim_test = []
    net.eval()
    ir_encoder.eval()
    ir_decoder.eval()
    vis_encoder.eval()
    vis_decoder.eval()
    for iter, batch_value in enumerate(test_loader):
        # org_imges = batch_value[0].float()
        # ref_imges = batch_value[1].float()
        # input_tesnors = batch_value[2].float()
        # patch_indices = batch_value[3].float()
        # h4p = batch_value[4].float()
        # gt_shift = batch_value[5].float().cuda()
        # gt_disp = batch_value[6].float().cuda()
        ir1 = batch_value[0].float()
        vis2 = batch_value[1].float()
        vis1 = batch_value[2].float()
        ir2 = batch_value[3].float()
        gt_shift = batch_value[4].float()
        gt_disp = batch_value[5].float()

        if torch.cuda.is_available():
            ir1 = ir1.cuda()
            vis2 = vis2.cuda()
            ir2 = ir2.cuda()
            vis1 = vis1.cuda()
            gt_disp = gt_disp.cuda()
            gt_shift = gt_shift.cuda()
            # input_tesnors = input_tesnors.cuda()
            # patch_indices = patch_indices.cuda()
            # h4p = h4p.cuda()
        # ir2 = ref_imges[:, 1:, ...]
        iters_lev0 = 4
        iters_lev1 = 2
        # shift, disp, vis_warp_gt, vis2_warp_field, mask_warp_gt,vis1\
        #           = net(org_imges, ref_imges, input_tesnors, h4p, patch_indices,ir_encoder,vis_encoder,ir_decoder,vis_decoder,
        #                 ir2, gt_shift, gt_disp, iters_lev0 = iters_lev0, iters_lev1= iters_lev1,test_mode=True)
        shift, disp, vis_warp_gt, vis2_warp_field, mask_warp_gt,vis1\
                  = net(ir1, vis2, vis1, ir2,ir_encoder, vis_encoder, ir_decoder, vis_decoder,
                        gt_shift, gt_disp, iters_lev0 = iters_lev0, iters_lev1= iters_lev1,test_mode=True)
        gt_mask = mask_warp_gt * vis1
        gt_warp = vis_warp_gt
        pre_warp = vis2_warp_field
        # middle = torch.cat((gt_mask,gt_warp,pre_warp),2)[0].permute(1,2,0).detach().cpu().numpy()*255
        ssim_test.append(ssim(gt_mask, pre_warp).item())
        return np.mean(np.array(ssim_test))
if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1, help='Number of splits')
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
    train(args)


