import torch
import torch.nn as nn
import time
import torch.nn.functional as F
def conv(in_channels, out_channels, kernel_size, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride)
    
    

class mergeblock(nn.Module):
    def __init__(self, n_feat, kernel_size, subspace_dim=16):
        super(mergeblock, self).__init__()
        self.conv_block = conv(n_feat * 2, n_feat, kernel_size)
        self.num_subspace = subspace_dim
        self.subnet = conv(n_feat * 2, self.num_subspace, kernel_size)

    def forward(self, x, bridge):
        out = torch.cat([x, bridge], 1)# b, 256, 32, 32
        b_, c_, h_, w_ = bridge.shape
        sub = self.subnet(out)# b, 16, 32, 32
        V_t = sub.view(b_, self.num_subspace, h_*w_)# b, 16, 32*32
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))
        V = V_t.permute(0, 2, 1)# b, 32*32, 16
        mat = torch.matmul(V_t, V)# b, 16, 16
        mat_inv = torch.inverse(mat)
        project_mat = torch.matmul(mat_inv, V_t)
        bridge_ = bridge.view(b_, c_, h_*w_)
        project_feature = torch.matmul(project_mat, bridge_.permute(0, 2, 1))
        bridge = torch.matmul(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)
        out = torch.cat([x, bridge], 1)
        out = self.conv_block(out)
        return out+x


# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride) #, padding)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# Residual Block
#   adapted from pytorch tutorial
#   https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-
#   intermediate/deep_residual_network/main.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out 
    
class ImageTransformNet(nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()
        
        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # encoding layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(32, affine=True)

        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(64, affine=True)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(128, affine=True)

        # residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # decoding layers
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2 )
        self.in3_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2 )
        self.in2_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv1 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1)
        self.in1_d = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        # residual layers
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))
        #y = self.tanh(self.in1_d(self.deconv1(y)))
        y = self.deconv1(y)

        return y
    
class Trans_Encoder(nn.Module):
    def __init__(self,input_dim=3):
        super(Trans_Encoder, self).__init__()
        
        # nonlineraity
        self.relu = nn.ReLU()

        # encoding layers
        self.conv1 = ConvLayer(input_dim, 64, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(64, affine=True)

        self.conv2 = ConvLayer(64, 96, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(96, affine=True)

        self.conv3 = ConvLayer(96, 128, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(128, affine=True)

        # residual layers
        # self.res1 = ResidualBlock(64)
        # self.res2 = ResidualBlock(96)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
 

        y1 = self.relu( self.in2_e(self.conv2(y)) )
        # print(y.shape)
        y = self.relu( self.in3_e(self.conv3(y1)) )
        y = self.res1(y)
        y = self.res2(y)
        y2 = self.res3(y)

        return y1, y2
    
class Trans_Decoder(nn.Module):
    def __init__(self, output_dim=3):
        super(Trans_Decoder, self).__init__()
        
        # nonlineraity
        self.relu = nn.ReLU()

        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # decoding layers
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2 )
        self.in3_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2 )
        self.in2_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv1 = UpsampleConvLayer(32, output_dim, kernel_size=9, stride=1)
        # self.in1_d = nn.InstanceNorm2d(output_dim, affine=True)

    def forward(self, x):

        y = self.res4(x)
        y = self.res5(y)

        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))
        #y = self.tanh(self.in1_d(self.deconv1(y)))
        y = self.deconv1(y)

        return y

class Mask_Decoder(nn.Module):
    def __init__(self, output_dim=3):
        super(Mask_Decoder, self).__init__()

        # nonlineraity
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # decoding layers
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        # self.in3_d = nn.InstanceNorm2d(64, affine=True)
        self.in3_d = nn.BatchNorm2d(64)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        # self.in2_d = nn.InstanceNorm2d(32, affine=True)
        self.in2_d = nn.BatchNorm2d(32)

        self.deconv1 = UpsampleConvLayer(32, output_dim, kernel_size=9, stride=1)
        self.in1_d = nn.BatchNorm2d(output_dim)

        # self.in1_d = nn.InstanceNorm2d(output_dim, affine=True)

    def forward(self, x):
        y = self.res4(x)
        y = self.res5(y)

        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))
        # y = self.tanh(self.in1_d(self.deconv1(y)))
        y = self.sig(self.in1_d(self.deconv1(y)))

        return y

# -------------------------------------------------------------------#
class ChannelAttention(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super(ChannelAttention, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        q, k, v = self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3,
                                                                                                                  dim=1)
        C_exp = self.expan_att_chans * C

        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.t

        x_ = attn.softmax(dim=-1) @ v
        x_ = x_.view(B, self.expan_att_chans, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()

        x_ = self.group_fus(x_)
        return x_


class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super(SpatialAttention, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        q, k, v = self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3,
                                                                                                                  dim=1)
        C_exp = self.expan_att_chans * C

        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-2), F.normalize(k, dim=-2)
        attn = q.transpose(-2, -1) @ k * self.t

        x_ = attn.softmax(dim=-1) @ v.transpose(-2, -1)
        x_ = x_.transpose(-2, -1).contiguous()

        x_ = x_.view(B, self.expan_att_chans, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()

        x_ = self.group_fus(x_)
        return x_


class CondensedAttentionNeuralBlock(nn.Module):
    def __init__(self, embed_dim=64, squeezes=(4,1), shuffle=2, expan_att_chans=4):
        super(CondensedAttentionNeuralBlock, self).__init__()
        self.embed_dim = embed_dim

        sque_ch_dim = embed_dim // squeezes[0]
        shuf_sp_dim = int(sque_ch_dim * (shuffle ** 2))
        sque_sp_dim = shuf_sp_dim // squeezes[1]

        self.sque_ch_dim = sque_ch_dim
        self.shuffle = shuffle
        self.shuf_sp_dim = shuf_sp_dim
        self.sque_sp_dim = sque_sp_dim

        self.ch_sp_squeeze = nn.Sequential(
            nn.Conv2d(embed_dim, sque_ch_dim, 1),
            nn.Conv2d(sque_ch_dim, sque_sp_dim, shuffle, shuffle, groups=sque_ch_dim)
        )

        self.channel_attention = ChannelAttention(sque_sp_dim, sque_ch_dim, expan_att_chans)
        self.spatial_attention = SpatialAttention(sque_sp_dim, sque_ch_dim, expan_att_chans)

        self.sp_ch_unsqueeze = nn.Sequential(
            nn.Conv2d(sque_sp_dim, shuf_sp_dim, 1, groups=sque_ch_dim),
            nn.PixelShuffle(shuffle),
            nn.Conv2d(sque_ch_dim, embed_dim, 1)
        )

    def forward(self, x):
        x = self.ch_sp_squeeze(x)

        group_num = self.sque_ch_dim
        each_group = self.sque_sp_dim // self.sque_ch_dim
        idx = [i + j * group_num for i in range(group_num) for j in range(each_group)]
        x = x[:, idx, :, :]

        x = self.channel_attention(x)
        nidx = [i + j * each_group for i in range(each_group) for j in range(group_num)]
        x = x[:, nidx, :, :]
        x = self.spatial_attention(x)
        
        x = self.sp_ch_unsqueeze(x)
        return x
class Fusion_Decoder(nn.Module):
    def __init__(self, output_dim=1):
        super(Fusion_Decoder, self).__init__()
        
        self.mask_conv1 = ConvLayer(1, 64, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.mask_conv2 = ConvLayer(64, 128, kernel_size=9, stride=1)
        self.in2 = nn.InstanceNorm2d(128, affine=True)
        self.mask_fuse = mergeblock(128, 3)

        # nonlineraity
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        # self.res4 = ResidualBlock(256)
        # self.res5 = ResidualBlock(256)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # decoding layers
        # self.deconv3 = UpsampleConvLayer(256, 64, kernel_size=3, stride=1, upsample=2)
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.BatchNorm2d(64)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        # self.in2_d = nn.InstanceNorm2d(32, affine=True)
        self.in2_d = nn.BatchNorm2d(32)

        self.deconv1 = UpsampleConvLayer(32, output_dim, kernel_size=9, stride=1)
        self.in1_d = nn.BatchNorm2d(output_dim)
        self.can = CondensedAttentionNeuralBlock(embed_dim=128)
        # self.in1_d = nn.InstanceNorm2d(output_dim, affine=True)

    def forward(self, ir_f, vis_f, mask1, mask2):
        m1_f = self.relu(self.in2(self.mask_conv2(self.relu(self.in1(self.mask_conv1(mask1))))))
        m2_f = self.relu(self.in2(self.mask_conv2(self.relu(self.in1(self.mask_conv1(mask2))))))
        
        ir_fus = self.mask_fuse(ir_f, m1_f)
        vis_fus = self.mask_fuse(vis_f, m2_f)
        x = torch.cat((ir_fus, vis_fus), 1)
        ####################################################
        x = self.can(ir_fus+vis_fus)#
        # x = ir_fus + vis_fus
        ####################################################
    
        y = self.res4(x)
        y = self.res5(y)
        
        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))
        # y = self.tanh(self.in1_d(self.deconv1(y)))
        y = self.sig(self.in1_d(self.deconv1(y)))

        return y

if __name__ == '__main__':
    A = torch.ones(4, 3, 32, 32)
    encoder = Trans_Encoder()
    _, B = encoder(A)
    print('done!')