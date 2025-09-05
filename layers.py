# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

gpu_use = True











def predict_flow(in_planes, mid_planes, d=2):
    dim = d
    conv_fn = getattr(nn, 'Conv%dd' % dim)
    conv_fn2 = getattr(nn, 'Conv%dd' % dim)
    layer = nn.Sequential(conv_fn(in_planes, mid_planes, kernel_size=3, padding=1),
                           conv_fn2(mid_planes, dim, kernel_size=3, padding=1))
    return layer


def conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))

