from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import time
import random as rd
def make_mesh(patch_w,patch_h):
    x_flat = np.arange(0,patch_w)
    x_flat = x_flat[np.newaxis,:]
    y_one = np.ones(patch_h)
    y_one = y_one[:,np.newaxis]
    x_mesh = np.matmul(y_one , x_flat)

    y_flat = np.arange(0,patch_h)
    y_flat = y_flat[:,np.newaxis]
    x_one = np.ones(patch_w)
    x_one = x_one[np.newaxis,:]
    y_mesh = np.matmul(y_flat,x_one)
    return x_mesh,y_mesh


    
class TestDataset_registration(Dataset):
    def __init__(self, data_path, patch_w = None, patch_h = None):

        self.path = data_path
        self.names = sorted(os.listdir(os.path.join(self.path,'VIS')))#[:1]
        self.WIDTH = patch_w
        self.HEIGHT = patch_h


    def __getitem__(self, index):
        name=self.names[index]
        ir1 = cv2.imread(os.path.join(self.path, 'IR', name))

        ir1_128 = cv2.resize(ir1, (128, 128))
        ir1_128 = cv2.cvtColor(ir1_128, cv2.COLOR_BGR2GRAY)[None]/ 255.
        
        ir1 = cv2.cvtColor(ir1, cv2.COLOR_BGR2GRAY)[None]/ 255.
        
        vis2 = cv2.imread(os.path.join(self.path, 'VIS', name))
        
        vis2_128 = cv2.resize(vis2, (128, 128))
        vis2_128 = cv2.cvtColor(vis2_128, cv2.COLOR_BGR2GRAY)[None]/ 255.
        
        vis2_color = np.transpose(vis2, [2, 0, 1])/255.
        
        vis2 = cv2.cvtColor(vis2, cv2.COLOR_BGR2GRAY)[None]/ 255.
        return (ir1_128, vis2_128, ir1, vis2, vis2_color, name)

    def __len__(self):

        return len(self.names)