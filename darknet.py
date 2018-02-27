
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt

def parse_cfg(cfgile):
    pass

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    return img_
        

#class darknet(nn.Module):
#    def __init__(self):
#        super(darknet, self).__init__()
#        self.conv_1 = nn.Conv2d(3,4,2)
#    
#    def forward(self, x):
#        return self.conv_1(x)
        
        


