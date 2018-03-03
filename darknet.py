from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')     #store the lines in a list
    lines = [x for x in lines if len(x) > 0] #get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']  
    
    block = {}
    blocks = []
    
    
    
    for line in lines:
        if line[0] == "[":               #This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1]
        else:
            key,value = line.split("=")
            block[key] = value
    blocks.append(block)
    print('\n'.join([repr(x) for x in blocks]))
    

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_

#class darknet(nn.Module):
#    def __init__(self):
#        super(darknet, self).__init__()
#        self.conv_1 = nn.Conv2d(3,3,2)
#    
#    def forward(self, x):
#        return self.conv_1(x)
        
        

##Code to test the function
cfgfile = 'cfg/yolo-voc.cfg'
parse_cfg(cfgfile)
