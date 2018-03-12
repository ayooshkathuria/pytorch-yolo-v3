from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np


def confidence_filter(result, confidence, num_classes, inp_dim):
    batch_size = result.size(0)
    network_stride = 32
    grid_size = inp_dim // network_stride
    bbox_attrs = 5 + num_classes
    
    
    
    result = result.view(batch_size, -1, grid_size*grid_size)
    result = result.view(batch_size, bbox_attrs, -1)
    
    result = result.transpose(1,2).contiguous()
    conf_mask = (torch.sigmoid(result[:,:,4]) > confidence).float().unsqueeze(2)
    result = result*conf_mask
    return result
