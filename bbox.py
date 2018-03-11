from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np


def confidence_filter(result, confidence):
    batch_size = result.size(0)
    result = result.view(-1,13*13).contiguous()
    result = result.view(-1, 5*13*13).transpose(0,1).contiguous()
    
    conf_mask = (torch.sigmoid(result[:, 4]) > confidence).view(-1,1).float()
    result_mask = result*conf_mask
    mask_index = torch.nonzero(result_mask[:,4].data)
    result_mask = result_mask[mask_index, :].squeeze()
    print (result_mask.data.shape)