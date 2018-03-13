from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np


def confidence_filter(result, confidence):

    conf_mask = (result[:,:,4] > confidence).float().unsqueeze(2)
    result = result*conf_mask    
    
    return result


