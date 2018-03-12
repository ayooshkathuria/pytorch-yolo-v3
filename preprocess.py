from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from util import count_parameters as count
from util import convert2cpu as cpu

        
def prep_image(img, network_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = cv2.imread(img)
    img = cv2.resize(img, network_dim) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_

def prep_batch(imlist, batch_size, network_dim):
    num_batches = len(imlist)//batch_size + 1
    im_batches = []
    for batch in range(num_batches):
        batchx = torch.zeros(1, 3, *network_dim)
        for img in range(batch_size):
            id = batch*batch_size + img
            try:
                image = imlist[id]
            except IndexError:
                break
            inp_image = prep_image(image, network_dim)
            if img == 0:
                batchx.copy_(inp_image.data)
            else:
                batchx = torch.cat((batchx, inp_image.data))
        im_batches.append(Variable(batchx))
    return im_batches