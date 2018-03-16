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
from PIL import Image, ImageDraw


        
def prep_image(img, network_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = cv2.imread(img)
    orig_im = cv2.resize(orig_im, network_dim) 
    img = orig_im.copy()
    dim = img.shape[1], img.shape[0]
    img = cv2.resize(img, network_dim) 
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)

    return img_, orig_im, dim

def prep_image_pil(img, network_dim):
    img = Image.open(img).convert('RGB')
    dim = img.size
    img = img.resize(network_dim)
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(*network_dim, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3,*network_dim)
    img = img.float().div(255.0)
    return (img, dim)

def inp_to_image(inp):
    inp = inp.cpu().squeeze()
    inp = inp*255
    try:
        inp = inp.data.numpy()
    except RuntimeError:
        inp = inp.numpy()
    inp = inp.transpose(1,2,0)

    inp = inp[:,:,::-1]
    return inp


def prep_batch(imlist, batch_size, network_dim):
    leftover = 0
    if (len(imlist) % batch_size):
        leftover = 1
    num_batches = len(imlist)//batch_size + leftover
    im_batches = []
    for batch in range(num_batches):
        batchx = torch.zeros(1, 3, *network_dim)
        for img in range(batch_size):
            id = batch*batch_size + img
            try:
                image = imlist[id]
            except IndexError:
                break
            inp_image = prep_image_pil(image, network_dim)
            if img == 0:
                batchx.copy_(inp_image.data)
            else:
                batchx = torch.cat((batchx, inp_image.data))
        im_batches.append(Variable(batchx, volatile = True))
    return im_batches

def prep_batch_alt(imlist, batch_size, network_dim):
    leftover = 0
    if (len(imlist) % batch_size):
        leftover = 1
    num_batches = len(imlist)//batch_size + leftover
    im_batches = []
    orig_ims = np.zeros((len(imlist), *network_dim, 3))
    im_dim_list = torch.FloatTensor(len(imlist), 2)
    
    for batch in range(num_batches):
        batchx = torch.zeros(batch_size, 3, *network_dim)
        for img in range(batch_size):
            id = batch*batch_size + img
            try:
                image = imlist[id]
            except IndexError:
                batchx = batchx[:img]        
                break

            inp_image, orig_im, im_dim = prep_image(image, network_dim)
            
            im_dim = torch.FloatTensor(im_dim)
            
            orig_ims[id] = orig_im
            
            batchx[img].copy_(inp_image.view_as(batchx[img]))
            
            im_dim_list[id].view_as(im_dim).copy_(im_dim)
        
        im_batches.append(Variable(batchx, volatile = True))
    return im_batches, np.asarray(orig_ims), im_dim_list.repeat(1,2)