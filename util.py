
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA):
    batch_size = prediction.size(0)
    network_stride = 32
    grid_size = inp_dim // network_stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    #Creating a new Tensor to store the transformed data
    #Just avoiding inplace operations    
    temp = torch.FloatTensor(prediction.shape)

    if CUDA:
        temp = temp.cuda()
    
    temp1 = temp[:,:,2]
    cacher = torch.FloatTensor(temp1.shape).cuda()
    cacher.copy_(temp1)
    
    #Sigmoid the  centre_X, centre_Y. and object confidencce
    temp[:,:,0].copy_(torch.sigmoid(prediction[:,:,0]).data)
    temp[:,:,1].copy_(torch.sigmoid(prediction[:,:,1]).data)
    temp[:,:,4].copy_(torch.sigmoid(prediction[:,:,4]).data)
    temp[:,:,2:4].copy_((prediction[:,:,2:4]).data)
    
    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    temp[:,:,:2] += x_y_offset
      
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    temp[:,:,2:4] = torch.exp(temp[:,:,2:4])*anchors

    #Softmax the class scores
    temp[:,:,5: 5 + num_classes].copy_(nn.Softmax(-1)(prediction[:,:, 5 : 5 + num_classes]).data)

    return temp

