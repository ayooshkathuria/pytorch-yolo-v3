from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, prep_batch, inp_to_image
from bbox import confidence_filter
import time


class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)
        
def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_



def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v2 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detectio upon",
                        default = "imgs", type = str)
    parser.add_argument("--cfg", dest = "cfg", help = "Configuration file to build the neural network",
                        default = "cfg/yolo-voc.cfg")
    parser.add_argument("--weightsfile", dest = "weightsfile", help = "Weights file for the network",
                        default = "yolo-voc.weights")
    
    

    
    
    return parser.parse_args()

#def predict_transform(prediction, inp_dim, anchors):
#    batch_size = prediction.size(0)
#    network_stride = 32
#    grid_size = inp_dim // network_stride
#    bbox_attrs = 5 + num_classes
#    num_anchors = len(anchors)
#    
#    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
#    prediction = prediction.transpose(1,2).contiguous()
#    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
#    
#    #Creating a new Tensor to store the transformed data
#    #Just avoiding inplace operations    
#    temp = torch.FloatTensor(prediction.shape)
#
#    if CUDA:
#        temp = temp.cuda()
#    
#    temp1 = temp[:,:,2]
#    cacher = torch.FloatTensor(temp1.shape).cuda()
#    cacher.copy_(temp1)
#    
#    #Sigmoid the  centre_X, centre_Y. and object confidencce
#    temp[:,:,0].copy_(torch.sigmoid(prediction[:,:,0]).data)
#    temp[:,:,1].copy_(torch.sigmoid(prediction[:,:,1]).data)
#    temp[:,:,4].copy_(torch.sigmoid(prediction[:,:,4]).data)
#    temp[:,:,2:4].copy_((prediction[:,:,2:4]).data)
#    
#    #Add the center offsets
#    grid_len = np.arange(grid_size)
#    a,b = np.meshgrid(grid_len, grid_len)
#    
#    x_offset = torch.FloatTensor(a).view(-1,1)
#    y_offset = torch.FloatTensor(b).view(-1,1)
#    
#    if CUDA:
#        x_offset = x_offset.cuda()
#        y_offset = y_offset.cuda()
#    
#    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
#    
#    temp[:,:,:2] += x_y_offset
#      
#    #log space transform height and the width
#    anchors = torch.FloatTensor(anchors)
#    
#    if CUDA:
#        anchors = anchors.cuda()
#    
#    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
#    temp[:,:,2:4] = torch.exp(temp[:,:,2:4])*anchors
#
#    #Softmax the class scores
#    temp[:,:,5: 5 + num_classes].copy_(nn.Softmax(-1)(prediction[:,:, 5 : 5 + num_classes]).data)
#
#    return temp

if __name__ ==  '__main__':
    parser = arg_parse()
    images = parser.images
    cfg = parser.cfg
    weightsfile = parser.weightsfile
    start = 0

    CUDA = torch.cuda.is_available()
    network_dim = (416,416)
    num_classes  = 20   #Will be updated in future to accomodate COCO

    #Set up the neural network
    print("Loading network.....")
    model = Darknet(cfg)
    model.load_weights(weightsfile)
    print("Network successfully loaded")
    
    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()
    
    #Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
        
    batch_size = 5
    im_batches = prep_batch(imlist, batch_size, network_dim)

    for batch in im_batches:
        #load the image 
        start = time.time()
        inp_dim = batch[0].size(2)
        if CUDA:
            batch = batch.cuda()
       
        pred = model(batch)

        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        
        prediction = predict_transform(pred, inp_dim, model.anchors, num_classes, CUDA)
        
        prediction = confidence_filter(prediction, 0.7)
        #flatten the prediction vector 
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        #get the boxes with object confidence > threshold
     

        #perform NMS on these boxes
        
        
        
        #Convert the cordinates to absolute coordinates 
        
        
        
        #plot them with openCV, and save the files as well
        

        end = time.time()

        
        
        #generate the detection image 
    
    batch_end_time = time.time()

torch.cuda.empty_cache()
        
        
        
        
    
    
