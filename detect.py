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
from bbox import confidence_filter, pred_abs_coord
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
        
    model.eval()
    
    #Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
        
    batch_size = 2
    im_batches = prep_batch(imlist, batch_size, network_dim)
    i = 0

    for batch in im_batches:
        #load the image 
        start = time.time()
        inp_dim = batch[0].size(2)
        if CUDA:
            batch = batch.cuda()
       
        prediction = model(batch)
        
        prediction = prediction.data
        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        
        prediction = predict_transform(prediction, inp_dim, model.anchors, num_classes, CUDA)
        

        #flatten the prediction vector 
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        #get the boxes with object confidence > threshold
        
        prediction = confidence_filter(prediction, 0.5)

        #Convert the cordinates to absolute coordinates 
        prediction = pred_abs_coord(prediction)
        a = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
        print(prediction[a[0], a[1]])
        assert False

        #perform NMS on these boxes
        
        
        
        
        
        
        
        
        
        #plot them with openCV, and save the files as well
        

        end = time.time()

        
        
        #generate the detection image 
    
    batch_end_time = time.time()


torch.cuda.empty_cache()
        
        
        
        
    
    
