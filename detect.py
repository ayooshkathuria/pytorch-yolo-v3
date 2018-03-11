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
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image
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
    
    CUDA = torch.cuda.is_available()
    network_dim = (416,416)
    
    
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
    
    
    for image in imlist:
        start = time.time()
        #load the image 
        inp_image = prep_image(image, network_dim)
        result = model(inp_image)
        
        #Apply offsets to the result predictions
        

        #get the boxes with object confidence > threshold
        
        
        #perform NMS on these boxes
        
        
        
        #Convert the cordinates to absolute coordinates 
        
        
        
        #plot them with openCV, and save the files as well
        
        
        
        
        
        end = time.time()
        print("Time taken for image {} : {}".format(image.split("/")[-1], str(end - start)))
        
        #generate the detection image 
        

        
        
        
        
        
        
        
        
    
    
