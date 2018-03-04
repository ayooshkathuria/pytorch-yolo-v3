from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt

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
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_


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
    return blocks
#    print('\n\n'.join([repr(x) for x in blocks]))
    

class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(nn.Module).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
    
    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode = "duplicate")
        pooled_x = F.max_pool2d(padded_x, self.kernel_size, padding = self.padding)
        return pooled_x
    
def create_modules(blocks):
    inp_info = blocks[0]     #Captures the information about the input and pre-processing
    modules = blocks[1:-1]   #The layers of the neural network
    loss = blocks[-1]        # Loss function 
    
    module_list = []
    
    index = 0    #indexing blocks helps with implementing route  layers (skip connections)

    
    prev_filters = 3
    
    output_filters = []
    
    for x in modules:
        module = nn.Sequential()
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
                
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
            
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU()
                module.add_module("leaky_{0}".format(index), activn)
            
            
            
        
        elif (x["type"] == "maxpool"):  #if it is a max pooling layer
        
        #Both YOLO f/ PASCAL and COCO don't use 2X2 pooling with stride 1
        #Tiny-YOLO does use it 
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            
            if stride > 1:
                pool = nn.MaxPool2d(kernel_size, stride)
            
            else:
                pool = MaxPoolStride1(kernel_size)

            module.add_module("pool_{0}".format(index), pool)
            
            
        
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            print(x["layers"])
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            

            
            try:
                route = RouteLayer(start, end)
                module.add_module("route_{0}".format(index), route)

            except:
                
                print("To be implemented")
            
            if end < 0:
                print("Hello")
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                output_filters[index + start]
                filters= output_filters[index + start]
            
             
        
        elif (x["type"] == "reorg"):
            stride = int(x["stride"])
            
            try:    
                reorg = ReorgLayer(stride)
                module.add_module("reorg_{0}".format(index), reorg)

            except:
                print("To be implemented")
            
            filters = filters*stride*stride


        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1
    
    print (list((enumerate(output_filters))))
    return module_list

class darknet(nn.Module):
    def __init__(self, cfgfile):
        super(darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.module_list = create_modules(self.blocks)
        
        
        
    def forward(self, x):
        return self.blocks
    
    def get_blocks(self):
        return self.blocks
    
    def get_module_list(self):
        return self.module_list
    


cfgfile = "cfg/yolo-voc.cfg"
dn = darknet(cfgfile)
blocks = dn.get_blocks()
module_list = dn.get_module_list()

w = 0
i = 0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

for x in module_list:
    w += count_parameters(x)
    print (i, w)
    i += 1


        
        
            
        
            
            
            
            
        

            
    
            
                
                
            
        
            
            
            
            
        
            
            
        
    
    



