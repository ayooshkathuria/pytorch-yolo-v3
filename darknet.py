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
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
    
    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode = "duplicate")
        pooled_x = F.max_pool2d(padded_x, self.kernel_size, padding = self.padding)
        return pooled_x

class RouteLayer(nn.Module):
    def __init__(self, index, start, end):
        super(RouteLayer, self).__init__()
        self.start = start
        self.end = end
        
#        
class ReOrgLayer(nn.Module):
    def __init__(self, stride):
        super(ReOrgLayer, self).__init__()
        self.stride= stride
        
    def forward(self,x):
        assert(x.data.dim() == 4)
        B,C,H,W = x.data.shape
        ws = hs = self.stride
        assert(H % hs == 0),  "The stride " + self.stride + " is not a proper divisor of height " + H
        assert(W % ws == 0),  "The stride " + self.stride + " is not a proper divisor of height " + W
        x = x.view(B,C, H // hs, hs, W // ws, ws).transpose(-2,-3).contiguous()
        x = x.view(B,C, H // hs * W // ws, hs, ws).contiguous()
        x = x.view(B,C, H // hs * W // ws, hs*ws).transpose(-1,-2).contiguous()
        x = x.view(B, C, ws*hs, H // ws, W // ws).contiguous()
        x = x.view(B, C*ws*hs, H // ws, W // ws).contiguous()
        return x
        
        
        
    
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
                activn = nn.LeakyReLU(0.1)
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
            
            
        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            

            
            route = RouteLayer(index, start, end)
            module.add_module("route_{0}".format(index), route)
                
            
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                output_filters[index + start]
                filters= output_filters[index + start]
            
             
        #If it's a reorganisation layer (Identity mappings in ResNet)
        elif (x["type"] == "reorg"):
            stride = int(x["stride"])
            
            reorg = ReOrgLayer(stride)
            module.add_module("reorg_{0}".format(index), reorg)
            
            filters = filters*stride*stride


        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1
    
    return (inp_info, module_list, loss)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.input, self.module_list, self.loss = create_modules(self.blocks)
        self.route_cache = generate_route_cache()

        
        
    
    def forward(self, x):
        
    
    def get_blocks(self):
        return self.blocks
    
    def get_module_list(self):
        return self.module_list[1]
    




cfgfile = "cfg/yolo-voc.cfg"
dn = Darknet(cfgfile)
print(dn.get_module_list())




        
        
            
        
            
            
            
            
        

            
    
            
                
                
            
        
            
            
            
            
        
            
            
        
    
    



