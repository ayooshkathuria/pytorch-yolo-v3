from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image, jaset
import pandas as pd
import random 
import pickle as pkl
import itertools
from torch.utils.data import DataLoader


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
        
def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_



def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()

if __name__ ==  '__main__':
    args = arg_parse()
    
    scales = args.scales
    
    
#        scales = [int(x) for x in scales.split(',')]
#        
#        
#        
#        args.reso = int(args.reso)
#        
#        num_boxes = [args.reso//32, args.reso//16, args.reso//8]    
#        scale_indices = [3*(x**2) for x in num_boxes]
#        scale_indices = list(itertools.accumulate(scale_indices, lambda x,y : x+y))
#    
#        
#        li = []
#        i = 0
#        for scale in scale_indices:        
#            li.extend(list(range(i, scale))) 
#            i = scale
#        
#        scale_indices = li

    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes('data/coco.names') 

    #Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()
    
    
    #Set the model in evaluation mode
    model.eval()
    
    read_dir = time.time()

    write = False
    colors = pkl.load(open("pallete", "rb"))
    
    load_batch = time.time()
    
    test = jaset("imgs")
    
    read_dir = time.time()
    
    imlist = test.imlist()
    
    batch_size = 4
    imloader = DataLoader(test, batch_size, num_workers = 0)

    
    start_det_loop = time.time()
    
    for ind, batch, dim in imloader:
        #load the image 
        start = time.time()
        if CUDA:
            batch = batch.cuda()
        

        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        #flatten the prediction vector 
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)
        
#        prediction = prediction[:,scale_indices]
  
        #get the boxes with object confidence > threshold
        #Convert the cordinates to absolute coordinates
        #perform NMS on these boxes, and save the results 
        #I could have done NMS and saving seperately to have a better abstraction
        #But both these operations require looping, hence 
        #clubbing these ops in one loop instead of two. 
        #loops are slower than vectorised operations. 
        
        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        
        if type(prediction) == int:
            continue

        end = time.time()

            
        batch_imlist = [imlist[ind] for ind in [int(a) for a in ind]]

    
        for im_num, image in enumerate(batch_imlist):
            objs = [classes[int(x[-1])] for x in prediction if int(x[0]) == im_num]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")
        

        
        if CUDA:
            torch.cuda.synchronize()
            
        prediction = de_letter_box(prediction, dim, inp_dim)
        
#            im_dim_list = torch.stack(dim, 1)
#            
#            im_dim_list = torch.index_select(im_dim_list, 0, prediction[:,0].long())
#            
#            im_dim_list = im_dim_list.float()
#            scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
#        
#        
#            prediction[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
#            prediction[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
#            
#            prediction[:,1:5] /= scaling_factor
#            
#            for i in range(prediction.shape[0]):
#                prediction[i, [1,3]] = torch.clamp(prediction[i, [1,3]], 0.0, im_dim_list[i,0])
#                prediction[i, [2,4]] = torch.clamp(prediction[i, [2,4]], 0.0, im_dim_list[i,1])
        
        
        orig_ims = [cv2.imread(x) for x in batch_imlist]
        
        
        list(map(lambda x: writer(x, orig_ims, classes, colors), prediction))
        
        det_names = pd.Series(batch_imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))
    
        list(map(cv2.imwrite, det_names, orig_ims))
        
        
        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))
            
            
    detection_fin = time.time() 
    try:
        output
    except NameError:
        print("No detections were made")
        exit()
    
    end = time.time()
    
    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", read_dir - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", detection_fin - start_det_loop))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (detection_fin - load_batch)/len(imlist)))
    print("----------------------------------------------------------")

    
    torch.cuda.empty_cache()
    
    
        
        
    
    
