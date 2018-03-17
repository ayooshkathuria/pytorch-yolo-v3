from __future__ import division

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
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl

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
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    
    
    

    
    
    return parser.parse_args()


if __name__ ==  '__main__':
    args = arg_parse()
    images = args.images
    cfg = args.cfg
    weightsfile = args.weightsfile
    batch_size = int(args.bs)
    batch_size = 4
    start = 0

    CUDA = torch.cuda.is_available()
    network_dim = (416,416)
    inp_dim = 416
    num_classes  = 20   #Will be updated in future to accomodate COCO

    #Set up the neural network
    print("Loading network.....")
    model = Darknet(cfg)
    model.load_weights(weightsfile)
    print("Network successfully loaded")
    
    model(get_test_input(inp_dim, CUDA))
    
    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()
    
    #Set the model in evaluation mode
    model.eval()
    
    read_dir = time.time()
    #Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
        
    load_batch = time.time()
    

    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    
    if CUDA:
        im_dim_list = im_dim_list.cuda()
    
    leftover = 0
    
    if (len(im_dim_list) % batch_size):
        leftover = 1
        
    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size, len(im_batches))]))  for i in range(num_batches)]        


    i = 0
    
    output = torch.FloatTensor(1, 8)
    write = False
#    model(get_test_input(inp_dim, CUDA))
    
    start_det_loop = time.time()
    for batch in im_batches:
        #load the image 
        start = time.time()
        if CUDA:
            batch = batch.cuda()
       
        prediction = model(Variable(batch, volatile = True))
        
        prediction = prediction.data 
        
        
        
        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        #flatten the prediction vector 
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        #get the boxes with object confidence > threshold
        #Convert the cordinates to absolute coordinates
        
        prediction = predict_transform(prediction, inp_dim, model.anchors, num_classes, 0.25, CUDA)
        
            
        if type(prediction) == int:
            i += 1
            continue
        
        #perform NMS on these boxes, and save the results 
        #I could have done NMS and saving seperately to have a better abstraction
        #But both these operations require looping, hence 
        #clubbing these ops in one loop instead of two. 
        #loops are slower than vectorised operations. 
        
        prediction = write_results(prediction, num_classes, nms = True, nms_conf = 0.4)
    
        end = time.time()
        
        print(end - start)
        
        prediction[:,0] += i*batch_size
        i += 1
        
          
        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))
    
    
    output_recast = time.time()
    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))
        
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())/inp_dim
    output[:,1:5] *= im_dim_list
    
    
    
    class_load = time.time()

    
    classes = load_classes('data/voc.names')
    
    colors = pkl.load(open("pallete", "rb"))
    
    
    
    draw = time.time()


    def write(x, batches, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        return img
    
            
    list(map(lambda x: write(x, im_batches, orig_ims), output))
      
    
    det_names = pd.Series(imlist).apply(lambda x: "det/det_{}".format(x.split("/")[-1]))
    
    list(map(cv2.imwrite, det_names, orig_ims))
    
    end = time.time()
    
    
    
    print("{:20s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:20s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:20s}: {:2.3f}".format("Det Loop", output_recast - start_det_loop))
    print("{:20s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:20s}: {:2.3f}".format("Class Loading", draw - class_load))    
    print("{:20s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:20s}: {:2.3f}".format("Average time_per_img", (end - start_det_loop)/len(imlist)))
    
    
    
    
    
    torch.cuda.empty_cache()
        
        
        
        
    
    
