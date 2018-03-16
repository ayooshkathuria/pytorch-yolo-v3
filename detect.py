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
from preprocess import prep_image, prep_batch, inp_to_image, prep_batch_alt
from bbox import confidence_filter, pred_corner_coord, bbox_iou, write_results
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
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    
    
    

    
    
    return parser.parse_args()

def draw_detections(output, im_batches):
    batches = torch.cat(im_batches).data
    
    
    inds = output[:,0].long()
    
    
    return output, batches

if __name__ ==  '__main__':
    parser = arg_parse()
    images = parser.images
    cfg = parser.cfg
    weightsfile = parser.weightsfile
    batch_size = parser.bs
    batch_size = 10
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
    
    #Set the model in evaluation mode
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
        
    im_batches, orig_ims, im_dim_list = prep_batch_alt(imlist, batch_size, network_dim)
    i = 0
    
    output = torch.FloatTensor(1, 8)
    write = False
    model(get_test_input().cuda())
    start = time.time()
    
    inp_dim = 416

    for batch in im_batches:
        #load the image 
        start = time.time()
        if CUDA:
            batch = batch.cuda()
       
        prediction = model(batch)
        
        prediction = prediction.data
        
#        b = batch[5]
#        cv2.imwrite()
        
        
        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        
        prediction__ = predict_transform(prediction, 416, model.anchors, num_classes, CUDA)
        

        #flatten the prediction vector 
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        #get the boxes with object confidence > threshold
        
        
        
        prediction_ = confidence_filter(prediction__, 0.5)
        

        #Convert the cordinates to absolute coordinates
        prediction = pred_corner_coord(prediction_)    

#        
        
        #perform NMS on these boxes, and save the results 
        #I could have done NMS and saving seperately to have a better abstraction
        #But both these operations require looping, hence 
        #clubbing these ops in one loop instead of two. 
        #loops are slower than vectorised operations. 
        
        prediction = write_results(prediction, num_classes, nms = True, nms_conf = 0.4)
        
        prediction[:,0] += i*batch_size
        i += 1
        
            
        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))

        
#    im_dim_list = torch.index_select(im_dim_list.cuda(), 0, output[:,0].long())/inp_dim
#    output[:,1:5] *= im_dim_list
    
    output, im_batches = draw_detections(output.cpu(), im_batches)
    results = (orig_ims)
    
    
    end = time.time()

classes = load_classes('data/voc.names')


def write(x, batches, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "Class Conf: {0:.2f} Object conf: {0:.2f}".format(classes[cls], x[5], x[4])
    color =[0,0,255]
    cv2.rectangle(img, c1, c2,color, 1)
    cv2.putText(img, label, (c1), cv2.FONT_HERSHEY_PLAIN, 1, color, 1);
#    cv2.imwrite("det/fg_{}.png".format(round(time.time(),4)), img)
    return img


def imwrite(x):
    i = 0
    
    while(True):
        print(i)
        cv2.imwrite("im_{}.png".format(i), x)
        i += 1    
        yield i
        
list(map(lambda x: write(x, im_batches, results), output))

for x in results:
    cv2.imwrite("im_{}.jpg".format(i), x)
    i += 1
    


print ((end - start)/len(imlist))
print(1/((end - start)/len(imlist)))

torch.cuda.empty_cache()
        
        
        
        
    
    
