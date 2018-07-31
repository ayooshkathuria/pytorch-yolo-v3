import torch 
from torchvision.datasets import CocoDetection
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from data_aug.bbox_util import draw_rect
from data_aug.data_aug import *
import time
import random
from torch.utils.data import DataLoader

#from kmeans.kmeans import *

class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, det_transforms = None):
        super().__init__(root, annFile, None, None)
        self.det_tranforms = det_transforms
        self.inp_dim = 416
    
    def __len__(self):
        return super().__len__()
    
    def set_inp_dim(self, inp_dim):
        self.inp_dim = inp_dim
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
        
    


def tiny_coco(cocoloader, num):
    i = 0
    li = []
    
    for x in cocoloader:
        print(i)
        
        li.append(x)
        i += 1
        if i > num - 1:
            break
    num = num / 1000
    pkl.dump(li, open("COCO_{}k.pkl.format(num)", "wb"))


def get_num_boxes(cocoloader):
    num_boxes = 0
    for x in cocoloader:
        x = trasform_annotation(x)
        bboxes = x[1]
        bbox_dims_h = bboxes[:,3] - bboxes[:,1]
        bbox_dims_w = bboxes[:,2] - bboxes[:,0]
        
    
        bbox_dims = np.stack((bbox_dims_w, bbox_dims_h)).T
        
        num_boxes += bbox_dims.shape[0]
    
    return num_boxes



def transform_annotation(x):
    #convert the PIL image to a numpy array
    image = np.array(x[0])
    
    #get the bounding boxes and convert them into proper format 
    boxes = [a["bbox"] for a in x[1]]
    
    boxes = np.array(boxes)
    
    boxes = boxes.reshape(-1,4)
    
    boxes[:,2] += boxes[:,0]
    boxes[:,3] += boxes[:,1]

    
    category_ids = [a["category_id"] for a in x[1]]
    
    return image, boxes, category_ids

    



def pickle_coco_dims(cocoloader):
    li = []
    
    
    i = 0
    for x in cocoloader:
        x = trasform_annotation(x)
        bboxes = x[1]
        bbox_dims_h = bboxes[:,3] - bboxes[:,1]
        bbox_dims_w = bboxes[:,2] - bboxes[:,0]
        
    
        bbox_dims = np.stack((bbox_dims_w, bbox_dims_h)).T
        
        li.append(bbox_dims)        
        
        print('Image {} of {}'.format(i, len(cocoloader)))
        
        i += 1
        
    li = np.vstack(li)
    pkl.dump(li, open("Entire_dims.pkl", "wb"))



def get_coco_sample(cocoloader):
    li = []
    i = 0
    for x in cocoloader:    
        if i == 9:
            break
        i+= 1
        li.append(x)
    pkl.dump(li, open("Coco_sample.pkl", "wb"))
    
        


