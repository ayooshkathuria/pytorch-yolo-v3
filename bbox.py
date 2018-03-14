from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np


def confidence_filter(result, confidence):

    conf_mask = (result[:,:,4] > confidence).float().unsqueeze(2)
    result = result*conf_mask    
    
    return result

def get_abs_coord(box):
    box[2], box[3] = abs(box[2]), abs(box[3])
    x1 = (box[0] - box[2]/2) - 1 
    y1 = (box[1] - box[3]/2) - 1 
    x2 = (box[0] + box[2]/2) - 1 
    y2 = (box[1] + box[3]/2) - 1
    return x1, y1, x2, y2
    


def sanity_fix(box):
    if (box[0] > box[2]):
        box[0], box[2] = box[2], box[0]
    
    if (box[1] >  box[3]):
        box[1], box[3] = box[3], box[1]
        
    return box

def bbox_iou(box1, box2, corner_coord = True):
    """
    Returns the IoU of two bounding boxes describe by a 
    
    (x,y,w,h) tuple, if corner_coord is False
    
    (x1, y1, x2, y2) representing corner of the box, if corner_coord is True
    
    """
    
    
    
    if not corner_coord:
        box1 = get_abs_coord(box1)
        
        box2 = get_abs_coord(box2)
    else:
        #sanity check so that the corner co-rdinates are
        #aligned towards the diagnal that lies in the 1st quadrant 
        #If the deal about quadrant puzzles you, ignore
        #Or maybe time to brush up some co-ordinate geometry.
        box1 = sanity_fix(box1)
        box2 = sanity_fix(box2)
        
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    
    inter_rect_x1 =  max(b1_x1, b2_x1)
    inter_rect_y1 =  max(b1_y1, b2_y1)
    inter_rect_x2 =  min(b1_x2, b2_x2)
    inter_rect_y2 =  min(b1_y2, b2_y2)
    
    inter_area = (inter_rect_x2 - inter_rect_x1 + 1)*(inter_rect_y2 - inter_rect_y1 + 1)
    
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    print(b1_area)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    print(b2_area)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou



