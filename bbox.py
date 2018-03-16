from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import load_classes


def confidence_filter(result, confidence):
    conf_mask = (result[:,:,4] > confidence).float().unsqueeze(2)
    result = result*conf_mask    
    
    return result

def confidence_filter_cls(result, confidence):
    max_scores = torch.max(result[:,:,5:25], 2)[0]
    res = torch.cat((result, max_scores),2)
    print(res.shape)
    
    
    cond_1 = (res[:,:,4] > confidence).float()
    cond_2 = (res[:,:,25] > 0.995).float()
    
    conf = cond_1 + cond_2
    conf = torch.clamp(conf, 0.0, 1.0)
    conf = conf.unsqueeze(2)
    result = result*conf   
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

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = (inter_rect_x2 - inter_rect_x1 + 1)*(inter_rect_y2 - inter_rect_y1 + 1)
    
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou


def pred_corner_coord(prediction):
    #Get indices of non-zero confidence bboxes
    ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    
    box = prediction[ind_nz[0], ind_nz[1]]
    
    
    box_a = box.new(box.shape)
    box_a[:,0] = (box[:,0] - box[:,2]/2)
    box_a[:,1] = (box[:,1] - box[:,3]/2)
    box_a[:,2] = (box[:,0] + box[:,2]/2) 
    box_a[:,3] = (box[:,1] + box[:,3]/2)
    box[:,:4] = box_a[:,:4]
    
    prediction[ind_nz[0], ind_nz[1]] = box
    
    return prediction

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def write_results(prediction, num_classes, nms = True, nms_conf = 0.4):
    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False
    
    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]

        
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        image_pred_ = image_pred[non_zero_ind.squeeze(),:]
        
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])
        
        #Get the class names
        classes = load_classes('data/voc.names')
                
        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-1]).squeeze()

            image_pred_class = image_pred_[class_mask_ind]

        
             #sort the detections such that the entry with the maximum objectness
             #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind]
                    
                    
            
            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output
