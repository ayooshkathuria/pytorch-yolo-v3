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
from bbox import corner_to_center, center_to_corner, bbox_iou
inp_dim = 416


transforms = Sequence([RandomHorizontalFlip(), RandomScaleTranslate(translate=0.05, scale=(0,0.3)), RandomRotate(10), RandomShear(), YoloResize(inp_dim)])
transforms = Sequence([])

random.seed(0)

#from kmeans.kmeans import *

def transform_annotation(x):
    #convert the PIL image to a numpy array
    image = np.array(x[0])
    
    #get the bounding boxes and convert them into proper format 
    boxes = [a["bbox"] for a in x[1]]
    
    boxes = np.array(boxes)
    
    boxes = boxes.reshape(-1,4)
    
    boxes[:,2] += boxes[:,0]
    boxes[:,3] += boxes[:,1]

    
    category_ids = np.array([a["category_id"] for a in x[1]]).reshape(-1,1)
    
    ground_truth = np.concatenate([boxes, category_ids], 1)
    
    
    
    return image, ground_truth


class CocoDataset(CocoDetection):
    def __init__(self, root = None, annFile = None, det_transforms = None):
#        super().__init__(root, annFile, None, None)
        self.root = pkl.load(open("Coco_sample.pkl", "rb"))

        self.annFile = None
        self.det_tranforms = det_transforms
        self.inp_dim = 416
        self.strides = [32,16,8]
        self.anchor_nums = [3,3,3]
        self.num_classes = 80
        
        self.anchors = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],
           [156,198],  [373,326]]
        
        self.anchors = np.array(self.anchors)[::-1]
        
        #Get the number of bounding boxes predicted PER each scale 
        self.num_pred_boxes = self.get_num_pred_boxes()

    
    def __len__(self):
#        return super().__len__()
        return len(self.root)
    
    def set_inp_dim(self, inp_dim):
        self.inp_dim = inp_dim
        
    def get_num_pred_boxes(self):    
        detection_map_dims = [(self.inp_dim//stride) for stride in self.strides]
        return [self.anchor_nums[i]*detection_map_dims[i]**2 for i in range(len(detection_map_dims))]
    
    def get_pred_box_cords(self, label_map):
        i = 0
        j = 0 
        
        for n, pred_boxes in enumerate(self.num_pred_boxes):
            unit = self.strides[n]
            corners = np.arange(0, inp_dim, unit)
            offset = unit // 2
            grid = np.meshgrid(corners, corners)
            grid = np.concatenate((grid[0][:,:,np.newaxis], grid[1][:,:,np.newaxis]), 2).reshape(-1,2)
            grid += offset
            grid = grid.repeat(self.anchor_nums[n], axis = 1).reshape(self.anchor_nums[n]*grid.shape[0], -1)
            label_map[i:i+pred_boxes,[0,1]] = grid
            
            scale_anchors =  self.anchors[j: j + self.anchor_nums[n]]
            
            scale_anchors = np.array(scale_anchors)
            
            scale_anchors = scale_anchors.repeat(int(pred_boxes/self.anchor_nums[n]), axis = 0)
            
            label_map[i:i+pred_boxes,[2,3]] = scale_anchors
         
            
            
            i += pred_boxes
            j += self.anchor_nums[n]
        return label_map        

    def get_ground_truth_predictors(self, ground_truth, label_map):
        i = 0    #indexes the anchor boxes
        j = 0    
                
        total_boxes_per_gt = sum(self.anchor_nums)
        
        num_ground_truth_in_im = ground_truth.shape[0]
        
        
        
        inds = np.zeros((num_ground_truth_in_im, total_boxes_per_gt), dtype = np.int)
        
        #n index the the detection maps
        for n, anchor in enumerate(self.anchor_nums):
            offset =  sum(self.num_pred_boxes[:n])
                
            center_cells = (ground_truth[:,[0,1]]) // self.strides[n]
            
            center_cells = center_cells 
            
            a = offset + self.anchor_nums[n]*(inp_dim//self.strides[n]*center_cells[:,1] + center_cells[:,0])
            
            inds[:,sum(self.anchor_nums[:n])] = a
            
            for x in range(1, self.anchor_nums[n]):
                inds[:,sum(self.anchor_nums[:n]) + x] = a + x 
      
    
            i += anchor
            j += self.num_pred_boxes[n]
        
        
        candidate_boxes = label_map[inds][:,:,:4]
        
        
        candidate_boxes = center_to_corner(candidate_boxes)
    
        ground_truth_boxes = center_to_corner(ground_truth.copy()[np.newaxis]).squeeze()[:,:4]
        
        
    
        candidate_boxes = candidate_boxes.transpose(0,2,1)
        
        ground_truth_boxes = ground_truth_boxes[:,:,np.newaxis]
        
        candidate_ious = bbox_iou(candidate_boxes, ground_truth_boxes, lib = "numpy")
        
        prediction_boxes = np.zeros((num_ground_truth_in_im,1), dtype = np.int)

        for i in range(num_ground_truth_in_im):
            #get the the row and the column of the highest IoU
            max_iou_ind = np.argmax(candidate_ious)
            max_iou_row = max_iou_ind // total_boxes_per_gt
            max_iou_col = max_iou_ind % total_boxes_per_gt
            
            
            #get the index (in label map) of the box with maximum IoU
            max_iou_box = inds[max_iou_row, max_iou_col]
            
            #assign the bounding box to the appropriate gt
            prediction_boxes[max_iou_row] = max_iou_box
            
            #zero out all the IoUs for this box so it can't be reassigned to any other gt
            box_mask = (inds != max_iou_ind).reshape(-1,9)
            candidate_ious *= box_mask
            
            #zero out all the values of the row representing gt that just got assigned so that it 
            #doesn't participate in the process again
            candidate_ious[max_iou_row] *= 0
    
        return (prediction_boxes)
    
    def __getitem__(self, idx):
#        return super().__getitem__(idx)
        
         example = self.root[idx]
         
         #Transform the annotation to a manageable format
         example = transform_annotation(example)
         
         #seperate images, boxes and class_ids
         image, ground_truth = example
         
         #apply the augmentations to the image and the bounding boxes
         image, ground_truth = transforms(image, ground_truth)
         
         #Convert the cv2 image into a PyTorch tensor
#         image = torch.Tensor(image)
         

         for i,cord in enumerate(ground_truth[:,:4]):
             if i in [6,7]:    
                 print(cord)
                 image = draw_rect(image, cord)
                 
                 
         plt.imshow(image)
         #Convert the box notation from x1,y1,x2,y2 ---> cx, cy, w, h
         ground_truth = corner_to_center(ground_truth[np.newaxis,:,:]).squeeze()


         
         #Generate a table of labels
         label_table = np.zeros((sum(self.num_pred_boxes), 5 + self.num_classes), dtype = np.float)
         
         label_table = self.get_pred_box_cords(label_table)
         
         
         #Get the bounding boxes to be assigned to the ground truth
         ground_truth_predictors = self.get_ground_truth_predictors(ground_truth, label_table)
         
         
         print(ground_truth_predictors)
         
         

         
         
                 
         
         assert False
        
        
    
coco = CocoDataset()
#
coco_loader = DataLoader(coco)

for x in coco_loader:
    print(x)    
    assert False    

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
    
        


