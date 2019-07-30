import torch 
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset
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
import cv2 
import os


inp_dim = 416
# Custom transform options
# custom_transforms = Sequence([ RandomHSV(), RandomHorizontalFlip(), RandomScaleTranslate(translate=0.05, scale=(0,0.3)), RandomRotate(10),  RandomShear(), YoloResize(inp_dim)])
custom_transforms = Sequence([ RandomHSV(), YoloResize(inp_dim)])

random.seed(0)

def transform_annotation(x):
    """Convert the annotation/target boxes to a format understood by
    dataset class"""
    if not x:
        return None
    boxes = np.array([a.rstrip().split(' ') for a in x], dtype='float32')
    
    # get the bounding boxes and convert them into proper format
    boxes = boxes[:, 1:]
    boxes = np.array(boxes)
    boxes = boxes.reshape(-1,4)
    
    category_ids = np.array(boxes[:,0]).reshape(-1,1)
    ground_truth = np.concatenate([boxes, category_ids], 1).reshape(-1,5)
  
    return ground_truth


class CustomDataset(Dataset):
    def __init__(self, root, num_classes, ann_file, det_transforms=None):
        """Note:  When using VoTT and exported to YOLO, 
        ann_file is a list to paths of images"""
        self.root = root
        
        self.examples = None
        with open(ann_file, 'r') as f:
            self.examples = f.readlines()
        self.det_transforms = det_transforms

        # The following, user needs to modify (TODO - create from args)
        self.inp_dim = 416
        self.strides = [32,16,8]
        self.anchor_nums = [3,3,3]
        self.num_classes = num_classes
        # self.anchors = [[10,14],  [23,27],  [37,58],  [81,82],  [135,169],  [344,319]]
        # self.anchors = [[116,78], [122,181], [243,38], [337,256], [486,458], [492,42], [605,162], [669,100], [1272,189]]
        self.anchors = [[151,94], [260,280], [346,178], [727,653], [742,388], [802,119], [1031,632], [1272,424], [1353,745]]
        
        self.anchors = np.array(self.anchors)[::-1]
        
        #Get the number of bounding boxes predicted PER each scale 
        self.num_pred_boxes = self.get_num_pred_boxes()
        
        self.box_strides = self.get_box_strides()
        self.debug_id = None

    def __len__(self):
#        return super().__len__()
        #return len(self.ids)
        return len(self.examples)
    
    def get_box_strides(self):
        box_strides = np.zeros((sum(self.num_pred_boxes),1))
        offset = 0
        for i,x in enumerate(self.num_pred_boxes):
            box_strides[offset : offset + x ] = self.strides[i]
            offset += x
        return box_strides
    
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
            grid = grid.repeat(self.anchor_nums[n], axis = 0)
            label_map[i:i+pred_boxes,[0,1]] = grid
            
            scale_anchors =  self.anchors[j: j + self.anchor_nums[n]]          
            scale_anchors = np.array(scale_anchors)
            
            num_boxes_in_scale = int(pred_boxes/self.anchor_nums[n])
            scale_anchors = scale_anchors.reshape(1,-1).repeat(num_boxes_in_scale, axis = 0).reshape(-1,2)
            
            label_map[i:i+pred_boxes,[2,3]] = scale_anchors
         
            i += pred_boxes
            j += self.anchor_nums[n]
        return label_map        

    def get_ground_truth_predictors(self, ground_truth, label_map, im=None):
        i = 0    #indexes the anchor boxes
        j = 0    
                
        total_boxes_per_gt = sum(self.anchor_nums)
        num_ground_truth_in_im = ground_truth.shape[0]
        inds = np.zeros((num_ground_truth_in_im, total_boxes_per_gt), dtype = np.int)
        
        #n index the the detection maps
        for n, anchor in enumerate(self.anchor_nums):
            offset =  sum(self.num_pred_boxes[:n])
            try:
                center_cells = (ground_truth[:,[0,1]]) // self.strides[n]
            except:
                print(ground_truth)
                assert False
            
            a = offset + self.anchor_nums[n]*(inp_dim//self.strides[n]*center_cells[:,1] + center_cells[:,0])
            inds[:,sum(self.anchor_nums[:n])] = a
            
            for x in range(1, self.anchor_nums[n]):
                inds[:,sum(self.anchor_nums[:n]) + x] = a + x 
      
            i += anchor
            j += self.num_pred_boxes[n]
        
        candidate_boxes = label_map[inds][:,:,:4]
        candidate_boxes = center_to_corner(candidate_boxes)
        ground_truth_boxes = center_to_corner(ground_truth.copy()[np.newaxis]).squeeze(0)[:,:4]
        candidate_boxes = candidate_boxes.transpose(0,2,1)
        ground_truth_boxes = ground_truth_boxes[:,:,np.newaxis]
        candidate_ious = bbox_iou(candidate_boxes, ground_truth_boxes, lib="numpy")
        prediction_boxes = np.zeros((num_ground_truth_in_im,1), dtype=np.int)

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
            box_mask = (inds != max_iou_ind).reshape(-1,len(self.anchors))
            candidate_ious *= box_mask
            
            #zero out all the values of the row representing gt that just got assigned so that it 
            #doesn't participate in the process again
            candidate_ious[max_iou_row] *= 0
        
        return (prediction_boxes)
    
    def get_no_obj_candidates(self, ground_truth, label_map, ground_truth_predictors):
                
        total_boxes_per_gt = sum(self.anchor_nums)
        num_ground_truth_in_im = ground_truth.shape[0]
        inds = np.zeros((num_ground_truth_in_im, total_boxes_per_gt), dtype = np.int)
        inds = np.arange(sum(self.num_pred_boxes)).astype(int)
        inds = inds[np.newaxis].repeat(num_ground_truth_in_im, axis = 0) 
        candidate_boxes = label_map[inds][:,:,:4]
        candidate_boxes = center_to_corner(candidate_boxes)
        ground_truth_boxes = center_to_corner(ground_truth.copy()[np.newaxis]).squeeze(0)[:,:4]
        candidate_boxes = candidate_boxes.transpose(0,2,1)
        ground_truth_boxes = ground_truth_boxes[:,:,np.newaxis]
        candidate_ious = bbox_iou(candidate_boxes, ground_truth_boxes, lib = "numpy")
        candidate_ious[:, ground_truth_predictors] = 1
        
        max_ious_per_box = np.max(candidate_ious, 0)
        no_obj_cands = (np.nonzero(max_ious_per_box < 0.5)[0].astype(int))
        
        return no_obj_cands
        
        
    def get_ground_truth_map(self, ground_truth, label_map, ground_truth_predictors, no_obj_cands):
    
        #Set the objectness confidence of these boxes to 1
        
        label_map[:,4] = -1
        predboxes = label_map[ground_truth_predictors]
        predboxes[:,4] = 1
        label_map[no_obj_cands] = 0
        
        assert ground_truth_predictors.shape[0] == predboxes.shape[0], print(self.debug_id)

        ground_truth_strides = self.box_strides[ground_truth_predictors]
        ground_truth[:,:4] /= ground_truth_strides

        try:
            predboxes[:,[0,1]] = ground_truth[:,[0,1]] - predboxes[:,[0,1]]
        
        except:
            print(self.debug_id)
            assert False

        if 0 in predboxes[:,[0,1]]:
            predboxes[:,[0,1]] += 0.0001*(predboxes[:,[0,1]] == 0)
        predboxes[:,[0,1]] = -1*np.log(1/(predboxes[:,[0,1]]) - 1)
        
        mask = np.logical_and(ground_truth[:,2], ground_truth[:,3])
        mask= mask.reshape(-1,1)
        ground_truth *= mask 
        
        nz_inds = np.nonzero(ground_truth[:,0])
        ground_truth = ground_truth[nz_inds]
        predboxes = predboxes[nz_inds]
        ground_truth_predictors = ground_truth_predictors[nz_inds]
        
        try:
            predboxes[:,[2,3]] = np.log(ground_truth[:,[2,3]] / predboxes[:,[2,3]])
        except:
            print(self.debug_id)
            assert False
        
        predboxes[:,5] = ground_truth[:,4]
        label_map[ground_truth_predictors] = predboxes
        
        return label_map

    def __getitem__(self, idx):
        """Get image and ground truth bounding boxes.

        The image is converted to PyTorch tensor.
        The bounding boxes are converted to x_c, y_c, w, h (the VoTT YOLO default format)
        """
        example = self.examples[idx]

        path = os.path.join(os.getcwd(), example).rstrip()
        image = cv2.imread(path)[:,:,::-1]   #Load the image from opencv and convert to RGB

        label_table = np.zeros((sum(self.num_pred_boxes), 6), dtype=np.float)
        label_table = self.get_pred_box_cords(label_table)
                
        #seperate images, boxes and class_ids
        ground_truth = None
        with open(example.replace(example.split('.')[-1], 'txt')) as f:
            ground_truth = transform_annotation(f.readlines())

        self.debug_id = example
        #apply the augmentations to the image and the bounding boxes
        if self.det_transforms:
            image, ground_truth = self.det_transforms(image, ground_truth)
        im = image.copy()

        #Convert the cv2 image into a PyTorch tensor
        image = image.transpose(2,0,1)/255.0
        image = torch.Tensor(image)
            
        #  ground_truth = corner_to_center(ground_truth[np.newaxis,:,:]).squeeze().reshape(-1,5)
            
        if len(ground_truth) > 0 and ground_truth.shape[0] > 0:
            ground_truth = ground_truth[np.newaxis,:,:].squeeze().reshape(-1,5)
            #Generate a table of labels
            #Get the bounding boxes to be assigned to the ground truth
            ground_truth_predictors = self.get_ground_truth_predictors(ground_truth, label_table)
            
            no_obj_cands = self.get_no_obj_candidates(ground_truth, label_table, ground_truth_predictors)

            ground_truth_predictors = ground_truth_predictors.squeeze(1)

            label_table[:,:2] //= self.box_strides 
            label_table[:,[2,3]] /= self.box_strides

            ground_truth_map = self.get_ground_truth_map(ground_truth, label_table, ground_truth_predictors, no_obj_cands)

            ground_truth_map = torch.Tensor(ground_truth_map)
        else:
            ground_truth_map = torch.zeros((0, 6), dtype=torch.float)

        return image, ground_truth_map

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels
        """

        images = list()
        boxes = list()

        for b in batch:
            if len(b[1]) > 0:
                images.append(b[0])
                boxes.append(b[1])
        
        if len(boxes) > 0:
            boxes = torch.stack(boxes, dim=0)
            images = torch.stack(images, dim=0)
        
        return images, boxes
