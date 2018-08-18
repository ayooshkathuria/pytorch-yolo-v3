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
import cv2 
import os


inp_dim = 416


transforms = Sequence([ RandomHSV(), RandomHorizontalFlip(), RandomScaleTranslate(translate=0.05, scale=(0,0.3)), RandomRotate(10),  RandomShear(), YoloResize(inp_dim)])
transforms = Sequence([YoloResize(inp_dim)])

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
    
    
    ground_truth = np.concatenate([boxes, category_ids], 1).reshape(-1,5)
    
    
    
    return image, ground_truth


class CocoDataset(CocoDetection):
    def __init__(self, root = None, annFile = None, det_transforms = None):
#        super().__init__(root, annFile, None, None)
        self.root = root
#        self.ids = list(self.coco.imgs.keys())
        #self.root = root 
        #self.annFile = pkl.load(open(".pkl", "rb"))
        #self.ids = list(self.annFile.keys())

        self.annFile = pkl.load(open(annFile, "rb"))
        
        self.examples = list(self.annFile.items())
        self.det_transforms = det_transforms
        self.inp_dim = 416
        self.strides = [32,16,8]
        self.anchor_nums = [3,3,3]
        self.num_classes = 80
        
        self.anchors = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],
           [156,198],  [373,326]]
        
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

    def get_ground_truth_predictors(self, ground_truth, label_map, im = None):
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
        
        candidate_ious = bbox_iou(candidate_boxes, ground_truth_boxes, lib = "numpy")
        
        
        
        prediction_boxes = np.zeros((num_ground_truth_in_im,1), dtype = np.int)

#        print(inds)
#        print(candidate_ious)
#        print(label_map[[252,253,254]])
#        assert False
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
        
        
#        print(predboxes)
        return label_map
    

    
    
    def __getitem__(self, idx):
         example = self.examples[idx]
         
         
         path = os.path.join(self.root, "000000040962.jpg")
         image = cv2.imread(path)[:,:,::-1]   #Load the image from opencv and convert to RGB
         
         plt.imshow(image)
         
         im  = image.copy()
         

         
         
         #seperate images, boxes and class_ids
         ground_truth = example[1]
         


         self.debug_id = example[0]
         #apply the augmentations to the image and the bounding boxes
         if self.det_transforms:
             image, ground_truth = self.det_transforms(image, ground_truth)
             
        
         im = image.copy()
#         
         #Convert the cv2 image into a PyTorch 
         image = image.transpose(2,0,1)/255.0
         image = torch.Tensor(image)
         
         label_table = np.zeros((sum(self.num_pred_boxes), 6), dtype = np.float)
         
         label_table = self.get_pred_box_cords(label_table)

         ground_truth = corner_to_center(ground_truth[np.newaxis,:,:]).squeeze().reshape(-1,5)
         
         
         if ground_truth.shape[0] > 0:
             

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
            ground_truth_map = torch.Tensor(label_table)
    
         
         
         return image, ground_truth_map
         

         
         
                 
         
        
##        
#####    
#coco = CocoDataset(root = "COCO/train2017", annFile="COCO_ann_mod.pkl", det_transforms = transforms)
###
#coco_loader = DataLoader(coco, batch_size = 5)
#
#for x in coco_loader:
#    print(x[0].shape, x[1].shape)    
#    assert False
#    pass

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
    
        


