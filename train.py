import torch
import os
import argparse
from darknet import *
#from cocoloader import transform_annotation
from util import *
from data_aug.data_aug import *
from preprocess import *
import numpy as np
import cv2
import pickle as pkl
import matplotlib.pyplot as plt
from bbox import bbox_iou, corner_to_center, center_to_corner
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def arg_parse():
    """
    Parse arguements to the detect module

    """


    parser = argparse.ArgumentParser(description='YOLO v3 Training Module')


    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--datacfg", dest = "datafile", help = "cfg file containing the configuration for the dataset",
                        type = str, default = "cfg/coco.data")
    return parser.parse_args()


args = arg_parse()

#Load the model
model = Darknet(args.cfgfile)
#model.load_weights(args.weightsfile)
#model = model.to(device)  ## Really? You're gonna train on the CPU?

# Load the config file
net_options =  model.net_info
# print(net_options)

##Parse the config file
batch = net_options['batch']
subdivisions = net_options['subdivisions']  #Irrelavant for our implementation simnce we laod the entire batch into our RAM
#In darknet, subdivisions would the number of examples loaded into the RAM at once (after being concatenated)
width = net_options['width']
height = net_options['height']
channels = net_options['channels']
momentum = net_options['momentum']
decay = net_options['decay']    #Penalty for regularisation
angle = net_options['angle']    #The angle with which you want to rotate images as a part of augmentation
saturation = net_options['saturation']     #saturation related augmentation
exposure = net_options['exposure']
hue = net_options['hue']
learning_rate = net_options['learning_rate']    #Initial learning rate
burn_in = net_options['burn_in']
#for the first n = burn_in steps, the learning rate used is ((steps/burn_in)**a)*learning_rate
#where a is a hyperparameter that must be chosen. In the official darknet YOLO implementation, a = 4
max_batches = net_options['max_batches']
policy = net_options['policy']
steps = net_options['steps']
scales = net_options['scales']

#    return ([num_boxes_per_dim*x**2 for x in detection_map_dims])
 
#for x in coco_loader:
#    x = transform_annotation(x)
#    
#    num_pred_boxes = get_num_pred_boxes(inp_dim, strides, anchor_nums)
#    
#    label_map = torch.FloatTensor(sum(num_pred_boxes), 5 + classes)
#    
#    
#    label_map = get_pred_box_cords(num_pred_boxes, label_map, 
#                                   strides, inp_dim, anchor_nums)
#    
#    boxes = x[1]
#    print(x[1].shape)
##    label_map = 
#    
#    
#    x = transforms(x[0], x[1])
#    im = draw_rect(x[0], x[1])
#    plt.imshow(im)	
#    plt.show()
#    assert False
#    i += 1
#    if i == 10:
#        break   

#transforms = Sequence([RandomHorizontalFlip()])

coco_loader = pkl.load(open("Coco_sample.pkl", "rb"))


strides = [32,16,8]
anchors = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],
           [156,198],  [373,326]]

anchors.reverse()
anchors = np.array(anchors)
inp_dim = 416
classes = 80
num_anchors = 9
anchor_nums = [3,3,3]

i = 0

transforms = Sequence([RandomHorizontalFlip(), RandomScaleTranslate(translate=0.05, scale=(0,0.3)), RandomRotate(10), RandomShear(), YoloResize(inp_dim)])



def get_pred_box_cords(num_pred_boxes, label_map, strides, inp_dim, anchors_nums):
    i = 0
    j = 0 
    
    for n, pred_boxes in enumerate(num_pred_boxes):
        unit = strides[n]
        corners = torch.arange(0, inp_dim, unit).to(device)
        offset = unit // 2
        grid = torch_meshgrid(corners, corners).view(-1,2)
        grid += offset
        grid = grid.repeat(1,anchors_nums[n]).view(anchors_nums[n]*grid.shape[0], -1)
        label_map[i:i+pred_boxes,[0,1]] = grid
        
        scale_anchors =  anchors[j: j + anchor_nums[n]]
        
        scale_anchors = torch.FloatTensor(scale_anchors).to(device)
        
        scale_anchors = scale_anchors.repeat(int(pred_boxes/anchor_nums[n]),1)
        
        label_map[i:i+pred_boxes,[2,3]] = scale_anchors
     
        
        
        i += pred_boxes
        j += anchor_nums[n]
    return label_map        





def get_num_pred_boxes(inp_dim, strides, anchor_nums):    
    detection_map_dims = [(inp_dim//stride) for stride in strides]
    return [anchor_nums[i]*detection_map_dims[i]**2 for i in range(len(detection_map_dims))]




def get_ground_truth_predictors(ground_truth, label_map):
    i = 0    #indexes the anchor boxes
    j = 0    
    
    center_cell_li = []
    
    total_boxes_per_gt = sum(anchor_nums)
    
    num_ground_truth_in_im = ground_truth.shape[0]
    
    
    
    inds = torch.LongTensor(num_ground_truth_in_im, total_boxes_per_gt).to(device)
    
    #n index the the detection maps
    for n, anchor in enumerate(anchor_nums):
        offset =  sum(num_pred_boxes[:n])

        scale_anchors = anchors[i: i + anchor]
        
        center_cells = (ground_truth[:,[0,1]]) / strides[n]
        
        center_cells = center_cells.long() 
        
        print(center_cells)
        a = offset + anchor_nums[n]*(inp_dim//strides[n]*center_cells[:,1] + center_cells[:,0])
        
        inds[:,sum(anchor_nums[:n])] = a
        
        for x in range(1, anchor_nums[n]):
            inds[:,sum(anchor_nums[:n]) + x] = a + x 
  

        i += anchor
        j += num_pred_boxes[n]
    
    
    candidate_boxes = label_map[inds.long()][:,:,:4]
    
    
    candidate_boxes = center_to_corner(candidate_boxes)

    ground_truth_boxes = center_to_corner(ground_truth.unsqueeze(0)).squeeze()[:,:4]

    candidate_boxes = candidate_boxes.transpose(1,2).contiguous()
    
    ground_truth_boxes = ground_truth_boxes.unsqueeze(2)
    
    candidate_ious = bbox_iou(candidate_boxes, ground_truth_boxes)

    pickle.dump(candidate_ious, open("candidate_ious.pkl", "wb"))
    
    prediction_boxes = candidate_ious.new(num_ground_truth_in_im,1)
    
    print(candidate_ious)
    print(inds)
    for i in range(num_ground_truth_in_im):
        #get the the row and the column of the highest IoU
        max_iou_ind = torch.argmax(candidate_ious)
        max_iou_row = max_iou_ind.int() / total_boxes_per_gt
        max_iou_col = max_iou_ind.int() % total_boxes_per_gt
        
        
        #get the index (in label map) of the box with maximum IoU
        max_iou_box = inds[max_iou_row, max_iou_col]
        
        #assign the bounding box to the appropriate gt
        prediction_boxes[max_iou_row] = max_iou_box
        
        #zero out all the IoUs for this box so it can't be reassigned to any other gt
        box_mask = (inds != max_iou_ind).float().view(-1,9)
        candidate_ious *= box_mask
        
        #zero out all the values of the row representing gt that just got assigned so that it 
        #doesn't participate in the process again
        candidate_ious[max_iou_row] *= 0

    return (prediction_boxes)
            
def get_ground_truth_map(ground_truth, label_map, ground_truth_predictors):
    
    #Set the objectness confidence of these boxes to 1
    predboxes = label_map[ground_truth_predictors]
    
    
    predboxes[:,4] = 1
    predboxes[:,:4] = ground_truth[:,:4]
    
    cls_inds = 5 + ground_truth[:,4].long()

    
    predboxes[torch.arange(ground_truth.shape[0]).long() , cls_inds] = 1    
    
    label_map[ground_truth_predictors] = predboxes
    return label_map
    






toyloader = DataLoader(toyset("data_aug/demo.jpeg", transform = transforms))

random.seed(2)
plt.rcParams["figure.figsize"] = (10,8)

#anchors = pkl.load(open("anchors.pkl", "rb"))

for x, ann in toyloader:
    x = x.squeeze().numpy()
    cls  = np.array([0,0,0,1])
    


    ann = ann.squeeze().numpy()

    print(ann.shape)
    x = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    num_pred_boxes = get_num_pred_boxes(inp_dim, strides, anchor_nums)
    
    label_map = torch.zeros(sum(num_pred_boxes), 5 + classes).to(device)
    
    for cord in ann[:,:4]:
        x = draw_rect(x, cord)
    plt.imshow(x)

    
    label_map = get_pred_box_cords(num_pred_boxes, label_map, 
                                   strides, inp_dim, anchor_nums)
    
    
    ground_truth = torch.FloatTensor(ann).to(device)
    
    
    ground_truth = corner_to_center(ground_truth.unsqueeze(0)).squeeze()
    

    ground_truth_predictors = get_ground_truth_predictors(ground_truth, label_map)
    
    ground_truth_predictors = ground_truth_predictors.long().squeeze()
    ground_truth_map = get_ground_truth_map(ground_truth, label_map, ground_truth_predictors)
    


        
    plt.imshow(x)
    plt.show()        



