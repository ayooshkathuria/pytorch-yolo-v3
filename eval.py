from darknet import Darknet, get_test_input
import torch
import os
import argparse
import random
from customloader import CustomDataset, YoloResize
from torch.utils.data import DataLoader
from data_aug.data_aug import Sequence
from util import bbox_iou
import numpy as np

random.seed(0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def arg_parse():
    """
    Parse arguments to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Evaluation Module')

    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolov3.weights", type = str)

    return parser.parse_args()


def score_pred(ground_truth, output):
    
    total_loss = 0
    
    #get the objectness loss
    loss_inds = torch.nonzero(ground_truth[:,:,-4] > -1)
    objectness_pred = output[loss_inds[:,0],loss_inds[:,1],4]
    target = ground_truth[loss_inds[:,0],loss_inds[:,1],4]
    
    #Only objectness loss is counted for all boxes
    object_box_inds = torch.nonzero(ground_truth[:,:,4] > 0).view(-1, 2)
    
    try:
        gt_ob = ground_truth[object_box_inds[:,0], object_box_inds[:,1]]
    except IndexError:
        return None
    
    pred_ob = output[object_box_inds[:,0], object_box_inds[:,1]]

    cls_scores_pred = pred_ob[:,5:]
    cls_scores_target = gt_ob[:,5].long()
    
    #get centre x and centre y 
    centre_x = pred_ob[:,0]
    centre_y = pred_ob[:,1]

    print('gt_ob ', gt_ob[:, :4])
    print('pred_ob ', pred_ob[:, :4])

    return cls_scores_pred

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 

    Input boxes are expected to be in x1y1x2y2 format.
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
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

if __name__ == "__main__":
    args = arg_parse()

    # Instantiate a model
    model = Darknet(args.cfgfile, train=False)

    # Get model specs
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    num_classes = int(model.net_info["classes"])
    bbox_attrs = 5 + num_classes

    # Load weights PyTorch style
    model.load_state_dict(torch.load(args.weightsfile))

    # Set to evaluation (don't accumulate gradients)
    model.eval()

    model = model.to(device)  ## Really? You're gonna eval on the CPU?

    # Load test data
    transforms = Sequence([YoloResize(inp_dim)])
    test_data = CustomDataset(root="data", ann_file="data/test.txt", det_transforms=transforms)
    data_loader = DataLoader(test_data, batch_size=1)

    mAP = []
    scores = []

    for image, ground_truth in data_loader:
        with torch.no_grad():
            image = image.to(device)
            ground_truth = ground_truth.to(device)
            output = model(image)
            score = output.unsqueeze(dim=1)[-1][-1][-1][5]
        scores.append(float(score.detach().cpu().numpy()))
    
    print('average class scores: {}'.format(np.mean(scores)))
        

    



