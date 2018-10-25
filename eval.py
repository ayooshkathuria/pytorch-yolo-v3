from darknet import Darknet, get_test_input
import torch
import os
import argparse
import random
from customloader import CustomDataset, YoloResize
from torch.utils.data import DataLoader
from data_aug.data_aug import Sequence
# from bbox import bbox_iou
from util import write_results, de_letter_box
from live import prep_image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep, 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

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

def average_precision(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def custom_eval(predictions_all,
             ground_truths_all,
             num_gts,
             ovthresh=0.5):
    """
    [ovthresh]: Overlap threshold (default = 0.5)
    """

    image_num = len(ground_truths_all)
    tp = np.zeros(image_num)
    fp = np.zeros(image_num)
    for i in range(image_num):

        predictions = predictions_all[i]
        ground_truths = ground_truths_all[i]
        # Predictions
        confidence = predictions[:, 4]
        BB = predictions[:, :4]

        # Sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]

        BBGT = ground_truths[:, :4]
        nd = BB.shape[0]
        ngt = BBGT.shape[0]

        # print(BB)
        # print(BBGT)
        
        # Go down detections and ground truths and calc overlaps (IOUs)
        overlaps = []
        for d in range(nd):
            bb = BB[d]
            for gt in range(ngt):
                bbox1 = torch.tensor(BBGT[np.newaxis, gt, :], dtype=torch.float)
                bbox2 = torch.tensor(bb[np.newaxis, :], dtype=torch.float)
                overlaps.append(bbox_iou(bbox1, bbox2))
        ovmax = np.max(np.array(overlaps))

        # Mark TPs and FPs
        if ovmax > ovthresh:
            tp[i] = 1.
        else:
            fp[i] = 1.

    # Compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_gts)
    # Avoid divide by zero
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = average_precision(rec, prec)

    return rec, prec, ap

def corner_to_center_1d(box):
    box[0] = (box[0] + box[2])/2
    
    box[1] = (box[1] + box[3])/2
    box[2] = 2*(box[2] - box[0])
    box[3] = 2*(box[3] - box[1])

    return box

def center_to_corner_2d(boxes):
    boxes[:,0] = (boxes[:,0] - boxes[:,2]/2)
    boxes[:,1] = (boxes[:,1] - boxes[:,3]/2)
    boxes[:,2] = (boxes[:,2] + boxes[:,0]) 
    boxes[:,3] = (boxes[:,3] + boxes[:,1])
    
    return boxes

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

    model = model.to(device)  ## Really? You're gonna eval on the CPU? :)

    # Load test data and resize only
    transforms = Sequence([YoloResize(inp_dim)])
    test_data = CustomDataset(root="data", ann_file="data/test.txt", det_transforms=transforms)
    test_loader = DataLoader(test_data, batch_size=1)

    ground_truths_all = []
    predictions_all = []
    num_gts = 0

    # for i, (img, target) in enumerate(test_loader):
    for i in range(len(test_data)):
        img_file = test_data.examples[i]
        print(i)

        # Read image and prepare for input to network
        img, orig_im, orig_im_dim = prep_image(plt.imread(img_file.rstrip()), inp_dim)
        im_dim = torch.FloatTensor(orig_im_dim).to(device)

        # Read ground truth labels
        ground_truths = np.array(pd.read_csv(img_file.replace(img_file.split('.')[-1], 'txt'),
            header=None, sep=' '))
        ground_truths[:, 1:] = center_to_corner_2d(ground_truths[:, 1:] * orig_im_dim[0])
        class_labels = ground_truths[:, 0]
        # x1y1x2y2
        ground_truths = ground_truths[:, 1:]
        num_gts += ground_truths.shape[0]

    #     # img = image[np.newaxis, :, :, :]


        img = img.to(device)
        output = model(img)
        # output = write_results(output, 0.7, num_classes, nms=True)
        # output = np.asarray(output)
        output = output.unsqueeze(0).view(2535, 6)
        keep = np.unique(np.asarray(nms(output, scores=output[:, 5], overlap=0.8)[0]))
        print(keep)
        output = output[keep, :]
        print(output[0])
        pred_class = output[:, 0]
        print(pred_class)

        
    #     if type(output) != int:
    #         output = de_letter_box(output, im_dim, inp_dim)
    #         # Get x1y1x2y2, mask conf, class conf
        output = output[:, 1:7]
    #         # Remember original image is square (or should be)
        # output[:,0:4] = (output[:,0:4] / inp_dim) * orig_im_dim[0]
        # print(pred_class, output)

        ground_truths_all.append(ground_truths)
        predictions_all.append(output)

    prec, rec, aps = custom_eval(predictions_all, ground_truths_all, num_gts=num_gts, ovthresh=0.2)
    print(prec, rec, np.mean(aps))
