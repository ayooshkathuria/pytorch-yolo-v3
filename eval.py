"""
Evaluation script for PyTorch Darknet model.

Evaluates the overlap between ground truth test labels and predicted labels
on the same images.  If there is one box from the predictions of an image
that overlaps by a user specifified threshhold, the true positive count is
increased and Average Precision will be higher.

If running on CPU-only, this script may be slow.
"""

from darknet import Darknet, get_test_input
import torch
import os
import argparse
import random
from customloader import CustomDataset, YoloResize, YoloResizeTransform
from torch.utils.data import DataLoader
from data_aug.data_aug import Sequence
from util import write_results, de_letter_box
from live import prep_image
from bbox import center_to_corner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle as pkl


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
    parser.add_argument("--overlap", dest="overlap_thresh", 
                        help="Overlap threshhold", default=0.5)

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

    # For each image look for overlaps between ground truth and predictions
    for i in range(image_num):
        predictions = predictions_all[i]
        ground_truths = ground_truths_all[i]
        if len(predictions) == 0:
            print('Prediction for image is emtpy.')
            fp[i] = 1.
            continue
        confidence = predictions[:, 4]
        BB = predictions[:, :4]

        # Sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]

        BBGT = ground_truths[:, :4]
        nd = BB.shape[0]
        ngt = BBGT.shape[0]

        print('BB number', BB.shape[0])
        print('BBGT number', BBGT.shape[0], '\n')
        
        # Go down detections and ground truths and calc overlaps (IOUs)
        overlaps = []
        for d in range(nd):
            bb = BB[d]
            for gt in range(ngt):
                bbox1 = torch.tensor(BBGT[np.newaxis, gt, :], dtype=torch.float)
                bbox2 = torch.tensor(bb[np.newaxis, :].clone().detach(), dtype=torch.float)
                overlaps.append(bbox_iou(bbox1, bbox2))
        # Get best IOU prediction/gt match
        ovmax = np.max(np.array(overlaps))

        # Mark TPs and FPs
        if ovmax > ovthresh:
            tp[i] = 1.
            print('tp!')
        else:
            fp[i] = 1.

    # Compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_gts)
    # Avoid divide by zero
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = average_precision(rec, prec, use_07_metric=False)

    return rec, prec, ap

def center_to_corner_2d(boxes):
    boxes[:,0] = (boxes[:,0] - boxes[:,2]/2)
    boxes[:,1] = (boxes[:,1] - boxes[:,3]/2)
    boxes[:,2] = (boxes[:,2]/2 + boxes[:,0]) 
    boxes[:,3] = (boxes[:,3]/2 + boxes[:,1])
    
    return boxes

if __name__ == "__main__":
    args = arg_parse()
    overlap_thresh = float(args.overlap_thresh)

    # Instantiate a model
    model = Darknet(args.cfgfile, train=False)

    # Get model specs
    model_dim = int(model.net_info["height"])
    assert model_dim % 32 == 0 
    assert model_dim > 32
    num_classes = int(model.net_info["classes"])
    bbox_attrs = 5 + num_classes

    # Load weights PyTorch style
    model.load_state_dict(torch.load(args.weightsfile))
    model = model.to(device)  ## Really? You're gonna eval on the CPU? :)

    # Set to evaluation (don't accumulate gradients)
    # Make sure to call eval() method after loading weights
    model.eval()

    # Load test data and resize only
    transforms = Sequence([YoloResizeTransform(model_dim)])
    test_data = CustomDataset(root="data", ann_file="data/test.txt", 
                              det_transforms=transforms,
                              num_classes=num_classes)
    test_loader = DataLoader(test_data, batch_size=1)

    ground_truths_all = []
    predictions_all = []
    num_gts = 0

    # Make a directory for the image files with their bboxes
    img_file_sample = test_data.examples[0].rstrip()
    eval_output_dir = img_file_sample.replace(img_file_sample.split(os.sep)[-2], 'eval_output')
    eval_output_dir = eval_output_dir.replace(img_file_sample, '')
    os.makedirs(eval_output_dir, exist_ok=True)

#     for i, (img, target) in enumerate(test_loader):
    for i in range(len(test_data)):
        img_file = test_data.examples[i].rstrip()


        print(i)

        # Read image and prepare for input to network
        img, orig_im, orig_dim = prep_image(plt.imread(img_file), model_dim)
        orig_dim = torch.FloatTensor(orig_dim).repeat(1,2)

        # Read ground truth labels
        ground_truths_file = img_file.replace(img_file.split('.')[-1], 'txt')
        try:
            ground_truths = np.array(pd.read_csv(ground_truths_file,
                header=None, sep=' '))
        except pd.errors.EmptyDataError:
            continue
        ground_truths[:, 1:] = center_to_corner_2d(ground_truths[:, 1:] * np.asarray(orig_dim[0]))

        class_labels = ground_truths[:, 0]
        # x1y1x2y2
        ground_truths = ground_truths[:, 1:]
        num_gts += ground_truths.shape[0]
        
        img = img.to(device)
        with torch.no_grad():
            output = model(img)
        # output = center_to_corner(output)
        
        # output = output.unsqueeze(0).view(-1, bbox_attrs)

        # keep = np.unique(np.asarray(nms(output, scores=output[:, 5], overlap=0.3, top_k=10)[0]))
        # print('keep ', keep)
        # output = output[keep, :]

        # # Convert to numpy
        # output = np.asarray(output.squeeze(0))
        
        if output.shape[0] > 0:
            # To later plot bboxes
            img_ = plt.imread(img_file)
            # # Get x1y1x2y2, mask conf, class conf
            # # Remember original image is square (or should be)
            # # Note:  inp_dim is model input size, im_dim is actual image size
            # output[:, 0] *= np.asarray(im_dim[0]/inp_dim)
            # output[:, 1] *= np.asarray(im_dim[1]/inp_dim)
            # output[:, 2] *= np.asarray(im_dim[0]/inp_dim)
            # output[:, 3] *= np.asarray(im_dim[1]/inp_dim)

            output = write_results(output, 0.0, num_classes, model_dim, orig_dim.unsqueeze(0), nms=True, nms_conf=0.5)

            # Scale
            scaling_factor = torch.min(model_dim/orig_dim,1)[0].view(-1,1)
            orig_dim = orig_dim.repeat(output.size(0), 1)
            output[:,[1,3]] -= (model_dim - scaling_factor*orig_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (model_dim - scaling_factor*orig_dim[:,1].view(-1,1))/2
            output[:,1:5] /= scaling_factor

            outputs = []
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, orig_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, orig_dim[i,1])
                if output[i,7] > 0.8:
                    # Draw bounding boxes on test images
                    colors = pkl.load(open("pallete", "rb"))
                    c1 = tuple(final[[0,1]])
                    c2 = tuple(final[[2,3]])                    
                    cv2.rectangle(img_, c1, c2, color=colors[1], thickness=2)
                final = np.asarray(output[i, 1:6], dtype=np.int32)
                outputs.append(final)

            # write image with bboxes to a new folder
            plt.imsave(img_file.replace(img_file.split('.')[-1], '_out.jpg').replace(
                img_file.split(os.sep)[-2], 'eval_output'), img_)

            outputs = torch.tensor(outputs)

            ground_truths_all.append(ground_truths)
            predictions_all.append(outputs)

    prec, rec, aps = custom_eval(predictions_all, ground_truths_all, num_gts=num_gts, ovthresh=overlap_thresh)
    print('Precision ', prec, 'Recall ', rec, 'Average precision ', np.mean(aps), sep='\n')
