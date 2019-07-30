"""
Process video from file or from camera using trained model
"""
from __future__ import division
import torch 
import torch.nn as nn
from util import de_letter_box, write_results, load_classes
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image

import time
import numpy as np
import cv2
import pandas as pd
import random 
import pickle as pkl
import argparse

# Choose backend device for tensor operations - GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def arg_parse():
    """Parse arguements to the detect module"""
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest='video', 
                        help="Video to run detection upon", type=str)
    parser.add_argument("--source", dest='source', 
                        help="Video source used by OpenCV VideoCapture", 
                        type=int, default=0)
    parser.add_argument("--confidence", dest="confidence", 
                        help="Object confidence to filter predictions", default=0.6)
    parser.add_argument("--nms_thresh", dest="nms_thresh", 
                        help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', 
                        help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', 
                        help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--datacfg", dest="datafile", 
                        help="Config file containing the configuration for the dataset",
                        type=str, default="cfg/coco.data")
    parser.add_argument("--reso", dest='reso', 
                        help="Input resolution of the network. Increase to increase \
                            accuracy. Decrease to increase speed",
                        default="416", type=str)
    return parser.parse_args()

def get_test_input(input_dim):
    """A single test image"""
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    
    img_ = img_.to(device)
    
    return img_

def prep_image(img, model_dim):
    """
    Prepare image for input to the neural network. 
    """
    orig_im = img
    orig_dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (model_dim, model_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, orig_dim

def write(x, img):
    """
    Arguments
    ---------
    x : array of float
        [batch_index, x1, y1, x2, y2, objectness, label, probability]
    img : numpy array
        original image

    Returns
    -------
    img : numpy array
        Image with bounding box drawn
    """

    if x[-1] is not None:
        c1 = tuple(x[[4,1]].int())
        c2 = tuple(x[[2,3]].int())
        # Is this an empty box?
        if c1[0] == c2[0] or c1[1] == c2[1]:
            return img
        try:
            label = int(x[-2])
        except ValueError:
            return img
        try:
            label = "{0}".format(classes[label])
        except IndexError:
            return img
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

def remove_empty_boxes(output):
    """In processed output prediction, if a box is emtpy (height
    or width is zero), then it is removed from the final output array"""
    ary = []
    output_ = output.clone()
    for i in range(output_.size(0)):
        inner_ary = output_[i].numpy()
        if inner_ary[0] == inner_ary[2] and inner_ary[1] == inner_ary[3]:
            continue
        else:
            ary.append(inner_ary)
    return torch.tensor(ary, dtype=torch.int32).view(-1, 7)

if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
        
    print("Loading network.....")
    model = Darknet(cfgfile=args.cfgfile, train=False)
    model.load_state_dict(torch.load(args.weightsfile))
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    model_dim = int(model.net_info["height"])
    assert model_dim % 32 == 0 
    assert model_dim > 32
    num_classes = int(model.net_info["classes"])
    bbox_attrs = 5 + num_classes

    model = model.to(device)
    model.eval()
    
    if args.video: # video file
        videofile = args.video
        cap = cv2.VideoCapture(videofile)
    else:
        # On mac, 0 is bulit-in camera and 1 is USB webcam on Mac
        # On linux, 0 is video0, 1 is video1 and so on
        cap = cv2.VideoCapture(args.source)
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            
            img, orig_im, orig_dim = prep_image(frame, model_dim)
            orig_dim = torch.FloatTensor(orig_dim).repeat(1,2)

            with torch.no_grad():
                output = model(img)

            # output is, after write_results, [batch index, x1, y1, x2, y2, objectness score, class index, class prob]
            output = write_results(output, confidence, num_classes, model_dim, orig_dim, nms=True, nms_conf=nms_thesh)

            # If no preds, just show image and go to next pred
            if type(output) == int or output.shape[0] == 0:
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                continue

            # output = output.float()

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
                final = np.asarray(output[i, 0:6])
                outputs.append(final)

            classes = load_classes('data/obj.names')
            colors = pkl.load(open("pallete", "rb"))
            
            list(map(lambda x: write(x, orig_im), output))
            
            cv2.imshow("frame", orig_im)
            cv2.imwrite('detection.png', orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            
        else:
            break
