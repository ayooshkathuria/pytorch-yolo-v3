from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import de_letter_box, write_results, load_classes
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
from bbox import center_to_corner, bbox_iou, corner_to_center_2d

import pandas as pd
import random 
import pickle as pkl
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a torch tensor and dimensions 
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def arg_parse():
    """Parse arguements to the detect module"""
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest='video', help = 
                        "Video to run detection upon", type = str)
    # parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained", default="pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default = 0.6)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest='cfgfile', help = 
                        "Config file",
                        default="cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest='weightsfile', help = 
                        "weightsfile",
                        default="yolov3.weights", type = str)
    parser.add_argument("--datacfg", dest="datafile", help="cfg file containing the configuration for the dataset",
                        type = str, default="cfg/coco.data")
    parser.add_argument("--reso", dest='reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type = str)
    return parser.parse_args()


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
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    num_classes = int(model.net_info["classes"])
    bbox_attrs = 5 + num_classes

    model = model.to(device)
    model.eval()
    
    if args.video: # video file
        videofile = args.video
        cap = cv2.VideoCapture(videofile)
    else: # webcam
        cap = cv2.VideoCapture(0) # TODO: webcam video source options
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:
            img, orig_im, orig_im_dim = prep_image(frame, inp_dim)
            orig_im_dim = torch.FloatTensor(orig_im_dim).to(device)                    
            
            img = img.to(device)
            
            with torch.no_grad():   
                output = model(img)
            # output = center_to_corner(output)
            
            # output = output.unsqueeze(0).view(-1, bbox_attrs)
            # output = np.asarray(output.squeeze(0))

            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

            # Classes for labels and colors for bbox
            classes = load_classes(args.datafile)
            colors = pkl.load(open("pallete", "rb"))

            if output.shape[0] > 0:
                # Get x1y1x2y2
                # Remember original image is square (or should be)
                output[:, 1] *= np.asarray(orig_im_dim[0])/inp_dim
                output[:, 2] *= np.asarray(orig_im_dim[1])/inp_dim
                output[:, 3] *= np.asarray(orig_im_dim[0])/inp_dim
                output[:, 4] *= np.asarray(orig_im_dim[1])/inp_dim

                for i in range(output.shape[0]):
                    score = output[i, 6]
                    if score >= confidence:
                        img = np.asarray(img)
                        # print(output[i, :])
                        # needs to be top, left
                        c1 = tuple((output[i, 0:2]))
                        # needs to be bottom, right
                        c2 = tuple((output[i, 2:4]))
                        label = int(output[i, 7])
                        label="{0}".format(classes[label])
                        color = random.choice(colors)
                        # need top-left corner and bottom-right corner of rectangle to draw
                        cv2.rectangle(orig_im, c1, c2, 3, 1)
                        # cv2.imshow("frame", orig_im)
                        # cv2.imwrite('detection.png', orig_im)
                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
                        c2 = c1[0], c1[1]
                        cv2.rectangle(img, c1, c2, color, -1)
                        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
                        cv2.imwrite('detection.png', img)
            
                cv2.imshow("frame", orig_im)
                cv2.imwrite('detection.png', orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

        else:
            break
