"""
Training script for PyTorch Darknet model.

e.g. python train.py --cfg cfg/yolov3-tiny-1xclass.cfg --weights yolov3-tiny.weights --datacfg data/obj.data

Output prediction vector is [batch, centre_x, centre_y, box_height, box_width, mask_confidence, class_confidence]
"""

import torch
from torchvision import transforms
import os
import argparse
from darknet import Darknet, parse_cfg
from util import *
from data_aug.data_aug import Sequence
from preprocess import *
import numpy as np
import cv2
import pickle as pkl
import matplotlib.pyplot as plt
from bbox import bbox_iou, corner_to_center, center_to_corner
import pickle 
from customloader import custom_transforms, CustomDataset
import torch.optim as optim
import torch.autograd.gradcheck
from tensorboardX import SummaryWriter
import sys
import datetime


# Folder to save checkpoints to
SAVE_FOLDER = datetime.datetime.now().strftime("%B-%d-%Y-%I:%M%p")
os.makedirs(os.path.join(os.getcwd(), 'runs', SAVE_FOLDER))

# For tensorboard
writer = SummaryWriter()
random.seed(0)

# Choose backend device for tensor operations - GPU or CPU
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
                        type = str, default = "data/obj.data")
    parser.add_argument("--lr", dest = "lr", type = float, default = 0.001)
    parser.add_argument("--mom", dest = "mom", type = float, default = 0)
    parser.add_argument("--wd", dest = "wd", type = float, default = 0)
    parser.add_argument("--unfreeze", dest = "unfreeze", type = int, default = 4,
                        help="Last number of layers to unfreeze for training")

    return parser.parse_args()

args = arg_parse()

#Load the model
model = Darknet(args.cfgfile, train=True)
model.load_weights(args.weightsfile)

# Unfreeze all but this number of layers at the beginning
layers_length = len(list(model.parameters()))
FINE_TUNE_STOP_LAYER = int(layers_length/1.3)

# Use CUDA device if availalbe and set to train
model = model.to(device)
model.train()

# Load the config file
net_options =  model.net_info

##Parse the config file
batch = net_options['batch']
angle = net_options['angle']    #The angle with which you want to rotate images as a part of augmentation

# For RandomHSV() augmentation (see data_aug.py)
saturation = int(float(net_options['saturation'])*255)    #saturation related augmentation
exposure = int(float(net_options['exposure'])*255)
hue = int(float(net_options['hue'])*179)

steps = int(net_options['steps'])
# scales = net_options['scales']
num_classes = net_options['classes']
bs = net_options['batch']
# Assume h == w
inp_dim = net_options['height']

# Assign from the command line args
lr = args.lr
wd = args.wd
momentum = args.mom
momentum = 0.9
wd = 0.0005

inp_dim = int(inp_dim)
num_classes = int(num_classes)
bs = int(bs)

def logloss(pred, target):
    assert pred.shape == target.shape, "Input and target must be the same shape"
    pred = pred.view(-1,1)
    target = target.view(-1,1)
    
    sigmoid = torch.nn.Sigmoid()(pred)    
    sigmoid = sigmoid.repeat(1,2)
    sigmoid[:,0] = 1 - sigmoid[:,0]
    sigmoid = sigmoid[torch.arange(sigmoid.shape[0]).long(), target.squeeze().long()]
    loss = -1*torch.log(sigmoid)
    loss[torch.nonzero(target).long()] *= 5
    
    if int(torch.isnan(loss).any()):
        pred_ = pred.detach().cpu().numpy()
        target_ = target.detach().cpu().numpy()
        sigmoid_ = sigmoid.detach().cpu().numpy()
        pkl.dump((pred_, target_, sigmoid_), open("nan_loss", "wb"))
        
        print("Nan Value Detected in the loss")
        assert False
        
    if (loss == float("inf")).any() or (loss == float("-inf")).any():
        print("Infinity encountered in loss")
        
        pred_ = pred.detach().cpu().numpy()
        target_ = target.detach().cpu().numpy()
        sigmoid_ = sigmoid.detach().cpu().numpy()
        pkl.dump((pred_, target_, sigmoid_), open("inf_loss", "wb"))
        
    loss = torch.sum(loss) / loss.shape[0]
    
    return loss
    
def YOLO_loss(ground_truth, output):
    """Function to calculate loss based on predictions
    and ground-truth labels"""
    total_loss = 0
    
    #get the objectness loss
    loss_inds = torch.nonzero(ground_truth[:,:,-4] > -1)
    objectness_pred = output[loss_inds[:,0],loss_inds[:,1],4]
    target = ground_truth[loss_inds[:,0],loss_inds[:,1],4]
    objectness_loss = torch.nn.MSELoss(size_average=False)(objectness_pred, target)
    #Only objectness loss is counted for all boxes
    object_box_inds = torch.nonzero(ground_truth[:,:,4] > 0).view(-1, 2)

    try:
        gt_ob = ground_truth[object_box_inds[:,0], object_box_inds[:,1]]
    except IndexError:
        return None
    
    pred_ob = output[object_box_inds[:,0], object_box_inds[:,1]]
    
    #get centre x and centre y 
    centre_x_loss = torch.nn.MSELoss(size_average=False)(pred_ob[:,0], gt_ob[:,0])
    centre_y_loss = torch.nn.MSELoss(size_average=False)(pred_ob[:,1], gt_ob[:,1])

    total_loss += centre_x_loss 
    total_loss += centre_y_loss 
    
    #get w,h loss
    w_loss = torch.nn.MSELoss(size_average=False)(pred_ob[:,2], gt_ob[:,2])
    h_loss = torch.nn.MSELoss(size_average=False)(pred_ob[:,3], gt_ob[:,3])
    
    total_loss += w_loss 
    total_loss += h_loss 

    #class_loss 
    # cls_scores_pred = pred_ob[:,5:]
    # cls_scores_target = gt_ob[:,5].long() 
    # cls_scores_pred = torch.log(torch.sigmoid(cls_scores_pred))
    # print(cls_scores_pred.shape)
    # print(cls_scores_target.shape)
    # cls_loss = torch.nn.NLLLoss(size_average=False, reduce = False)(cls_scores_pred, cls_scores_target)
    cls_labels = torch.zeros(gt_ob.shape[0], num_classes).to(device)
    cls_labels[torch.arange(gt_ob.shape[0]).long(), gt_ob[:,5].long()] = 1
    cls_loss = 0    

    for c_n in range(num_classes):
        targ_labels = pred_ob[:,5 + c_n].view(-1,1)
        targ_labels = targ_labels.repeat(1,2)
        targ_labels[:,0] = 1 - targ_labels[:,0]
        cls_loss += torch.nn.CrossEntropyLoss(size_average=False)(targ_labels, cls_labels[:,c_n].long())

    total_loss += cls_loss

    return total_loss

### DATA ###

# Overloading custom data transforms from customloader (may add more here)
# custom_transforms = Sequence([RandomHSV(hue=hue, saturation=saturation, brightness=exposure), 
#     YoloResizeTransform(inp_dim)])
custom_transforms = Sequence([YoloResizeTransform(inp_dim)])

# Data instance and loader
data = CustomDataset(root="data", num_classes=num_classes, 
                     ann_file="data/train.txt", 
                     det_transforms=custom_transforms)
print('Batch size ', bs)
data_loader = DataLoader(data, 
                         batch_size=bs,
                         shuffle=True,
                         collate_fn=data.collate_fn)

### TRAIN MODEL ###

# Use this optimizer calculation for training loss
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

itern = 0
epochs = int(len(data) / bs)
lr_update_step = epochs - 1

# "unfreeze" refers to the last number of layers to tune (allows gradients to be tracked)
p_i = 1
print("Number of layers in network = {}".format(layers_length))
stop_layer = layers_length - args.unfreeze # Unfreeze all but this number of layers at the beginning

for p in model.parameters():
    if p_i < stop_layer:
        p.requires_grad = False
    else:
        p.requires_grad = True
    p_i += 1

for step in range(steps):
    for image, ground_truth in data_loader:

        if len(ground_truth) == 0:
            continue

        # # Track gradients in backprop
        image = image.to(device)
        ground_truth = ground_truth.to(device)
        
        output = model(image)

        # Clear gradients from optimizer for next iteration
        optimizer.zero_grad()

        print("\n\n")
        print('Iteration ', itern)

        if (torch.isnan(ground_truth).any()):
            print("Nans in Ground_truth")
            assert False
            
        if (torch.isnan(output).any()):
            print("Nans in Output")
            assert False
            
        if (ground_truth == float("inf")).any() or (ground_truth == float("-inf")).any():
            print("Inf in ground truth")
            assert False
            
        if (output == float("inf")).any() or (output == float("-inf")).any():
            print("Inf in output")
            assert False

        loss  = YOLO_loss(ground_truth, output)

        if loss:
            print("Loss for iter no: {0}: {1:.4f}".format(itern, float(loss)/bs))
            writer.add_scalar("Loss/vanilla", float(loss), itern)
            loss.backward()
            optimizer.step()

        print('lr: ', optimizer.param_groups[0]["lr"])

        itern += 1

    # Update LR for next epoch
    for param_group in optimizer.param_groups:
        if itern >= lr_update_step:
            lr_update_step += itern
            optimizer.param_groups[0]["lr"] = (lr*pow((itern / lr_update_step),4))
            print('lr updated: ', optimizer.param_groups[0]["lr"])
        

# Save intermediate model in pytorch format (the state dictionary only, i.e. parameters only)
torch.save(model.state_dict(), os.path.join('runs', SAVE_FOLDER, 
    'epoch{0}-intermediate-bs{1}-loss{2:.4f}.pth'.format(itern, bs, float(loss)/bs))) 

### DATA FOR FINE-TUNING ###

# Reset data loader
# Data instance with transforms (augmentations) and PyTorch loader
data = CustomDataset(root="data", num_classes=num_classes, 
                     ann_file="data/train.txt", 
                     det_transforms=custom_transforms)
data_loader = DataLoader(data, batch_size=bs,
                         shuffle=True,
                         collate_fn=data.collate_fn)

### FINE TUNE MODEL ON MORE LAYERS ###

# "unfreeze" refers to the last number of layers to tune (allow gradients to be tracked)
p_i = 1

# Unfreeze more layers (tensors holding weights) to fine-tune on network
for p in model.parameters():
    if p_i < FINE_TUNE_STOP_LAYER:
        p.requires_grad = False
    else:
        p.requires_grad = True
    p_i += 1

# New iteration counter and make sure LR is set correctly
# itern_fine = 0
lr_update_step = epochs + itern - 1
lr = optimizer.param_groups[0]["lr"] / 10
# lr_updated = False
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

for step in range(steps):
    for image, ground_truth in data_loader:
        if len(ground_truth) == 0:
            continue

        # # Track gradients in backprop
        image = image.to(device)
        ground_truth = ground_truth.to(device)
        
        output = model(image)

        # Clear gradients from optimizer for next iteration
        optimizer.zero_grad()

        print("\n\n")
        print('Iteration ', itern)

        if (torch.isnan(ground_truth).any()):
            print("Nans in Ground_truth")
            assert False
            
        if (torch.isnan(output).any()):
            print("Nans in Output")
            assert False
            
        if (ground_truth == float("inf")).any() or (ground_truth == float("-inf")).any():
            print("Inf in ground truth")
            assert False
            
        if (output == float("inf")).any() or (output == float("-inf")).any():
            print("Inf in output")
            assert False

        loss  = YOLO_loss(ground_truth, output)
        
        if loss:
            print("Loss for iter no: {0}: {1:.4f}".format(itern, float(loss)/bs))
            writer.add_scalar("Loss/vanilla", float(loss), itern)
            if itern % 10 == 0:
                torch.save(model.state_dict(), os.path.join('runs', SAVE_FOLDER,
                     'epoch{0}-bs{1}-loss{2:.4f}.pth'.format(itern, bs, float(loss)/bs)))
            loss.backward()
            optimizer.step()
 
        print('lr: ', optimizer.param_groups[0]["lr"])

        # itern_fine += 1
        itern += 1

    # Update LR for next epoch
    for param_group in optimizer.param_groups:
        if itern >= lr_update_step:
            lr_update_step += itern
            optimizer.param_groups[0]["lr"] = (lr*pow((itern / lr_update_step),4))
            print('lr updated: ', optimizer.param_groups[0]["lr"])

writer.close()

# Save final model in pytorch format (the state dictionary only, i.e. parameters only)
torch.save(model.state_dict(), os.path.join('runs', SAVE_FOLDER, 
    'epoch{0}-final-bs{1}-loss{2:.4f}.pth'.format(itern, bs, float(loss)/bs)))    
    
    



