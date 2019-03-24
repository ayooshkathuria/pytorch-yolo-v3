"""
Training script for PyTorch Darknet model.

e.g. python train.py --cfg cfg/yolov3-tiny-1xclass.cfg --weights yolov3-tiny.weights --datacfg data/obj.data

Output prediction vector is [centre_x, centre_y, box_height, box_width, mask_confidence, class_confidence]

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


writer = SummaryWriter()

random.seed(0)

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
                        type = str, default = "cfg/data.data")
    parser.add_argument("--lr", dest = "lr", type = float, default = 0.001)
    parser.add_argument("--mom", dest = "mom", type = float, default = 0)
    parser.add_argument("--wd", dest = "wd", type = float, default = 0)
    parser.add_argument("--unfreeze", dest = "unfreeze", type = int, default = 4,
                        help="Last number of layers to unfreeze for training")

    return parser.parse_args()

args = arg_parse()

#Load the model
model = Darknet(args.cfgfile, train=True)

# "unfreeze" refers to the last number of layers to tune (allow gradients to be tracked)
p_i = 1
p_len = len(list(model.parameters()))
unfreeze = args.unfreeze
stop_layer = p_len - unfreeze

model.load_weights(args.weightsfile, stop=stop_layer)

# Freeze all weights before layer "stop_layer" from "unfreeze" argument
for p in model.parameters():
    if p_i < stop_layer:
        p.requires_grad = False
    else:
        p.requires_grad = True
    p_i += 1

model.train()
model = model.to(device)

# Load the config file
net_options =  model.net_info

##Parse the config file
batch = net_options['batch']
subdivisions = net_options['subdivisions']  #Irrelavant for our implementation simnce we laod the entire batch into our RAM
#In darknet, subdivisions would the number of examples loaded into the RAM at once (after being concatenated)
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

# Overloading custom data transforms from customloader (may add more here)
custom_transforms = Sequence([YoloResizeTransform(inp_dim)])

# Data instance and loader
data = CustomDataset(root="data", num_classes=num_classes, 
                     ann_file="data/train.txt", 
                     det_transforms=custom_transforms)
print('Batch size ', bs)
data_loader = DataLoader(data, 
                         batch_size=bs,
                         shuffle=False,
                         collate_fn=data.collate_fn)

# Use this optimizer calculation for training loss
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

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

    total_loss = 0
    
    #get the objectness loss
    loss_inds = torch.nonzero(ground_truth[:,:,-4] > -1)
    objectness_pred = output[loss_inds[:,0],loss_inds[:,1],4]
    target = ground_truth[loss_inds[:,0],loss_inds[:,1],4]
    objectness_loss = torch.nn.MSELoss(size_average=False)(objectness_pred, target)
    print("Obj Loss", objectness_loss)
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
    
    print("Num_gt:", gt_ob.shape[0])
    print("Center_x_loss", float(centre_x_loss))
    print("Center_y_loss", float(centre_y_loss))

    total_loss += centre_x_loss 
    total_loss += centre_y_loss 
    
    #get w,h loss
    w_loss = torch.nn.MSELoss(size_average=False)(pred_ob[:,2], gt_ob[:,2])
    h_loss = torch.nn.MSELoss(size_average=False)(pred_ob[:,3], gt_ob[:,3])
    
    total_loss += w_loss 
    total_loss += h_loss 
    
    print("w_loss:", float(w_loss))
    print("h_loss:", float(h_loss))


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

    print(cls_loss)
    total_loss += cls_loss

    return total_loss

### TRAIN MODEL ###

itern = 0
epochs = int(len(data) / bs)
lr_update_step = 0.8 * epochs
lr_updated = False
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
    
    # Update LR
    for param_group in optimizer.param_groups:
        if itern >= lr_update_step and lr_updated == False:
            optimizer.param_groups[0]["lr"] = (lr*pow((itern / lr_update_step),4))
            lr_updated = True
    print('lr: ', optimizer.param_groups[0]["lr"])

    if loss:
        print("Loss for iter no: {}: {}".format(itern, float(loss)/bs))
        writer.add_scalar("Loss/vanilla", float(loss), itern)
        loss.backward()
        optimizer.step()

    itern += 1

### FINE TUNE MODEL ON MORE LAYERS ###

# "unfreeze" refers to the last number of layers to tune (allow gradients to be tracked)
p_i = 1
p_len = len(list(model.parameters()))
stop_layer = 5 # Unfreeze all but this number of layers at the beginning

# Unfreeze more layers for fine-tuning
for p in model.parameters():
    if p_i < stop_layer:
        p.requires_grad = False
    else:
        p.requires_grad = True
    p_i += 1

# New iteration counter and make sure LR is set correctly
itern_fine = 0
lr = optimizer.param_groups[0]["lr"] / 10
lr_updated = False
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

# Reset data loader
print('Batch size ', bs)
# Data instance and loader
data = CustomDataset(root="data", num_classes=num_classes, 
                     ann_file="data/train.txt", 
                     det_transforms=custom_transforms)
data_loader = DataLoader(data, batch_size=bs,
                         shuffle=False,
                         collate_fn=data.collate_fn)

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
    
    # Update learning rate (decrease) at lr_update_step specified above
    for param_group in optimizer.param_groups:
        if itern_fine >= lr_update_step and lr_updated == False:
            optimizer.param_groups[0]["lr"] = (lr*pow((itern_fine / lr_update_step),4))
            lr_updated == True
    
    print('lr: ', optimizer.param_groups[0]["lr"])
    if loss:
        print("Loss for iter no: {}: {}".format(itern, float(loss)/bs))
        writer.add_scalar("Loss/vanilla", float(loss), itern)
        if itern_fine % 5 == 0:
            torch.save(model.state_dict(), os.path.join('runs', 'epoch{0}-bs{1}-loss{2:.4f}.pth'.format(itern, bs, float(loss)/bs)))
        loss.backward()
        optimizer.step()

    itern_fine += 1
    itern += 1

writer.close()

# Save final model in pytorch format (the state dictionary only, i.e. parameters only)
torch.save(model.state_dict(), os.path.join('runs', 'epoch{0}-final-bs{1}-loss{2:.4f}.pth'.format(itern, bs, float(loss)/bs)))    
    
    



