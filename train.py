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
from cocoloader import *
import torch.optim as optim
random.seed(0)
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
                        type = str, default = "cfg/coco.data")
    parser.add_argument("--lr", dest = "lr", type = float, default = 0.001)
    parser.add_argument("--bs", dest = "bs", type = int, default = 1)
    parser.add_argument("--mom", dest = "mom", type = float, default = 0)
    parser.add_argument("--wd", dest = "wd", type = float, default = 0)

    return parser.parse_args()


args = arg_parse()

args.weightsfile = "darknet53.conv.74"
#Load the model
model = Darknet(args.cfgfile).to(device)
model.load_weights(args.weightsfile, stop = 74)
model.train()
#assert False
model = model.to(device)  ## Really? You're gonna train on the CPU?

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


#lr = sys.argv[0]
#wd = sys.argv[1]

num_classes = 80
lr = args.lr
wd = args.wd
bs = args.bs
momentum = args.mom
momentum = 0.9
wd = 0.0005
bs = 2


inp_dim = 416
transforms = Sequence([YoloResize(inp_dim)])

coco = CocoDataset(root = "COCO/train2017", annFile="COCO_ann_mod.pkl", det_transforms = transforms)

coco_loader = DataLoader(coco, batch_size = bs)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay = wd)



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
    cls_scores_pred = pred_ob[:,5:]
    
    
    cls_scores_target = gt_ob[:,5].long()
    
    
    
    
    
    
    cls_scores_pred = torch.log(torch.sigmoid(cls_scores_pred))
    
#    print(cls_scores_pred.shape)
#    print(cls_scores_target.shape)
#    cls_loss = torch.nn.NLLLoss(size_average=False, reduce = False)(cls_scores_pred, cls_scores_target)
        
    cls_loss = 0
    
    cls_labels = torch.zeros(gt_ob.shape[0], num_classes)
    
    cls_labels[torch.arange(gt_ob.shape[0]).long(), gt_ob[:,5].long()] = 1
    
    
    cls_loss = 0    
    
    for cls in range(num_classes):
        targ_labels = pred_ob[:,5 + cls].view(-1,1)
        targ_labels = targ_labels.repeat(1,2)
        targ_labels[:,0] = 1 - targ_labels[:,0]
        cls_loss += torch.nn.CrossEntropyLoss(size_average=False)(targ_labels, cls_labels[:,cls].long())
        
    
    
    
    print(cls_loss)
    total_loss += cls_loss
    
    
    
    
    return total_loss
    

    
    
    #get the centerx, and centre y
    
    

    
    
itern = 0
    
for batch in coco_loader:
    batch[0] = batch[0].to(device)
    batch[1] = batch[1].to(device)
    
    output = model(batch[0])
    ground_truth= batch[1]
    
        
    
    print("\n\n")
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
    
    for param_group in optimizer.param_groups:
        if itern < 2000:
            param_group["lr"] = (lr*pow((itern / 2000),4))
            
        param_group["lr"] /= args.bs
            
        
            
    
    print(optimizer.param_groups[0]["lr"])
    if loss:
        print("Loss for iter no: {}: {}".format(itern, float(loss)/args.bs))
        writer.add_scalar("Loss/vanilla", float(loss), itern)
        loss.backward()
        optimizer.step()
    

    itern += 1
    
writer.close()
    
    

#    res = torch.autograd.gradcheck(YOLO_loss, (ground_truth, output), raise_exception=True)

    
    
    



