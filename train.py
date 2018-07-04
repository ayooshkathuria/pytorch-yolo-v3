import torch
import os
import argparse
from darknet import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def arg_parse():
    """
    Parse arguements to the detect module

    """


    parser = argparse.ArgumentParser(description='YOLO v3 Training Module')


    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "cfg/yolov3-train.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--datacfg", dest = "datafile", help = "cfg file containing the configuration for the dataset",
                        type = str, default = "cfg/coco.data")
    return parser.parse_args()


args = arg_parse()

#Load the model
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
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
