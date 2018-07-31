import torch
import os
import argparse
from darknet import *
from cocoloader import CocoDataset, transform_annotation
from util import *
import pickle
from data_aug.data_aug import *
from preprocess import *



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



transforms = Sequence([RandomHorizontalFlip(), RandomScaleTranslate(translate=0.05, scale=(0,0.3)), RandomRotate(10), RandomShear(), YoloResize(608)])

coco_loader = pkl.load(open("Coco_sample.pkl", "rb"))


strides = [32,16,8]
anchors = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],
           [156,198],  [373,326]]
inp_dim = 416
classes = 80
num_anchors = 9
anchor_nums = [3,3,3]

i = 0


def get_pred_box_cords(num_pred_boxes, label_map, strides, inp_dim, anchors_nums):
    i = 0
    for n, pred_boxes in enumerate(num_pred_boxes):
        unit = strides[n]
        corners = torch.arange(0, inp_dim, unit)
        offset = unit // 2
        grid = torch_meshgrid(corners, corners).view(-1,2)
        grid += offset
        grid = grid.repeat(1,anchors_nums[n]).view(anchors_nums[n]*grid.shape[0], -1)
        label_map[i:i+pred_boxes,[0,1]] = grid
        i += pred_boxes
    return label_map        

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

toyloader = DataLoader(toyset("data_aug/demo.jpeg", transform = transforms))


plt.rcParams["figure.figsize"] = (10,8)
for x, ann in toyloader:
    x = x.squeeze().numpy()
    cls  = np.array([0,0,0,1])
    ann = ann.squeeze().numpy()
    ann = np.hstack((ann, cls.reshape(-1,1)))
    x = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    num_pred_boxes = get_num_pred_boxes(inp_dim, strides, anchor_nums)
    
    label_map = torch.FloatTensor(sum(num_pred_boxes), 5 + classes)
    
    
    label_map = get_pred_box_cords(num_pred_boxes, label_map, 
                                   strides, inp_dim, anchor_nums)
    
    ground_truth = torch.FloatTensor(ann)
    
    ground_truth_map = get_ground_truth_map(ground_truth, label_map)
    
    for cord in ann[:,:4]:
        x = draw_rect(x, cord)

        
    plt.imshow(x)
    plt.show()        



