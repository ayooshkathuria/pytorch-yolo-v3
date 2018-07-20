import torch 
from torchvision.datasets import CocoDetection
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from data_aug.bbox_util import draw_rect
from data_aug.data_aug import *

class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, det_transforms = None):
        super().__init__(root, annFile, None, None)
        self.det_tranforms = det_transforms
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
    

#dirname = os.path.realpath("..")
#filename = os.path.join(dirname, "cocoapi/train2017")
#print(filename)
#print(os.path.exists(filename))

#cocoloader = CocoDataset(root = "../cocoapi/train2017", annFile = "../cocoapi/annotations/instances_train2017.json")

def tiny_coco(cocoloader, num):
    i = 0
    li = []
    
    for x in cocoloader:
        print(i)
        li.append(x)
        i += 1
        if i > num - 1:
            break
    num = num / 1000
    pkl.dump(li, open("COCO_{}k.pkl.format(num)", "wb"))


coco_loader = pkl.load(open("COCO_100.pkl", "rb"))

def trasform_annotation(x):
    #convert the PIL image to a numpy array
    image = np.array(x[0])
    
    #get the bounding boxes and convert them into proper format 
    boxes = [a["bbox"] for a in x[1]]
    
    boxes = np.array(boxes)
    
    boxes = boxes.reshape(-1,4)
    
    boxes[:,2] += boxes[:,0]
    boxes[:,3] += boxes[:,1]

    
    category_ids = [a["category_id"] for a in x[1]]
    
    return image, boxes, category_ids
    
    
transforms = Sequence([RandomHorizontalFlip(), RandomScaleTranslate(translate=0.05, scale=(0,0.3)), RandomRotate(10), RandomShear(), YoloResize(448)])
    
i = 0
boxes = []
li = []
#for x in coco_loader:
#    x = trasform_annotation(x)
#    a = transforms(x[0], x[1])
#    im = draw_rect(a[0], a[1])
#    plt.imshow(im)	
#    plt.show()
#    i += 1
#    if i == 10:
#        break

for x in coco_loader:
    x = trasform_annotation(x)
    bboxes = x[1]
    bbox_dims_h = bboxes[:,3] - bboxes[:,1]
    bbox_dims_w = bboxes[:,2] - bboxes[:,0]
    

    bbox_dims = np.stack((bbox_dims_w, bbox_dims_h)).T
    
    li.append(bbox_dims)


dims = np.vstack(li)
    