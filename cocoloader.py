import torch 
from torchvision.datasets import CocoDetection
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from data_aug.bbox_util import draw_rect
from data_aug.data_aug import *
from kmeans.kmeans import *

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

cocoloader = CocoDataset(root = "../COCO/train2017", annFile = "../COCO/instances_train2017.json")

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


#coco_loader = pkl.load(open("COCO_100.pkl", "rb"))

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

assert False    

#for x in coco_loader:
#    x = trasform_annotation(x)
#    a = transforms(x[0], x[1])
#    im = draw_rect(a[0], a[1])
#    plt.imshow(im)	
#    plt.show()
#    i += 1
#    if i == 10:
#        break


# KMeans --------------------------------------------------------------------------------------------

def get_bbox_dims(coco_loader):
    li = []
    for x in coco_loader:
        x = trasform_annotation(x)
        bboxes = x[1]
        bbox_dims_h = bboxes[:,3] - bboxes[:,1]
        bbox_dims_w = bboxes[:,2] - bboxes[:,0]
        
    
        bbox_dims = np.stack((bbox_dims_w, bbox_dims_h)).T
        
        li.append(bbox_dims)


    dims = np.vstack(li)
    return dims 


a = get_bbox_dims(coco_loader)
    

def YOLO_kmeans(points):
    centroids = random.sample(range(points.shape[0]), 5)
    centroids = points[centroids]

    for iter in range(10):
        clusters = get_clusters(points, centroids)
        
        for k in range(5):
            arr = points[clusters == k]
            centroids[k] = arr.mean(0)
            
    return centroids, clusters 
            
        
        
        
        
    
        
        
def get_clusters(points, centroids):
    
    points = points.reshape(points.shape[0], 1, points.shape[1])
    
    centroids = centroids.reshape(centroids.shape[0], 1, centroids.shape[1])
    centroids = centroids.transpose((1,0,2))
    
    min_w  = np.minimum(points[:,:,0], centroids[:,:,0])
    min_h  = np.minimum(points[:,:,1], centroids[:,:,1])
    
    a_points = points[:,:,0]*points[:,:,1]
    c_points = centroids[:,:,0]*centroids[:,:,1]
    
    iou = (min_h*min_w)/(a_points + c_points - min_h*min_w)
    
    clusters = np.argmax(iou, 1)
                         
    return clusters
    
    
    
   
b = YOLO_kmeans(a)
    
    