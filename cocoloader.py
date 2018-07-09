import torch 
from torchvision.datasets import CocoDetection
import os


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

cocoloader = CocoDataset(root = "../cocoapi/train2017", annFile = "../cocoapi/annotations/instances_train2017.json")


        
