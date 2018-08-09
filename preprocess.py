from __future__ import division
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from util import count_parameters as count
from util import convert2cpu as cpu
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
import os
import random
from torchvision import transforms
from data_aug.data_aug import *
from data_aug.bbox_util import draw_rect



def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas


        
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0)
    return img_, orig_im, dim

def prep_image_pil(img, network_dim):
    orig_im = Image.open(img)
    img = orig_im.convert('RGB')
    dim = img.size
    img = img.resize(network_dim)
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(*network_dim, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3,*network_dim)
    img = img.float().div(255.0).unsqueeze(0)
    return (img, orig_im, dim)

def inp_to_image(inp):
    inp = inp.cpu().squeeze()
    inp = inp*255
    try:
        inp = inp.data.numpy()
    except RuntimeError:
        inp = inp.numpy()
    inp = inp.transpose(1,2,0)

    inp = inp[:,:,::-1]
    return inp


class inferset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, inp_dim = 416, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.list_ims = [os.path.join(root_dir, x) for x in os.listdir(self.root_dir)]
        self.transform = transform
        self.inp_dim = inp_dim

    def __len__(self):
        return len(self.list_ims)
    
    def imlist(self):
        return self.list_ims

    def __getitem__(self, idx):
        img = self.list_ims[idx]
        image, orig_im, dim = prep_image(img, self.inp_dim)
        return idx, image, dim
    
    



    
    
class toyset(Dataset):

    def __init__(self, img, transform=None):
    
        self.img = img
        self.transform = transform

    def __len__(self):
        return 1
    

    def __getitem__(self, idx):
        image = cv2.imread(self.img)
        name =  self.img.split(".")[0]
        name = "{}.txt".format(name)
        
        annots = open(name, "r").readlines()
        
        annots_ = []
        
        for annot in annots:
            temp = annot.rstrip().split(" ")[1:]
            temp = [float(x) for x in temp]
            annots_.append(temp)
            


        annots_ = np.array(annots_)
        annots_transform = np.zeros(annots_.shape)
        
        annots_transform[:,0] = annots_[:,0] - annots_[:,2]/2
        annots_transform[:,1] = annots_[:,1] - annots_[:,3]/2
        annots_transform[:,2] = annots_[:,0] + annots_[:,2]/2
        annots_transform[:,3] = annots_[:,1] + annots_[:,3]/2
        
        
        im_dim = np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        
        
        cls = np.array([0,0,0,1]).reshape(-1,1)
        
        
        annots_transform  = np.hstack((annots_transform, cls.reshape(-1,1))) 
        
        annots_transform[:,:4] *= im_dim

        
        if self.transform:
            image, annots_transform = self.transform(image, annots_transform)
        
        image = image.copy()
        
        return image, annots_transform
    
#tran = Sequence([RandomRotate(10)])

#tran = Sequence([RandomHSV(40, 40, 30),RandomHorizontalFlip(), RandomScaleTranslate(), RandomRotate(10), RandomShear(), YoloResize(608)])

#toyloader = DataLoader(toyset("data_aug/demo.jpeg", transform = tran))

#for x, ann in toyloader:
#    x = x.squeeze().numpy()
#    ann = ann.squeeze().numpy()
#    x = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2RGB)
#    
#    for cord in ann:
#        x = draw_rect(x, cord)
#        
#    plt.imshow(x)
#    cv2.imwrite("test.png", x)
#    plt.show() 
    
    
