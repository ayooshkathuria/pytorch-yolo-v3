import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
import os 

a = os.path.join(os.path.realpath("."), "data_aug")

sys.path.append(a)
from bbox_util import *

a = 1

class Sequence(object):
    """
    Takes in a list of augmentation functions to apply consecutively
    
    """
    
    def __init__(self, augmentations, probs = 1):
        self.augmentations = augmentations
        self.probs = probs
        
    
    def __call__(self, images, bboxes):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs
                
            if random.random() < prob:
                images, bboxes = augmentation(images, bboxes)
        
        return images, bboxes
    
class RandomHorizontalFlip(object):
    """Horizontally flip the image with the probability p.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        
        
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))
        if random.random() < self.p:
            img =  img[:,::-1,:]
            bboxes[:,[0,2]] += 2*(img_center[[0,2]] - bboxes[:,[0,2]])
            
        return img, bboxes

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

class RandomScaleTranslate(object):
    """

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, scale = 0.2, translate = 0.2):
        self.scale = scale
        self.translate = translate
        

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        
        
        #Chose a random digit to scale by 
        img_shape = img.shape
        
        scale = random.uniform(-self.scale, self.scale)
        
        resize_scale = 1 + scale
        
        if resize_scale > 1:
            img=  cv2.resize(img, None, fx = resize_scale, fy = resize_scale)
        else:
            img = cv2.resize(img, None, fx = resize_scale, fy = resize_scale, interpolation = cv2.INTER_AREA)
        
        bboxes *= [resize_scale, resize_scale, resize_scale, resize_scale]
        
        #translate the image
        
        #percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(-self.translate, self.translate)
        translate_factor_y = random.uniform(-self.translate, self.translate)
        
        
        #Get the center co-ordinates of the shifted Image
        cx, cy = img.shape[1]/2, img.shape[0]/2
        cx += translate_factor_x*img.shape[1]
        cy += translate_factor_y*img.shape[0]
        

        #get the top-left corner co-ordinates of the shifted box 
        corner_x = int(cx) - int(img.shape[1]/2)
        corner_y = int(cy) - int(img.shape[0]/2)
        
        #change the origin to the top-left corner of the translated box
        orig_box_cords =  [max(0,-corner_y), max(-corner_x,0), min(img_shape[0], -corner_y + img.shape[0]), min(img_shape[1],-corner_x + img.shape[1])]

        
        
        canvas = np.zeros(img_shape) 
        canvas[:,:] = [127,127,127]
        mask = img[max(corner_y, 0):min(img.shape[0], corner_y + img_shape[0]), max(corner_x, 0):min(img.shape[1], corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas
        
        bboxes -= [corner_x, corner_y, corner_x, corner_y]
        
        
        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)
        

        

        
        return img, bboxes
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

class RandomHorizontalFlip(object):
    """Horizontally flip the image with the probability p.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        
        
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))
        if random.random() < self.p:
            img =  img[:,::-1,:]
            bboxes[:,[0,2]] += 2*(img_center[[0,2]] - bboxes[:,[0,2]])
            
        return img, bboxes

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

class RandomTranslate(object):
    """
    
    Translate an image randomly by a length (translate*dimension) of the image in both the vertical and 
    horizontal direction

    Args:
        translate (float): Default value 0.2
        
    """

    def __init__(self, scale = 0.2, translate = 0.2):
        self.scale = scale
        self.translate = translate
        

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        
        
        #Chose a random digit to scale by 
        img_shape = img.shape
        
        #translate the image
        
        #percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(-self.translate, self.translate)
        translate_factor_y = random.uniform(-self.translate, self.translate)
        
        
        #Get the center co-ordinates of the shifted Image
        cx, cy = img.shape[1]/2, img.shape[0]/2
        cx += translate_factor_x*img.shape[1]
        cy += translate_factor_y*img.shape[0]
        

        #get the top-left corner co-ordinates of the shifted box 
        corner_x = int(cx) - int(img.shape[1]/2)
        corner_y = int(cy) - int(img.shape[0]/2)
        
        #change the origin to the top-left corner of the translated box
        orig_box_cords =  [max(0,-corner_y), max(-corner_x,0), min(img_shape[0], -corner_y + img.shape[0]), min(img_shape[1],-corner_x + img.shape[1])]

        
        
        canvas = np.zeros(img_shape) 
        canvas[:,:] = [127,127,127]
        mask = img[max(corner_y, 0):min(img.shape[0], corner_y + img_shape[0]), max(corner_x, 0):min(img.shape[1], corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas
        
        bboxes -= [corner_x, corner_y, corner_x, corner_y]
        
        
        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)
        
        print(bboxes.shape)

        

        
        return img, bboxes
    
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    
    
class RandomScale(object):
    """
    
    Randomly scale the image by a factor of (1 - scale, 1 + scale)

    Args:
        scale (float)
    """

    def __init__(self, scale = 0.2):
        self.scale = scale
        

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        
        
        #Chose a random digit to scale by 
        
        img_shape = img.shape
        scale = random.uniform(-self.scale, self.scale)
        
        resize_scale = 1 + scale
        
        if resize_scale > 1:
            img=  cv2.resize(img, None, fx = resize_scale, fy = resize_scale)
        else:
            img = cv2.resize(img, None, fx = resize_scale, fy = resize_scale, interpolation = cv2.INTER_AREA)
        
        bboxes *= [resize_scale, resize_scale, resize_scale, resize_scale]
        
        
        
        #Get the center co-ordinates of the shifted Image
        cx, cy = img.shape[1]/2, img.shape[0]/2


        #get the top-left corner co-ordinates of the shifted box 
        corner_x = int(cx) - int(img.shape[1]/2)
        corner_y = int(cy) - int(img.shape[0]/2)
        
        #change the origin to the top-left corner of the translated box
        orig_box_cords =  [max(0,-corner_y), max(-corner_x,0), min(img_shape[0], -corner_y + img.shape[0]), min(img_shape[1],-corner_x + img.shape[1])]

        
        
        canvas = np.zeros(img_shape) 
        canvas[:,:] = [127,127,127]
        mask = img[max(corner_y, 0):min(img.shape[0], corner_y + img_shape[0]), max(corner_x, 0):min(img.shape[1], corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas
        
        bboxes -= [corner_x, corner_y, corner_x, corner_y]
        
        
        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)

        return img, bboxes
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

class RandomRotate(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
            
            
        """
        
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
        
        print(bboxes.shape)
        corners = get_corners(bboxes)
        img = rotate_bound(img, self.angle)
        
        corners = rotate_box(corners, self.angle, cx, cy, h, w)
        
        new_bbox = get_enclosing_box(corners)

        return img, bboxes
        
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)