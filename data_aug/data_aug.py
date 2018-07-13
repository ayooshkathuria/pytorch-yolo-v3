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
    
class HorizontalFlip(object):
    """Horizontally flip the image with the probability p.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self):
        pass

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
        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range for scaling factor"   
        else:
            self.scale = (-self.scale, self.scale)
            
        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range for scaling factor"   
        else:
            self.translate = (-self.translate, self.translate)
    
        self.scale= random.uniform(*self.scale)
        self.translate_factor_x = random.uniform(*self.translate)
        self.translate_factor_y = random.uniform(*self.translate)
            
        

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        
        
        #Chose a random digit to scale by 
        img_shape = img.shape
        
    
        resize_scale = 1 + self.scale
        
        if resize_scale > 1:
            img=  cv2.resize(img, None, fx = resize_scale, fy = resize_scale)
        else:
            img = cv2.resize(img, None, fx = resize_scale, fy = resize_scale, interpolation = cv2.INTER_AREA)
        
        bboxes *= [resize_scale, resize_scale, resize_scale, resize_scale]
        
        #translate the image
        
        #percentage of the dimension of the image to translate

        
        
        #Get the center co-ordinates of the shifted Image
        cx, cy = img.shape[1]/2, img.shape[0]/2
        cx += self.translate_factor_x*img.shape[1]
        cy += self.translate_factor_y*img.shape[0]
        

        #get the top-left corner co-ordinates of the shifted box 
        corner_x = int(cx) - int(img.shape[1]/2)
        corner_y = int(cy) - int(img.shape[0]/2)
        
        #change the origin to the top-left corner of the translated box
        orig_box_cords =  [max(0,-corner_y), max(-corner_x,0), min(img_shape[0], -corner_y + img.shape[0]), min(img_shape[1],-corner_x + img.shape[1])]

        
        
        canvas = np.zeros(img_shape) 
        canvas[:,:] = [0,0,0]
        mask = img[max(corner_y, 0):min(img.shape[0], corner_y + img_shape[0]), max(corner_x, 0):min(img.shape[1], corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas
        
        bboxes -= [corner_x, corner_y, corner_x, corner_y]
        
        
        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)
        

        

        
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

    def __init__(self, translate = 0.2):
        self.translate = translate
        
        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"   
        else:
            self.translate = (-self.translate, self.translate)
            
        self.translate_factor_x = random.uniform(*self.translate)
        self.translate_factor_y = random.uniform(*self.translate)


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
    
class Translate(object):
    """
    
    Translate an image randomly by a length (translate*dimension) of the image in both the vertical and 
    horizontal direction

    Args:
        translate (float): Default value 0.2
        
    """

    def __init__(self, translate_x = 0.2, translate_y = 0.2):

        self.translate_factor_x = translate_x
        self.translate_factor_y = translate_y
        
        

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
       
        
        #Get the center co-ordinates of the shifted Image
        cx, cy = img.shape[1]/2, img.shape[0]/2
        cx += self.translate_factor_x*img.shape[1]
        cy += self.translate_factor_y*img.shape[0]
        

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
        
        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"   
        else:
            self.scale = (-self.scale, self.scale)
        
        self.scale = random.uniform(*self.scale)
        

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        
        
        #Chose a random digit to scale by 
        
        img_shape = img.shape
        
        resize_scale = 1 + self.scale
        
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
    
    
class Scale(object):
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
        
        resize_scale = 1 + self.scale
        
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
        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"   
        else:
            self.angle = (-self.angle, self.angle)
            
        self.angle = random.uniform(*self.angle)

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
            
            
        """
        
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
        
        corners = get_corners(bboxes)
        img = rotate_bound(img, self.angle)
        
        corners, nW, nH = rotate_box(corners, self.angle, cx, cy, h, w)
        
        new_bbox = get_enclosing_box(corners)
        
        scale_factor_x = img.shape[1] / w
        
        scale_factor_y = img.shape[0] / h
        
        img = cv2.resize(img, (w,h))
        
        new_bbox /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
        
        
        
        
        
        bboxes  = new_bbox

        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
        
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
        
        corners = get_corners(bboxes)
        img = rotate_bound(img, self.angle)
        
        corners, nW, nH = rotate_box(corners, self.angle, cx, cy, h, w)
        
        new_bbox = get_enclosing_box(corners)
        
        scale_factor_x = img.shape[1] / w
        
        scale_factor_y = img.shape[0] / h
        
        img = cv2.resize(img, (w,h))
        
        new_bbox /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
        
        
        
        
        
        bboxes  = new_bbox

        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
        
        return img, bboxes
        
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    
class RandomShear(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, shear_factor = 0.2):
        self.shear_factor = shear_factor
        
        if type(self.shear_factor) == tuple:
            assert len(self.shear_factor) == 2, "Invalid range for scaling factor"   
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)
        
        self.shear_factor = random.uniform(*self.shear_factor)

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
            
            
        """
        M = np.array([[1, self.shear_factor, 0],[0,1,0]])
                
        nW =  img.shape[1] + abs(self.shear_factor*img.shape[0])
        
        bboxes[:,[0,2]] += (bboxes[:,[1,3]]*self.shear_factor).astype(int) 

        
        if self.shear_factor < 0:
            M[0,2] += (nW - img.shape[1])
            bboxes[:,[0,2]] += (nW - img.shape[1])
        
        
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
        
        
        return img, bboxes
        
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class Shear(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, shear_factor = 0.2):
        self.shear_factor = shear_factor
        
    
    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
            
            
        """
        M = np.array([[1, self.shear_factor, 0],[0,1,0]])
                
        nW =  img.shape[1] + abs(self.shear_factor*img.shape[0])
        
        bboxes[:,[0,2]] += (bboxes[:,[1,3]]*self.shear_factor).astype(int) 

        
        if self.shear_factor < 0:
            M[0,2] += (nW - img.shape[1])
            bboxes[:,[0,2]] += (nW - img.shape[1])
        
        
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
        
        
        return img, bboxes
        
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class YoloResize(object):
    
    def __init__(self, inp_dim):
        self.inp_dim = inp_dim
    
    def __call__(self, img, bboxes):
        w,h = img.shape[1], img.shape[0]
        img = letterbox_image(img, self.inp_dim)
        

        scale = min(self.inp_dim/h, self.inp_dim/w)
        bboxes *= (scale)

        new_w = scale*w
        new_h = scale*h
        
        inp_dim = self.inp_dim   
        
        del_h = (inp_dim - new_h)/2
        del_w = (inp_dim - new_w)/2
        
        add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)
        
        bboxes += add_matrix
        

        return img, bboxes