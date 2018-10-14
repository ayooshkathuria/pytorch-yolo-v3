import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
import os 

a = os.path.join(os.path.realpath("."), "data_aug")

sys.path.append(a)
from data_aug.bbox_util import *


class Sequence(object):

    """Initialise Sequence object
    
    Apply a Sequence of transformations to the images/boxes.
    
    Parameters
    ----------
    augemnetations : list 
        List containing Transformation Objects in Sequence they are to be 
        applied
    
    probs : int or list 
        If **int**, the probability with which each of the transformation will 
        be applied. If **list**, the length must be equal to *augmentations*. 
        Each element of this list is the probability with which each 
        corresponding transformation is applied
    
    Returns
    -------
    
    Sequence
        Sequence Object 
        
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
    
    """Randomly horizontally flips the Image with the probability *p*
    
    Parameters
    ----------
    p: float
        The probability with which the image is flipped
        
        
    Returns
    -------
    
    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):

        
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))
        if random.random() < self.p:
            img =  img[:,::-1,:]
            bboxes[:,[0,2]] += 2*(img_center[[0,2]] - bboxes[:,[0,2]])
            
        return img, bboxes

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class HorizontalFlip(object):
    """Horizontally Flips the Image 
    
    Parameters
    ----------
    
        
    Returns
    -------
    
    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """
    def __init__(self):
        pass

    def __call__(self, img, bboxes):
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))
        img =  img[:,::-1,:]
        bboxes[:,[0,2]] += 2*(img_center[[0,2]] - bboxes[:,[0,2]])
            
        return img, bboxes

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

class RandomScaleTranslate(object):
    """Randomly Scales and Translate the image    
    
    The image is first scaled followed by translation.Bounding boxes which have 
    an area of less than 25% in the remaining in the transformed image is dropped.
    The resolution is maintained, and the remaining area if any is filled by
    black color.
    
    
    
    Parameters
    ----------
    scale: float or tuple(float)
        if **float**, The image is scaled by a factor drawn 
        randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**, the `scale`
        is drawn randomly from values specified by the tuple
        
    translate: float or tuple(float)
        if **float**, The image is translated in both the x and y directions
        by factors drawn randomly from a range (1 - `translate` , 1 + `translate`). 
        If **tuple**, `translate` is drawn randomly from values specified by 
        the tuple. 
        
    Returns
    -------
    
    numpy.ndaaray
        Scaled and translated image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
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
    


    def __call__(self, img, bboxes):
        
        scale = random.uniform(*self.scale)
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)

        
        #Chose a random digit to scale by 
        img_shape = img.shape
        
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)

        
    
        resize_scale = 1 + scale
        
        if resize_scale > 1:
            img=  cv2.resize(img, None, fx = resize_scale, fy = resize_scale)
        else:
            img = cv2.resize(img, None, fx = resize_scale, fy = resize_scale, interpolation = cv2.INTER_AREA)
        
        bboxes[:,:4] *= [resize_scale, resize_scale, resize_scale, resize_scale]
        
        #translate the image
        
        #percentage of the dimension of the image to translate

        
        
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
        canvas[:,:] = [0,0,0]
        mask = img[max(corner_y, 0):min(img.shape[0], corner_y + img_shape[0]), max(corner_x, 0):min(img.shape[1], corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas
        
        
        bboxes[:,:4] -= [corner_x, corner_y, corner_x, corner_y]
        
        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.5)
        

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        


        return img, bboxes
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

    


    
    

class RandomTranslate(object):
    """Randomly Translates the image    
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    translate: float or tuple(float)
        if **float**, the image is translated by a factor drawn 
        randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
        `translate` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Translated image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
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
        
        bboxes[:,:4] -= [corner_x, corner_y, corner_x, corner_y]
        
        
        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)
        

        

        
        return img, bboxes
    
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class Translate(object):
    """Translates the image    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
        
    translate_x: float
       The factor by which the image is translated in the x direction

    translate_y: float
       The factor by which the image is translated in the y direction
        
    Returns
    -------
    
    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, translate_x = 0.2, translate_y = 0.2):

        self.translate_factor_x = translate_x
        self.translate_factor_y = translate_y
        
        

    def __call__(self, img, bboxes):

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
        
        bboxes[:,:4] -= [corner_x, corner_y, corner_x, corner_y]
        
        
        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)
        

        

        
        return img, bboxes
    
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    
    
class RandomScale(object):
    """Randomly scales an image    
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    scale: float or tuple(float)
        if **float**, the image is scaled by a factor drawn 
        randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**,
        the `scale` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, scale = 0.2):
        self.scale = scale
        
        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"   
        else:
            self.scale = (-self.scale, self.scale)
        

        

    def __call__(self, img, bboxes):

        
        #Chose a random digit to scale by 
        
        img_shape = img.shape
        
        scale = random.uniform(*self.scale)
        resize_scale = 1 + scale
        
        if resize_scale > 1:
            img=  cv2.resize(img, None, fx = resize_scale, fy = resize_scale)
        else:
            img = cv2.resize(img, None, fx = resize_scale, fy = resize_scale, interpolation = cv2.INTER_AREA)
        
        bboxes[:,:4] *= [resize_scale, resize_scale, resize_scale, resize_scale]
        
        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)

        return img, bboxes
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    
class Scale(object):
    """Scales the image    
        
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    
    Parameters
    ----------
    scale: float
        The factor by which the image is scaled.
        
    Returns
    -------
    
    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, scale = 0.2):
        self.scale = scale
        

    def __call__(self, img, bboxes):

        #Chose a random digit to scale by 
        
        img_shape = img.shape
        
        resize_scale = 1 + self.scale
        
        if resize_scale > 1:
            img=  cv2.resize(img, None, fx = resize_scale, fy = resize_scale)
        else:
            img = cv2.resize(img, None, fx = resize_scale, fy = resize_scale, interpolation = cv2.INTER_AREA)
        
        bboxes[:,:4] *= [resize_scale, resize_scale, resize_scale, resize_scale]
        
        
        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)

        return img, bboxes
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

class RandomRotate(object):
    """Randomly rotates an image    
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    angle: float or tuple(float)
        if **float**, the image is rotated by a factor drawn 
        randomly from a range (-`angle`, `angle`). If **tuple**,
        the `angle` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, angle = 10):
        self.angle = angle
        
        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"   
        else:
            self.angle = (-self.angle, self.angle)
            
    def __call__(self, img, bboxes):
        
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
            
            
        """
        
        angle = random.uniform(*self.angle)

        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
        
        corners = get_corners(bboxes)
        
        corners = np.hstack((corners, bboxes[:,4:]))

        img = rotate_bound(img, angle)
        
        corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
        
        
        
        
        new_bbox = get_enclosing_box(corners)
        
        
        scale_factor_x = img.shape[1] / w
        
        scale_factor_y = img.shape[0] / h
        
        img = cv2.resize(img, (w,h))
        
        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
        
        
        
        
        
        bboxes  = new_bbox

        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
        
        return img, bboxes
        
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

class Rotate(object):
    """Rotates an image    
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    angle: float
        The angle by which the image is to be rotated 
        
        
    Returns
    -------
    
    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
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
        
        corners = np.hstack((corners, bboxes[:,4:]))

        img = rotate_bound(img, angle)
        
        corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
        
        
        
        
        new_bbox = get_enclosing_box(corners)
        
        
        scale_factor_x = img.shape[1] / w
        
        scale_factor_y = img.shape[0] / h
        
        img = cv2.resize(img, (w,h))
        
        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
        
        
        bboxes  = new_bbox

        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
        
        return img, bboxes
        
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    
class RandomShear(object):
    """Randomly shears an image in horizontal direction   
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    shear_factor: float or tuple(float)
        if **float**, the image is sheared horizontally by a factor drawn 
        randomly from a range (-`shear_factor`, `shear_factor`). If **tuple**,
        the `shear_factor` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, shear_factor = 0.2):
        self.shear_factor = shear_factor
        
        if type(self.shear_factor) == tuple:
            assert len(self.shear_factor) == 2, "Invalid range for scaling factor"   
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)
        
        shear_factor = random.uniform(*self.shear_factor)

    def __call__(self, img, bboxes):
        
        shear_factor = random.uniform(*self.shear_factor)


        M = np.array([[1, shear_factor, 0],[0,1,0]])
                
        nW =  img.shape[1] + abs(shear_factor*img.shape[0])
        
        bboxes[:,[0,2]] += (bboxes[:,[1,3]]*shear_factor).astype(int) 
        

        
        if shear_factor < 0:
            M[0,2] += (nW - img.shape[1])
            bboxes[:,[0,2]] += (nW - img.shape[1])
        
        
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
        
        
        return img, bboxes
        
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class Shear(object):
    """Shears an image in horizontal direction   
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    shear_factor: float
        Factor by which the image is sheared in the x-direction
       
    Returns
    -------
    
    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, shear_factor = 0.2):
        self.shear_factor = shear_factor
        
    
    def __call__(self, img, bboxes):

        M = np.array([[1, self.shear_factor, 0],[0,1,0]])
                
        nW =  img.shape[1] + abs(self.sheashearsr_factor*img.shape[0])
        
        bboxes[:,[0,2]] += (bboxes[:,[1,3]]*self.shear_factor).astype(int) 

        
        if self.shear_factor < 0:
            M[0,2] += (nW - img.shape[1])
            bboxes[:,[0,2]] += (nW - img.shape[1])
        
        
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
        
        
        return img, bboxes
        
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class YoloResize(object):
    """Resize the image in accordance to `image_letter_box` function in darknet 
    
    The aspect ratio is maintained. The longer side is resized to the input 
    size of the network, while the remaining space on the shorter side is filled 
    with black color. **This should be the last transform**
    
    
    Parameters
    ----------
    inp_dim : tuple(int)
        tuple containing the size to which the image will be resized.
        
    Returns
    -------
    
    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """
    
    def __init__(self, inp_dim):
        self.inp_dim = inp_dim
    
    def __call__(self, img, bboxes):
        img, bboxes = bboxes # TODO: fix bboxes (it also hold img data for custom)
        w,h = img.shape[1], img.shape[0]
        img = letterbox_image(img, self.inp_dim)        

        scale = min(self.inp_dim/h, self.inp_dim/w)
        bboxes[:,:4] *= (scale)
        new_w = scale*w
        new_h = scale*h
        inp_dim = self.inp_dim   
        
        del_h = (inp_dim - new_h)/2
        del_w = (inp_dim - new_w)/2
        
        add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)
        
        bboxes[:,:4] += add_matrix
        
        img = img.astype(np.uint8)
        
        return img, bboxes


class RandomHSV(object):
    """HSV Transform to vary hue saturation and brightness
    
    Hue has a range of 0-179
    Saturation and Brightness have a range of 0-255. 
    Chose the amount you want to change thhe above quantities accordingly. 
    
    
    
    
    Parameters
    ----------
    hue : None or float or tuple (float)
        If None, the hue of the image is left unchanged. If float, 
        a random float is uniformly sampled from (-hue, hue) and added to the 
        hue of the image. If tuple, the float is sampled from the range 
        specified by the tuple.   
        
    saturation : None or float or tuple(float)
        If None, the saturation of the image is left unchanged. If float, 
        a random float is uniformly sampled from (-saturation, saturation) 
        and added to the hue of the image. If tuple, the float is sampled
        from the range  specified by the tuple.   
        
    brightness : None or float or tuple(float)
        If None, the brightness of the image is left unchanged. If float, 
        a random float is uniformly sampled from (-brightness, brightness) 
        and added to the hue of the image. If tuple, the float is sampled
        from the range  specified by the tuple.   
    
    Returns
    -------
    
    numpy.ndaaray
        Transformed image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """
    
    def __init__(self, hue = None, saturation = None, brightness = None):
        if hue:
            self.hue = hue 
        else:
            self.hue = 0
            
        if saturation:
            self.saturation = saturation 
        else:
            self.saturation = 0
            
        if brightness:
            self.brightness = brightness
        else:
            self.brightness = 0
            
            

        if type(self.hue) != tuple:
            self.hue = (-self.hue, self.hue)
            
        if type(self.saturation) != tuple:
            self.saturation = (-self.saturation, self.saturation)
        
        if type(brightness) != tuple:
            self.brightness = (-self.brightness, self.brightness)
    
    def __call__(self, img, bboxes):

        hue = random.randint(*self.hue)
        saturation = random.randint(*self.saturation)
        brightness = random.randint(*self.brightness)
        
        img = img.astype(int)
        
        a = np.array([hue, saturation, brightness]).astype(int)
        img += np.reshape(a, (1,1,3))
        
        img = np.clip(img, 0, 255)
        img[:,:,0] = np.clip(img[:,:,0],0, 179)
        
        img = img.astype(np.uint8)

        
        
        return img, bboxes
    
