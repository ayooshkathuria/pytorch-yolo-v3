import random
import numpy as np
import cv2


class Sequence(object):
    """
    Takes in a list of augmentation functions
    
    """
    
    def __init__(self, augmentations):
        self.augmentations = augmentations
    
    def __call__(self, images, bboxes):
        for augmentation in self.augmentations:
            images, bboxes = augmentation(images, bboxes)
        
        return images, bboxes
    
class RandomHorizontalFlipForDet(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

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
    

class RandomZoomForDet(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, zoom_factor = 0.2):
        self.zoom_factor = zoom_factor

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        
        
        zoom = random.uniform(-self.zoom_factor, self.zoom_factor)
        
        img_center = [ int(x/2) for x in img.shape[:2]]
        
        img = cv2.circle(img.copy(), tuple(img_center[::-1]), 15, 0, -1)
        
        if zoom > 0:
            img = cv2.resize(img, None, fx = 1 + zoom, fy = 1 + zoom, interpolation = cv2.INTER_CUBIC)
            
            return img, bboxes
            a = int(img.shape[0]*(zoom)*2)
            corner_y = random.randint(0, a)
            a = int(img.shape[1]*(zoom)*2)
            corner_x = random.randint(0, a)
            
            img = img[corner_y:corner_y + img.shape[0], corner_x:corner_x + img.shape[1], :]
        
        
        
        return img, bboxes
        
        
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)