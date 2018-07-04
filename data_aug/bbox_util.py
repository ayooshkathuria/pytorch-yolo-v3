import cv2
import numpy as np

def draw_rect(im, cords):
    pt1, pt2 = (cords[0], cords[1]) , (cords[2], cords[3])
            
    pt1 = int(pt1[0]), int(pt1[1])
    pt2 = int(pt2[0]), int(pt2[1])

    im = cv2.rectangle(im.copy(), pt1, pt2, [0,0,0], int(max(im.shape[:2])/150))
    return im, pt1, pt2

def clip_box(bbox, clip_box, alpha):
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:,0], clip_box[0])
    y_min = np.maximum(bbox[:,1], clip_box[1])
    x_max = np.minimum(bbox[:,2], clip_box[2])
    y_max = np.minimum(bbox[:,3], clip_box[3])
    
    bbox = np.vstack((x_min, y_min, x_max, y_max)).T
    
    delta_area = ((ar_ - bbox_area(bbox))/ar_)
    
    mask = (delta_area < (1 - alpha)).astype(int)
    
    bbox = bbox[mask == 1,:]


    return bbox
    
    
def bbox_area(bbox):
    ar = (bbox[:,3] - bbox[:,1])*(bbox[:,2] - bbox[:,0])
    return ar