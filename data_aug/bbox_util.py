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


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))
    image = cv2.resize(image, (w,h))
    return image



def get_corners(bboxes):
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.vstack(x1,y1,x2,y2,x3,y3)

    