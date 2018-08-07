import pickle as pkl
from cocoloader import transform_annotation
import random
import numpy as np
import matplotlib.pyplot as plt



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


a = pkl.load(open("Entire_dims.pkl", "rb"))



def YOLO_kmeans(points, num_k):
    centroids = random.sample(range(points.shape[0]), num_k)
    centroids = points[centroids]

    avg_ious = []
    for iter in range(150):
        clusters = get_clusters(points, centroids)
        
        for k in range(num_k):
            arr = points[clusters == k]
            centroids[k] = np.mean(arr, 0)
            
    
        ious = IOU_dist(points, centroids[clusters])
        
        avg_iou = np.mean(ious)
        
        
        avg_ious.append((avg_iou))
    
    
    plt.plot((range(len(avg_ious))), avg_ious)
    
    plt.savefig("Avg_IOU.jpeg")
    
    plt.show()
    
    print(avg_ious[-1])
    return centroids, clusters 
            
        
        
def IOU_dist(points, centroids):    
    min_w  = np.minimum(points[:,0], centroids[:,0])
    min_h  = np.minimum(points[:,1], centroids[:,1])
    
    a_points = points[:,0]*points[:,1]
    c_points = centroids[:,0]*centroids[:,1] 

    iou = (min_h*min_w)/(a_points + c_points - min_h*min_w)
    
    return iou
        
def get_clusters(points, centroids):
    
    points = points.reshape(points.shape[0], 1, points.shape[1]).transpose((0,2,1))
    
    centroids = centroids.reshape(centroids.shape[0], 1, centroids.shape[1])
    centroids = centroids.transpose((1,2,0))
    
    
    iou = IOU_dist(points, centroids)
    
    dist = 1 - iou
    
    
    clusters = np.argmin(dist, 1)

    
    
    return  clusters

num_k = 9

color_dict = dict([(i, np.random.rand(3,)) for i in range(num_k)])



b = YOLO_kmeans(a, num_k)

colors = np.array(list(map(lambda x: color_dict[x], b[1])))

plt.scatter(a[:,0], a[:,1], c = colors, s  = 0.001)
plt.scatter(b[0][:,0], b[0][:,1], s = 15, c = "black")
plt.savefig("kmeans.jpg")
plt.show()
