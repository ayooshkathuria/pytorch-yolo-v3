# A Fork of PyTorch Implemation of YOLOv3 to Accomodate Custom Data

_We love you COCO, but we have our own interests now._

This project is a "You Only Look Once" v3 sample using PyTorch, a fork of https://github.com/ayooshkathuria/pytorch-yolo-v3, with updates and improvements specifically for the Tiny architecture on custom data labeled with VoTT (versus the classic download of VOC or COCO data and labels).  This fork allows the user to create their own dataset.

<img src="imgs/id_plumeria_sml.png" width="70%" align="center">

Note:  This project is a work in progress and is based upon a research effort.

## Setup

* Install the required Python packages (`pip install -r requirements.txt`).
* Download the [full YOLO v3 (237 MB)](https://pjreddie.com/media/files/yolov3.weights) or [tiny YOLO v3 (33.8 MB)](https://pjreddie.com/media/files/yolov3-tiny.weights) model.

## Collect and Label Data

1. Use the [VoTT](https://github.com/Microsoft/VoTT) labeling tool to create bounding boxes around objects of interest in images and export to YOLO format.  The `data` output folder should be a subdirectory here with the images, labels and pointer file.
2. If you wish to train on all labeled images, make sure they are all in the `train.txt` file (this is read by the `customloader.py`).

## Train

Ensure the `yolov3-tiny.cfg` is set up to train (see first lines of file).  Note, the number of classes will affect the last convolutional layer filter numbers (conv layers before the yolo layer) as well as the yolo layers themselves - so will need to be modified manually to suit the needs of the user.  The tiny architecture has 6 anchors, whereas, the non-tiny or full sized YOLOv3 architecture has 9 anchors.  These anchors should be manually discovered with `kmeans.py` and specified in the `cfg` file. 

Cmd:

    python train.py --cfg cfg/yolov3-tiny.cfg --weights yolov3-tiny.weights --datacfg data/obj.data

Usage:

    python train.py --help

## Demo

Here, you will use your trained model in a live video feed.  Ensure the `yolov3-tiny.cfg` is set up to test (see first lines of file).  `runs` is where trained models get saved by default.

Cmd:

    python video_demo.py --cfg cfg/yolov3-tiny.cfg --weights runs/<your trained model>.weights --datacfg data/obj.data --confidence 0.6

Usage:
    
    python video_demo.py --help


## Updates/Improvements

* Custom data possibility
* Clean up of several portions of code and generalizing/parameterizing

## Helpful Definitions

- YOLOv3:  You Only Look Once v3.  Improvments over v1, v2 and YOLO9000 which include [Ref](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b):
  - Predicts more bounding boxes per image (hence a bit slower)
  - Detections at 3 scales
  - Addressed issue of detecting small objects
  - New loss function (cross-entropy replaces squared error terms)
  - Can perform multi-label classification (no more mutually exclusive labels)
  - Performance on par with other architectures (a bit faster than SSD, even)
- Tiny-YOLOv3:  A reduced network architecture for smaller models designed for mobile, IoT and edge device scenarios
- Anchors:  There are 5 anchors per box.  The anchor boxes are designed for a specific dataset using K-means clustering, i.e., a custom dataset must use K-means clustering to generate anchor boxes.  It does not assume the aspect ratios or shapes of the boxes. [Ref](https://medium.com/@vivek.yadav/part-1-generating-anchor-boxes-for-yolo-like-network-for-vehicle-detection-using-kitti-dataset-b2fe033e5807)
- Loss, `loss.backward()` and `nn.MSELoss` (for loss confidence):  Mean Squared Error
- IOU:  intersection over union between predicted bounding boxes and ground truth boxes

**The original YOLOv3 paper by Joseph Redmon and Ali Farhadi:  https://arxiv.org/pdf/1804.02767.pdf**
