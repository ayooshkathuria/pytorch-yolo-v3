# A Fork of PyTorch Implemation of YOLOv3 to Accomodate Custom Data

**This fork is a work in progress.  It will be noted here when this is ready for broader, more production, use.  Issues are welcome.**

## General Updates/Improvements

* User may bring custom data with a custom number of classes
* Code cleaner and parameterized
* Training has fine-tuning
* For more details on updates and status of this fork, see notes at the bottom of the README.md

_We love you COCO, but we have our own interests now._

This project is a "You Only Look Once" v3 sample using PyTorch, a fork of https://github.com/ayooshkathuria/pytorch-yolo-v3, with updates and improvements specifically for architecture on custom data labeled with VoTT (versus the classic download of VOC or COCO data and pre-existing labels).  This fork allows the user to **bring their own dataset**.

<img src="imgs/id_plumeria_sml.png" width="70%" align="center">

Important Notes
--- 
* This project is a work in progress.
* Training is very sensitive to LR and LR decreases (please be aware and watch out for this).
* The example config files are 2 classes, see below on how to change the numbe of classes.

## Setup

* Install the required Python packages (`pip install -r requirements.txt`).
* Download the [full YOLO v3 (237 MB)](https://pjreddie.com/media/files/yolov3.weights) or [tiny YOLO v3 (33.8 MB)](https://pjreddie.com/media/files/yolov3-tiny.weights) model.  **Fun fact:  this project utilizes the weights originating in Darknet format**.

## Collect and Label Data

1. Use the <a href="https://github.com/Microsoft/VoTT" target="_blank">VoTT</a> labeling tool to create bounding boxes around objects of interest in images and export to YOLO format.  The `data` output folder should be a subdirectory here with the images, labels and pointer file.
2. If you wish to train on all labeled images, make sure they are all in the `train.txt` file (this is read by the `customloader.py`).

## Train Model

### Modifications for Custom

**Filters and Classes**

Ensure the `yolov3-tiny.cfg` or `yolov3.cfg` is set up correctly.  Note, the number of classes will affect the last convolutional layer filter numbers (conv layers before the yolo layer) as well as the yolo layers themselves - so **will need to be modified manually** to suit the needs of the user.

Change the number of classes appropriately (e.g. `classes=2`).

Modify the filter number of the CNN layer directly before each [yolo] layer to be:  `filters=`, then calculate (classes + 5)x3, and place after.  So, if `classes=1` then should be `filters=18`. If `classes=2` then write `filters=21`, and so on.

**Anchors**

The tiny architecture has 6 anchors, whereas, the non-tiny or full sized YOLOv3 architecture has 9 anchors.  These anchors should be manually discovered with `kmeans.py` and specified in the `cfg` file.

**Additional Instructions**

* Create a list of the training images file paths, one per line, called `train.txt` and place it in the `data` folder.  e.g.

`train.txt`
```
data/obj/482133.JPG
data/obj/482128.JPG
data/obj/482945.jpg
data/obj/483153.JPG
data/obj/481427.jpg
data/obj/480836.jpg
data/obj/483522.JPG
data/obj/482535.JPG
data/obj/483510.JPG
```

* To update the number of layers that are tracking gradients for fine-tuning, at the beginning of the `train.py` script, update the default `FINE_TUNE_STOP_LAYER`.

### Run Training

Cmd example:

    python train.py --cfg cfg/yolov3-2class.cfg --weights yolov3.weights --datacfg data/obj.data --lr 0.0005 --unfreeze 2

Usage:

    python train.py --help

## Inference

Here, you will use your trained model for evaluation on test data and a live video analysis.  The folder `runs` is where trained models get saved by default under a date folder.

**Additional Instructions**

* Create a list of the test images file paths, one per line, called `test.txt` and place it in the `data` folder.  e.g.

`test.txt`
```
data/obj/482308.JPG
data/obj/483367.JPG
data/obj/483037.jpg
data/obj/481962.JPG
data/obj/481472.jpg
data/obj/483303.JPG
data/obj/483326.JPG
```

### Evaluation

Cmd example:

    python eval.py --cfg cfg/yolov3-2class.cfg --weights runs/<your trained model>.pth --overlap 0.3

Usage:

    python eval.py --help

### Run Video Detection

Cmd example:

    python live.py --cfg cfg/yolov3-tiny.cfg --weights runs/<your trained model>.pth --datacfg data/obj.data --confidence 0.6

Usage:
    
    python live.py --help

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

Updates to original codebase
---
* [x] Made it possible to bring any image data for object detection with `customloader.py` (using <a href="https://github.com/Microsoft/VoTT" target="_blank">VoTT</a> to label)
* [x] Replace CUDA flag in lieu of the simple `tensor_xyz.to(device)` method
* [x] Fix `customloader.py` to take multiple classes as a parameter in config file (e.g. `yolov3-2class.cfg`)
* [x] Add a custom collate function to `train.py` to detect empty boxes and exclude
* [x] Fix resizing transform by creating a custom `YoloResize` transform called `YoloResizeTransform`
* [x] Add finetuning to the `train.py` script
* [x] Fix the learning rate adjustment to decrease more consistently during training and finetuning
* [x] Created method to find optimal anchor box sizes with `kmeans.py` and a script to temporarily convert labels `scripts/convert_labels.py` (the converted labels are only used for calculating anchor values)
* [x] Ensure this codebase works with full sized YOLOv3 network
---
* [ ] Checkpoint only models with better loss values than previous ones (use checkpoint functionality in PyTorch)
* [ ] Fix `customloader.py` to take custom (as an argument) anchors, anchor numbers and model input dims
* [ ] Ensure `live.py` is correctly drawing bounding boxes
* [ ] Ensure `eval.py` is correctly evaluating predictions
* [ ] flake8 (clean up extra blank lines, long lines, etc.)
* [ ] Remove `*` imports in place of explicit imports
* [ ] Clean up unnecessary params in config files
