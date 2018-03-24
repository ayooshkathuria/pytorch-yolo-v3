# A PyTorch implementation of a YOLO v2 Object Detector

This repository contains code for a object detector based on [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf), implementedin PyTorch. The code is based on the official code of [YOLO v2](https://github.com/pjreddie/darknet), as well as a PyTorch 
port of the original code, by [marvis](https://github.com/marvis/pytorch-yolo2). One of the goals of this code is to improve
upon the original port by removing redundant parts of the code (The official code is basically a fully blown deep learning 
library, and includes stuff like sequence models, which are not used in YOLO). I've also tried to keep the code minimal, and 
document it as well as I can. 

As of now, the code only contains the detection module, but you should expect the training module soon. :) 

## Running the detector

### On single or multiple images

Clone, and `cd` into the repo directory. Then, you have two variants of the detector, one that has been trained on PASCAL VOC 
data (faster, but less accurate and recognises only 20 object categories), or the one trained on COCO (Slower, more accurate, 
detects 80 categories). 

For example, running the detector trained on PASCAL VOC download [here](https://pjreddie.com/media/files/yolo-voc.weights), and place 
the weights file into your repo directory. Or, you could just type (if you're on Linux)


```
wget https://pjreddie.com/media/files/yolo-voc.weights 
python detect.py --images imgs --det det --dataset pascal 
```
For running with one trainined on coco, download [this](https://pjreddie.com/media/files/yolo-voc.weights) weightsfile, and run
the code with `--dataset` flag set to `coco`.

`--images` flag defines the directory to load images from, or a single image file (it will figure it out), and `--det` is the directory
to save images to. Other setting such as batch size, object threshold confidence can be tweaked with flags that can be looked up with 

```
python detect.py -h
```
### On Video
For this, you should run the file, video_demo.py with --video flag specifying the video file. The video file should be in .avi format
since openCV only accepts OpenCV as the input format. 

```
python video_demo.py --video video.avi --dataset pascal
```

Tweakable settings can be seen with -h flag. 

To speed video inference, you can try using the video_demo_half.py file instead which does all the inference with 16-bit half 
precision floats instead of 32-bit float. I haven't seen big improvements, but I attribute that to having an older card 
(Tesla K80, Kepler arch, getting around 22 fps with PASCAL). If you have one of cards with fast float16 support, try it out, and if possible, benchmark it. 

### On a Camera
Same as video module, but you don't have to specify the video file since feed will be taken from your camera. To be precise, 
feed will be taken from what the OpenCV, recognises as camera 0. You can tweak this setting in code. 

You'll have to download [Tiny-yolo weightsfile](https://pjreddie.com/media/files/tiny-yolo-voc.weights) in your repo folder. 
Excpect higher FPS and lower accuracy. 

```
python cam_demo.py
```
You can easily tweak the code to use different weightsfiles, available at [yolo website](https://pjreddie.com/darknet/yolo/)

## Coming Soon

Training module should arrive soon. 
