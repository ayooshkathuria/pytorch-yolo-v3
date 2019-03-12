import cv2
import numpy as np
import torch
import torch.onnx
import onnx
from onnx import numpy_helper

from darknet import Darknet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


input_dim = 416
img = cv2.imread("imgs/dog.jpg")
img = cv2.resize(img, (input_dim, input_dim)) 
img_ =  img[:,:,::-1].transpose((2,0,1))
img_ = img_[np.newaxis,:,:,:]/255.0
img_ = torch.from_numpy(img_).float()

# If GPU/CUDA available device with transform input
img_ = img_.to(device)

# Set up the neural network
print("Loading network.....")
model = Darknet('cfg/yolov3-tiny-2class.cfg')
model.load_weights('runs/epoch94-bs1-loss2.0198988914489746.pth')
print("Network successfully loaded")

# If GPU/CUDA is available convert model to this form
model = model.to(device)

# Export to ONNX format
torch.onnx.export(model, img_, "from_torch_yolov3_tiny.onnx")

# Test load model back in with onnx
print("Test loading model with ONNX")
model = onnx.load('from_torch_yolov3_tiny.onnx')
print("Test successful!")
