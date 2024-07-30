#For initialising model training

#Dependencies 
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import json 

import SimpleITK as sitk
reader = sitk.ImageFileReader()
reader.SetImageIO("MetaImageIO")

import numpy as np

import os

import pathlib

from natsort import natsorted

from dataset import SpiderDataset
import metric

#Set GPU/Cuda Device to run model on
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

np.random.seed(46)

json_path = "C:/Users/user/Desktop/Spider test/data.json"

#Load tensor parameters from .json
with open(json_path, 'r') as file:
    data = json.load(file)

# Assign each value to a variable
row_max = data["row_max"]
col_max = data["col_max"]
image_tensor_min = data["image_tensor_min"]
image_tensor_max = data["image_tensor_max"]
label_tensor_min = data["label_tensor_min"]
label_tensor_max = data["label_tensor_max"]
masks_no = data["masks_no"]
masks_array = data["masks_array"]

#Directories test Work Desktop 
train_img_slice_dir = pathlib.Path("C:/Users/user/Desktop/Spider test/train_slice_cropped_images")
train_label_slice_dir = pathlib.Path("C:/Users/user/Desktop/Spider test/train_slice_cropped_labels")

test_img_slice_dir = pathlib.Path("C:/Users/user/Desktop/Spider test/test_slice_images")
test_label_slice_dir= pathlib.Path("C:/Users/user/Desktop/Spider test/test_slice_labels")

#Sorting Directories 
image_path = train_img_slice_dir
label_path = train_label_slice_dir

image_dir_list = os.listdir(image_path)
label_dir_list = os.listdir(label_path)

#sort lists
image_dir_list = natsorted(image_dir_list)
label_dir_list = natsorted(label_dir_list)

dirlen = len(image_dir_list)

dummy_train_set = SpiderDataset(train_label_slice_dir, train_img_slice_dir)

dummy_test_set = SpiderDataset(test_label_slice_dir, test_img_slice_dir)

print("train dataset len",dummy_train_set.__len__())
print("test dataset len",dummy_test_set.__len__())

from models import unet 

input_channels = 1 #Hounsfield scale
output_channels = masks_no #one for every class 0-9 vertebrae 10 spinal canal 11-19 ivd
start_filts = 16 #unet filters 
up_mode = 'upsample' #options are either 'upsample' for NN upsampling or 'transpose' for transpose conv

model = unet.UNet(in_channels= input_channels,num_classes=output_channels, start_filts=start_filts, up_mode=up_mode) #testing model hyperparams
model.to(device)
model.to(torch.float32)
#for param in model.parameters():
 #   print(param.device)

#Training Hyperparameters 
epochs = 4 #setting this to 3 epochs per training session takes about 6-8 hours
lr = 0.0001 #0.001 too large 
batchsize = 6 #max on local machine
loss_func = nn.BCEWithLogitsLoss() 
loss_func.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)

print(epochs)
print(lr)
print(batchsize)
print(loss_func)
print(start_filts)
print(up_mode)

print((loss_func))

#Dataloaders
train_dataloader = DataLoader(dummy_train_set, batch_size = batchsize, shuffle=True)

test_dataloader = DataLoader(dummy_test_set, batch_size = batchsize, shuffle=True)

#Accuracy Metrics 
metric_calculator = metric.SegmentationMetrics(average=True, ignore_background=True,activation='sigmoid') 

metric_calculator_binary = metric.BinaryMetrics(activation='sigmoid') #for calculating spinal canal metrics since it's only 1 class

#TODO epoch.py for 1 epoch 