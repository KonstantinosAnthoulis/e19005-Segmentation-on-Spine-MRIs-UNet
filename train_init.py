#For initialising model training

#Dependencies 
import torch
from torch import nn
from torch.utils.data import DataLoader
import json 

import SimpleITK as sitk
reader = sitk.ImageFileReader()
reader.SetImageIO("MetaImageIO")

import numpy as np

import os

import pathlib

from natsort import natsorted

from training import dataset
from training import metric
from models import unet

from training import epoch as ep #not to conlfict with var name in loop

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

json_path = "tensor_data/data.json"

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
train_img_slice_dir = pathlib.Path(r"D:/Spider Data Slices/train_cropped_image_slices")
train_label_slice_dir = pathlib.Path(r"D:/Spider Data Slices/train_cropped_label_slices")

test_img_slice_dir = pathlib.Path(r"D:/Spider Data Slices/test_image_slices")
test_label_slice_dir= pathlib.Path(r"D:/Spider Data Slices/test_label_slices")

#Sorting Directories 
image_path = train_img_slice_dir
label_path = train_label_slice_dir

image_dir_list = os.listdir(image_path)
label_dir_list = os.listdir(label_path)

#sort lists
image_dir_list = natsorted(image_dir_list)
label_dir_list = natsorted(label_dir_list)

dirlen = len(image_dir_list)

dummy_train_set = dataset.SpiderDataset(train_label_slice_dir, train_img_slice_dir)

dummy_test_set = dataset.SpiderDataset(test_label_slice_dir, test_img_slice_dir)

print("train dataset len",dummy_train_set.__len__())
print("test dataset len",dummy_test_set.__len__())

input_channels = 1 #Hounsfield scale
output_channels = masks_no - 1 #-1 not to count in backround 
start_filts = 16 #unet filters 
up_mode = 'upsample' #options are either 'upsample' for NN upsampling or 'transpose' for transpose conv

model = unet.UNet(in_channels= input_channels,num_classes=output_channels, start_filts=start_filts, up_mode=up_mode) #testing model hyperparams
model.to(device)
model.to(torch.float32)
#for param in model.parameters():
 #   print(param.device)

#Training Hyperparameters 
epochs = 10
lr = 0.0001 #0.001 too large 
batchsize = 12
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

#Training Loop
    #Training loop and epoch code from official Pytorch documentation link: ++

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#writer = SummaryWriter('runs/spider_seg_unet_epochs={}_lr={}_batchsize={}_loss=BCEWithLogits_startfilts={}_upmode={}'.format(epochs,lr, batchsize,start_filts,up_mode))
writer = SummaryWriter('runs/spider_batchsize_{}_lr_{}_trainses_0_{}'.format(batchsize, lr, timestamp))
epoch_number = 0 #Intial epoch for training 


best_vloss = 1_000_000.

for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = ep.train_one_epoch(epoch_number, writer, optim = optim, loss_func = loss_func, train_dataloader=train_dataloader, model = model,
                               metric_calculator=metric_calculator, metric_calculator_binary=metric_calculator_binary, device = device)
    #print("avg loss in epoch", avg_loss)

    running_vloss = 0.0
    running_vaccu = 0.0
    running_vdice = 0.0

    vert_running_vaccu = 0.0
    vert_running_vdice = 0.0
    
    sc_running_vaccu = 0.0
    sc_running_vdice = 0.0

    ivd_running_vaccu = 0.0
    ivd_running_vdice = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(test_dataloader):
            vinputs, vlabels = vdata

            voutputs = model(vinputs)
            vloss = loss_func(voutputs, vlabels)
            
            vaccu, vdice, vprec, vrecall = metric_calculator(vlabels, voutputs)

            #Vertebrae Metrics (0-9)
            vert_vlabels = vlabels[:, :9, :, :]
            vert_voutputs = voutputs[:, :9, :, :]
            vert_vaccu, vert_vdice, vert_vprec, vert_vrecall = metric_calculator(vert_vlabels, vert_voutputs)

            #Spinal Canal Metrics (10)
            sc_vlabels = vlabels[:, 10, :, :].unsqueeze(1)
            sc_voutputs = voutputs[:, 10, :, :].unsqueeze(1)
            sc_vaccu, sc_vdice, sc_vprec, sv_vspecif, sc_vrecall = metric_calculator_binary(sc_vlabels, sc_voutputs)

            #IVD Metrics (11-19)
            ivd_vlabels = vlabels[:, -9:, :, :]
            ivd_voutputs = voutputs[:, -9:, :, :]
            ivd_vaccu, ivd_vdice, ivd_vprec, ivd_vrecall = metric_calculator(ivd_vlabels, ivd_voutputs)


            running_vloss += vloss
            running_vaccu += vaccu
            running_vdice += vdice

            vert_running_vaccu += vert_vaccu
            vert_running_vdice += vert_vdice

            sc_running_vaccu += sc_vaccu
            sc_running_vdice += vert_vdice

            ivd_running_vaccu += ivd_vaccu
            ivd_running_vdice += ivd_vdice



    avg_vloss = running_vloss / (i + 1)
    avg_vaccu = running_vaccu / (i + 1)
    avg_vdice = running_vdice / (i + 1)

    vert_avg_vaccu = vert_running_vaccu / (i + 1)
    vert_avg_vdice = vert_running_vdice / (i + 1)

    sc_avg_vaccu = sc_running_vaccu / (i + 1)
    sc_avg_vdice = sc_running_vdice / (i + 1)

    ivd_avg_vaccu = ivd_running_vaccu / (i + 1)
    ivd_avg_vdice = ivd_running_vdice / (i + 1)


    #print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    
    writer.add_scalars('Loss/train vs validation',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss},
                    epoch_number + 1)
    
    writer.add_scalars('General/accuracy_validation',
                    {'Accuracy': avg_vaccu},
                    epoch_number + 1)

    writer.add_scalars('General/dice_validation',
                    {'Dice': avg_vdice},
                    epoch_number + 1)
    #vert
    writer.add_scalars('Vertebrae/accuracy_validation',
                    {'Accuracy': vert_avg_vaccu},
                    epoch_number + 1)

    writer.add_scalars('Vertebrae/dice_validation',
                    {'Dice': vert_avg_vdice},
                    epoch_number + 1)
    #spinal canal
    writer.add_scalars('Spinal Canal/accuracy_validation',
                    {'Accuracy': sc_avg_vaccu},
                    epoch_number + 1)

    writer.add_scalars('Spinal Canal/dice_validation',
                    {'Dice': sc_avg_vdice},
                    epoch_number + 1)
    #ivd
    writer.add_scalars('Intervertebral Discs/accuracy_validation',
                    {'Accuracy': ivd_avg_vaccu},
                    epoch_number + 1)

    writer.add_scalars('Intervertebral Discs/dice_validation',
                    {'Dice': ivd_avg_vdice},
                    epoch_number + 1)
    
    writer.flush()
    
    #Change path to save model accordingly     
    model_path = 'C:/Users/kosta/Desktop/Spider Models Optims/spider_model_{}_{}'.format(timestamp, epoch_number)
    
    torch.save({'model_dict': model.state_dict(), 'optimizer_dict': optim.state_dict()}, model_path)
        
    epoch_number += 1