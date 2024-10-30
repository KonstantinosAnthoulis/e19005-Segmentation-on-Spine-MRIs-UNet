#For initialising model training

#Dependencies 
import torch
from torch import nn
from torch.utils.data import DataLoader
import json 
import sys

import SimpleITK as sitk
reader = sitk.ImageFileReader()
reader.SetImageIO("MetaImageIO")

import numpy as np

import os

import pathlib

from natsort import natsorted

from training import dataset_torch_cupy as dataset #change accordingly 
from training import metric
from models import unet

from training import epoch as ep #not to conlfict with var name in loop


print(torch.cuda.is_available())


#Set GPU/Cuda Device to run model on
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



torch.manual_seed(46)

json_path = "tensor_data/augmented_data.json"

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

#Tensor Directories 
train_img_slice_dir = pathlib.Path(r"D:/Spider Data/colab_train_image_numpy")
train_label_slice_dir = pathlib.Path(r"D:/Spider Data/colab_train_label_numpy")

test_img_slice_dir = pathlib.Path(r"D:/Spider Data/colab_test_image_numpy")
test_label_slice_dir= pathlib.Path(r"D:/Spider Data/colab_test_label_numpy")

#Sorting Directories 
image_path = train_img_slice_dir
label_path = train_label_slice_dir

image_dir_list = os.listdir(image_path)
label_dir_list = os.listdir(label_path)

#sort lists
image_dir_list = natsorted(image_dir_list)
label_dir_list = natsorted(label_dir_list)

dirlen = len(image_dir_list)

dummy_train_set = dataset.SpiderDatasetCupy(train_label_slice_dir, train_img_slice_dir)

dummy_test_set = dataset.SpiderDatasetCupy(test_label_slice_dir, test_img_slice_dir)

print("train dataset len",dummy_train_set.__len__())
print("test dataset len",dummy_test_set.__len__())

input_channels = 1 #Hounsfield scale
output_channels = masks_no - 1 #-1 not to count in backround 
depth = 3
start_filts = 16 #unet filters 

up_mode = 'upsample' #options are either 'upsample' for NN upsampling or 'transpose' for transpose conv

#up_mode = 'transpose'


model = unet.UNet(in_channels= input_channels,num_classes=output_channels, depth= depth, start_filts=start_filts, up_mode=up_mode) #testing model hyperparams
model.to(device)
model.to(torch.float32)
#for param in model.parameters():
 #   print(param.device)

#Training Hyperparameters 
epochs = 1

lr = 0.0001
batchsize = 4
loss_func = nn.BCEWithLogitsLoss() 
loss_func.to(device)

#optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1) #testing adamW adam was overfitting
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

timestamp = datetime.now().strftime('%Y%m%d_%H')
#writer = SummaryWriter('runs/spider_seg_unet_epochs={}_lr={}_batchsize={}_loss=BCEWithLogits_startfilts={}_upmode={}'.format(epochs,lr, batchsize,start_filts,up_mode))
writer = SummaryWriter('runs/spider_test_{}'.format(timestamp))
epoch_number = 0 #Intial epoch for training 


best_vloss = 1_000_000.

for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = ep.train_one_epoch(epoch_number, writer, optim = optim, loss_func = loss_func, train_dataloader=train_dataloader, model = model,
                               metric_calculator=metric_calculator, metric_calculator_binary=metric_calculator_binary, device = device)
    #print("avg loss in epoch", avg_loss)
    '''
    running_accu = 0.0
    running_dice = 0.0

    vert_running_accu = 0.0
    vert_running_dice = 0.0

    sc_running_accu = 0.0
    sc_running_dice = 0.0

    ivd_running_accu = 0.0
    ivd_running_dice = 0.0
    '''
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
        '''
        #training metrics on fully trained model up to that epoch
        for i, data in enumerate(train_dataloader):
            
            print("valuation training set batch", i)

            inputs, labels = data
            outputs = model(inputs)

            accu, dice, prec, recall = metric_calculator(labels, outputs)

            #Vertebrae metrics (0-9)
            vert_labels = labels[:, :9, :, :]
            vert_outputs = outputs[: ,:9, :, :]
            vert_accu, vert_dice, vert_prec, vert_recall = metric_calculator(vert_labels, vert_outputs)

            #Spinal Canal Metrics (10)
            sc_labels = labels[: , 10, :, :].unsqueeze(1)
            sc_outputs = outputs[:, 10, :, :].unsqueeze(1)
            sc_accu, sc_dice, sc_prec, sc_specif, sc_recall = metric_calculator_binary(sc_labels, sc_outputs)

            #IVD metrics (11-19)
            ivd_labels = labels[:, -9:, : ,:]
            ivd_outputs = outputs[:, -9:, :, :]
            ivd_accu, ivd_dice, ivd_prec, ivd_recall = metric_calculator(ivd_labels, ivd_outputs)

            running_accu += accu
            running_dice += dice

            vert_running_accu += vert_accu
            vert_running_dice += vert_dice

            sc_running_accu += sc_accu
            sc_running_dice += vert_dice

            ivd_running_accu += ivd_accu
            ivd_running_dice += ivd_dice
        '''

        #eval metrics
        for j, vdata in enumerate(test_dataloader):

            print("valuation test set batch", j)
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

    '''
    avg_accu = running_accu / (i + 1)
    avg_dice = running_dice / (i + 1)

    vert_avg_accu = vert_running_accu / (i + 1)
    vert_avg_dice = vert_running_dice / (i + 1)

    sc_avg_accu = sc_running_accu / (i + 1)
    sc_avg_dice = sc_running_dice / (i + 1)

    ivd_avg_accu = ivd_running_accu / (i + 1)
    ivd_avg_dice = ivd_running_dice / (i + 1)
    '''

    avg_vloss = running_vloss / (j + 1)
    avg_vaccu = running_vaccu / (j + 1)
    avg_vdice = running_vdice / (j + 1)

    vert_avg_vaccu = vert_running_vaccu / (j + 1)
    vert_avg_vdice = vert_running_vdice / (j + 1)

    sc_avg_vaccu = sc_running_vaccu / (j + 1)
    sc_avg_vdice = sc_running_vdice / (j + 1)

    ivd_avg_vaccu = ivd_running_vaccu / (j + 1)
    ivd_avg_vdice = ivd_running_vdice / (j + 1)


    #print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    
    writer.add_scalars('Loss/validation',
                    { 'Validation' : avg_vloss},
                    epoch_number + 1)
    
    writer.add_scalars('General/accuracy_valid',
                    {'Validation': avg_vaccu},
                    epoch_number + 1)

    writer.add_scalars('General/dice_valid',
                    {'Validation': avg_vdice},
                    epoch_number + 1)
    #vert
    writer.add_scalars('Vertebrae/accuracy_valid',
                    {'Validation': vert_avg_vaccu},
                    epoch_number + 1)

    writer.add_scalars('Vertebrae/dice_valid',
                    {'Validation': vert_avg_vdice},
                    epoch_number + 1)
    #spinal canal
    writer.add_scalars('Spinal Canal/accuracy_valid',
                    {'Validation': sc_avg_vaccu},
                    epoch_number + 1)

    writer.add_scalars('Spinal Canal/dice_valid',
                    {'Validation': sc_avg_vdice},
                    epoch_number + 1)
    #ivd
    writer.add_scalars('Intervertebral Discs/accuracy_valid',
                    {'Validation': ivd_avg_vaccu},
                    epoch_number + 1)

    writer.add_scalars('Intervertebral Discs/dice_valid',
                    {'Validation': ivd_avg_vdice},
                    epoch_number + 1)
    
    writer.flush()
    
    #Change path to save model accordingly     
    model_path = 'C:/Users/kosta/Desktop/Spider Optims Final/spider_uncropped_{}'.format(epoch_number)
    
    torch.save({'model_dict': model.state_dict(), 'optimizer_dict': optim.state_dict()}, model_path)
        
    epoch_number += 1