#Preprocessing 7

#Due to large overhead with SimpleITK reading the images, converting the sitk images to np arrays, and then the arrays to tensors takes up a lot of overhead and
    #linearly increases epoch time as batch size increases

#This script is for saving the dataset straight as np tensors to avoid all the conversions and not have to load Sitk during training

#In theory it *should* make training take a lot less time 


#Dependencies
import SimpleITK as sitk
import torch 
import torch.nn as nn
import numpy as np
import pathlib
import os
from natsort import natsorted
import json
import time 

from transforms import tensor_transforms
from image import mri_slice
from preprocessing import one_hot

#Set GPU/Cuda Device to run model on
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

start_time = time.time()

json_path = "tensor_data/uncropped_data.json"

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

value_map = one_hot.one_hot_encoding(masks_no = masks_no, masks_array= masks_array)

#Paths 
train_images = pathlib.Path(r"D:/Spider Data Slices/train_image_slices")
train_labels = pathlib.Path(r"D:/Spider Data Slices/train_label_slices")
test_images = pathlib.Path(r"D:/Spider Data Slices/test_image_slices")
test_labels = pathlib.Path(r"D:/Spider Data Slices/test_label_slices")

train_image_tensors = pathlib.Path(r"D:/Spider Data Slices/train_image_tensors")
train_label_tensors = pathlib.Path(r"D:/Spider Data Slices/train_label_tensors")
test_image_tensors = pathlib.Path(r"D:/Spider Data Slices/test_image_tensors")
test_label_tensors =  pathlib.Path(r"D:/Spider Data Slices/test_label_tensors")

train_images_sitk_list = os.listdir(train_images)
train_labels_sitk_list = os.listdir(train_labels)
test_images_sitk_list = os.listdir(test_images)
test_labels_sitk_list = os.listdir(test_labels)

train_images_sitk_list = natsorted(train_images_sitk_list)
train_labels_sitk_list = natsorted(train_labels_sitk_list)
test_images_sitk_list = natsorted(test_images_sitk_list)
test_labels_sitk_list = natsorted(test_labels_sitk_list)

print(len(train_images_sitk_list))
print(len(test_images_sitk_list))


#train set 
for idx in range(0, len(train_images_sitk_list)):

    img_path = train_images.joinpath(train_images_sitk_list[idx])
    label_path = train_labels.joinpath(train_labels_sitk_list[idx])

    tensor_filename = train_images_sitk_list[idx].split('.')[0]

    print("image", tensor_filename)

    image = mri_slice.Mri_Slice(img_path)
    label = mri_slice.Mri_Slice(label_path)

    image_a = image.hu_a
    label_a = label.hu_a

    #value mapping label tensor 0-19
    label_a = value_map(label_a)
        
    image_tensor = torch.from_numpy(image_a)
    label_tensor = torch.from_numpy(label_a)
        
    image_tensor = image_tensor.to(torch.float32)
    label_tensor = label_tensor.to(torch.float32) #techincally these are ints not floats 

    #pad to max resolution of slice in dset 
    image_tensor = tensor_transforms.pad_to_resolution(image_tensor, [row_max, col_max])
    label_tensor = tensor_transforms.pad_to_resolution(label_tensor, [row_max, col_max]) #torch.functional.pad takes care of cropping, no need to call array_transforms.crop_zero()

    #normalise image tensor
    image_tensor = (image_tensor - image_tensor_min) / (image_tensor_max - image_tensor_min)
    #label_tensor = (label_tensor - label_tensor_min) / (label_tensor_max - label_tensor_min) #INTS HERE DO NOT NORMALISE 

    #one hot label, this works only if the range on the label tensors is from 0 to x steps of 1 
    label_tensor = nn.functional.one_hot(label_tensor.long(), num_classes= masks_no) 

    label_tensor = label_tensor.float()
    #print("tensor dims", image_tensor.shape)
    #print("label tensor min clamp",torch.min(label_tensor))
    #print("label tensor max clamp", torch.max(label_tensor))

    image_tensor = image_tensor.unsqueeze(0)
    #label_tensor = label_tensor.unsqueeze(0) 
        
    #print("label before permute shape", label_tensor.shape)
    #permute axes label tensor
    label_tensor = torch.permute(label_tensor, (2, 0, 1))

    #remove backround (0) channel in channel dim of label tensor 
    label_tensor = label_tensor[1:, :, :]     

    image_tensor_path = train_image_tensors.joinpath(f"{tensor_filename}.pt")
    label_tensor_path = train_label_tensors.joinpath(f"{tensor_filename}.pt")

    torch.save(image_tensor, image_tensor_path)
    torch.save(label_tensor, label_tensor_path)

    

#test set
for idx in range(0, len(test_images_sitk_list)):

    img_path = test_images.joinpath(test_images_sitk_list[idx])
    label_path = test_labels.joinpath(test_labels_sitk_list[idx])

    tensor_filename = train_images_sitk_list[idx].split('.')[0]

    print("label", tensor_filename)

    image = mri_slice.Mri_Slice(img_path)
    label = mri_slice.Mri_Slice(label_path)

    image_a = image.hu_a
    label_a = label.hu_a

    #value mapping label tensor 0-19
    label_a = value_map(label_a)
        
    image_tensor = torch.from_numpy(image_a)
    label_tensor = torch.from_numpy(label_a)
        
    image_tensor = image_tensor.to(torch.float32)
    label_tensor = label_tensor.to(torch.float32) #techincally these are ints not floats 

    #pad to max resolution of slice in dset 
    image_tensor = tensor_transforms.pad_to_resolution(image_tensor, [row_max, col_max])
    label_tensor = tensor_transforms.pad_to_resolution(label_tensor, [row_max, col_max]) #torch.functional.pad takes care of cropping, no need to call array_transforms.crop_zero()

    #normalise image tensor
    image_tensor = (image_tensor - image_tensor_min) / (image_tensor_max - image_tensor_min)
    #label_tensor = (label_tensor - label_tensor_min) / (label_tensor_max - label_tensor_min) #INTS HERE DO NOT NORMALISE 

    #one hot label, this works only if the range on the label tensors is from 0 to x steps of 1 
    label_tensor = nn.functional.one_hot(label_tensor.long(), num_classes= masks_no) 

    label_tensor = label_tensor.float()
    #print("tensor dims", image_tensor.shape)
    #print("label tensor min clamp",torch.min(label_tensor))
    #print("label tensor max clamp", torch.max(label_tensor))

    image_tensor = image_tensor.unsqueeze(0)
    #label_tensor = label_tensor.unsqueeze(0) 
        
    #print("label before permute shape", label_tensor.shape)
    #permute axes label tensor
    label_tensor = torch.permute(label_tensor, (2, 0, 1))

    #remove backround (0) channel in channel dim of label tensor 
    label_tensor = label_tensor[1:, :, :]     

    image_tensor_path = test_image_tensors.joinpath(f"{tensor_filename}.pt")
    label_tensor_path = test_label_tensors.joinpath(f"{tensor_filename}.pt")

    torch.save(image_tensor, image_tensor_path)
    torch.save(label_tensor, label_tensor_path)

    

end_time = time.time()

total_time = end_time - start_time

print("total time for converting from sitk to tensor", total_time)