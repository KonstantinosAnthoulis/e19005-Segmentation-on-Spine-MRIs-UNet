#Dependencies
import torch
from torch import nn 
from torch.utils.data import Dataset
import SimpleITK as sitk 
import os
from natsort import natsorted 
import json
import numpy as np 

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
print(f"Using {device} device")

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


value_map = one_hot.one_hot_encoding(masks_no = masks_no, masks_array= masks_array)

class SpiderDataset(Dataset):
    def __init__(self, labels_dir, img_dir, transform=None, target_transform=None):
        self.labels_dir = labels_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.labels_dir))

    def __getitem__(self, idx):
        label_dir_list = os.listdir(self.labels_dir)
        image_dir_list = os.listdir(self.img_dir)

        image_dir_list = natsorted(image_dir_list)
        label_dir_list = natsorted(label_dir_list)

        img_path = self.img_dir.joinpath(image_dir_list[idx])
        label_path = self.labels_dir.joinpath(label_dir_list[idx])

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

        #one hot encoding label tensor BUG FIX
        #one hot label, this works only if the range on the label tensors is from 0 to 19 steps of 1 
        label_tensor = nn.functional.one_hot(label_tensor.long(), num_classes= masks_no) #resulting shape is torch.Size([8, 20, 448, 224]

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
        
        #print("tensor shape after removing first channel 2nd dim", label_tensor.shape)

        label_flattened = label_tensor.flatten()
        #print("label tensor unique values", torch.unique(label_flattened))

        image_tensor = image_tensor.to(device)
        label_tensor = label_tensor.to(device)

        #print(image_tensor.shape)
        return image_tensor, label_tensor


