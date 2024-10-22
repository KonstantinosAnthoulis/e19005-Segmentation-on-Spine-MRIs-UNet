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

class SpiderDatasetSITK(Dataset):
    def __init__(self, labels_dir, img_dir, transform=None, target_transform=None):
        self.labels_dir = labels_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # List tensor files
        self.label_dir_list = natsorted(os.listdir(self.labels_dir))
        self.image_dir_list = natsorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.label_dir_list)

    def __getitem__(self, idx):
        # Get the paths for the image and label tensors
        img_path = os.path.join(self.img_dir, self.image_dir_list[idx])
        label_path = os.path.join(self.labels_dir, self.label_dir_list[idx])

        # Load the image and get the image array
        image = mri_slice.Mri_Slice(img_path)
        label = mri_slice.Mri_Slice(label_path)

        image_a = image.hu_a
        label_a = label.hu_a

        # Convert to tensor and process
        image_tensor = torch.from_numpy(image_a).to(torch.float32)

        # Pad to max resolution of slice in dataset
        image_tensor = tensor_transforms.pad_to_resolution(image_tensor, [row_max, col_max])

        # Normalize image tensor
        image_tensor = (image_tensor - image_tensor_min) / (image_tensor_max - image_tensor_min)

        # Add a channel dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        # Map label values
        label_a = value_map(label_a)
        
        # Convert to tensor and process
        label_tensor = torch.from_numpy(label_a).to(torch.float32)  # technically these are ints not floats

        # Pad to max resolution of slice in dataset
        label_tensor = tensor_transforms.pad_to_resolution(label_tensor, [row_max, col_max])

        # One-hot encode the label tensor
        label_tensor = nn.functional.one_hot(label_tensor.long(), num_classes=masks_no).float()

        # Permute axes of the label tensor
        label_tensor = torch.permute(label_tensor, (2, 0, 1))

        # Remove the background (0) channel
        label_tensor = label_tensor[1:, :, :]        

        # Apply any transformations if needed
        if self.transform:
            image_tensor = self.transform(image_tensor)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)

        # Ensure tensors are on the correct device
        image_tensor = image_tensor.to(device)
        label_tensor = label_tensor.to(device)

        return image_tensor, label_tensor


