# Dependencies
import torch
from torch import nn 
from torch.utils.data import Dataset
import SimpleITK as sitk 
import os
from natsort import natsorted 
import json
import numpy as np  # Replace cupy with numpy

from transforms import tensor_transforms
from image import mri_slice
from preprocessing import one_hot

# Set GPU/Cuda Device to run model on
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

json_path = "tensor_data/augmented_data.json"

# Load tensor parameters from .json
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

# Use numpy version of value_map for one-hot encoding
value_map = one_hot.value_map(masks_no=masks_no, masks_array=masks_array)
assert callable(value_map)

class SpiderDatasetNumpy(Dataset):
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
        img_path = os.path.join(self.img_dir, self.image_dir_list[idx])
        label_path = os.path.join(self.labels_dir, self.label_dir_list[idx])

        # Load the image and label as Numpy arrays
        image_a = np.load(img_path)
        label_a = np.load(label_path)

        # Apply value mapping to the label
        label_a = value_map(label_a)  # Assuming one_hot_encoding is defined in one_hot module
    
        # Convert Numpy arrays directly to torch tensors
        image_tensor = torch.as_tensor(image_a, dtype=torch.float32, device=device).unsqueeze(0)
        label_tensor = torch.as_tensor(label_a, dtype=torch.float32, device=device)

        # Pad to max resolution of slice in dataset
        image_tensor = tensor_transforms.pad_to_resolution(image_tensor, [row_max, col_max])
        label_tensor = tensor_transforms.pad_to_resolution(label_tensor, [row_max, col_max])

        # Normalize image tensor
        image_tensor = (image_tensor - image_tensor_min) / (image_tensor_max - image_tensor_min)

        # One-hot encode the label tensor
        label_tensor = nn.functional.one_hot(label_tensor.long(), num_classes=masks_no).float()

        # Permute axes of the label tensor
        label_tensor = torch.permute(label_tensor, (2, 0, 1))

        # Remove the background (0) channel
        label_tensor = label_tensor[1:, :, :]

        # Ensure tensors are on the correct device
        image_tensor = image_tensor.to(device)
        label_tensor = label_tensor.to(device)

        # Ensure image and label sizes match
        assert image_tensor.shape[1:] == label_tensor.shape[1:], "Image and label sizes do not match!"

        return image_tensor, label_tensor
