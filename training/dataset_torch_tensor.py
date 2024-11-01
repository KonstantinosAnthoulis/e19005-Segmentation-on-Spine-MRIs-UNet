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


value_map = one_hot.value_map(masks_no = masks_no, masks_array= masks_array)

class SpiderDatasetTorchTensor(Dataset):
    def __init__(self, labels_dir, img_dir, transform=None, target_transform=None):
        self.labels_dir = labels_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # List tensor files
        self.label_dir_list = natsorted(os.listdir(self.labels_dir))
        self.image_dir_list = natsorted(os.listdir(self.img_dir))

        # Create a mapping of image files to label files
        self.image_to_label_map = self._create_image_to_label_map()

    def _create_image_to_label_map(self):
        # Create a dictionary to map each image to its corresponding label
        image_to_label = {}
        
        # Go through each image and determine its base label
        for img_name in self.image_dir_list:
            # Split the image name by underscores
            parts = img_name.split('_')
            
            # Extract the key parts for matching with the label
            base_label = '_'.join(parts[:3])
            if len(parts) > 3 and parts[3] == 'f':  # Handle flipped cases
                base_label += '_f'
            
            # Store in the mapping
            image_to_label[img_name] = base_label
        
        return image_to_label

    def __len__(self):
        return len(self.image_dir_list)

    def __getitem__(self, idx):
        img_name = self.image_dir_list[idx]
        label_name = self.image_to_label_map.get(img_name)

        if not label_name:
            raise ValueError(f"No corresponding label found for image {img_name}")

        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.labels_dir, label_name)

        # Check if the image file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Check if the label file exists
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        # Load images using SimpleITK
        image_sitk = sitk.ReadImage(img_path)
        label_sitk = sitk.ReadImage(label_path)

        image_np = sitk.GetArrayFromImage(image_sitk)
        label_np = sitk.GetArrayFromImage(label_sitk)

        image_tensor = torch.tensor(image_np, dtype=torch.float32)
        label_tensor = torch.tensor(label_np, dtype=torch.float32)

        if self.transform:
            image_tensor = self.transform(image_tensor)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)

        return image_tensor, label_tensor



