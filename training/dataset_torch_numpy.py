# Dependencies
import torch
from torch import nn 
from torch.utils.data import Dataset
import os
from natsort import natsorted 
import json
import numpy as np  # Replace cupy with numpy
from typing import Union

from transforms import tensor_transforms
from training import one_hot
import albumentations as A


#torch.cuda.set_device(1) #2nd gpu 
device = torch.device("cuda:1")

json_path = "home/kanthoulis/spider/tensor_data/"

# Load tensor parameters from .json
# Define the filename of the JSON file
json_filename = 'tensor_data.json'

# Get the path to the current directory (where the script is located)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Combine the current directory with the JSON filename
#json_path = os.path.join(current_directory, json_filename)
json_path = "/home/kanthoulis/spider/tensor_data/tensor_data.json"

# Load tensor parameters from the .json file
try:
    with open(json_path, 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"The file '{json_filename}' was not found in the same directory as the script.")
except json.JSONDecodeError:
    print("Error decoding the JSON file. Please check the file's content.")
except Exception as e:
    print(f"An error occurred: {e}")

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
    def __init__(self, labels_dir, img_dir, one_hot_labels = False, augmentation = False, transform=None, target_transform=None):
        self.labels_dir = labels_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.one_hot_labels = one_hot_labels
        self.augmentation = augmentation

        # List tensor files
        self.label_dir_list = natsorted(os.listdir(self.labels_dir))
        self.image_dir_list = natsorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.label_dir_list)

    def __getitem__(self, idx):
        #Assuming directories are sorted
        img_path = os.path.join(self.img_dir, self.image_dir_list[idx])
        label_path = os.path.join(self.labels_dir, self.label_dir_list[idx])

        # Load the image and label as Numpy arrays
        image_a = np.load(img_path)
        label_a = np.load(label_path)


        if(self.augmentation == True):
                 # Define the augmentation pipeline for the image
            noise_transform = A.Compose([
                A.GaussNoise(var_limit=(0.0001, 0.01),  mean = 0, p=1.0), # Gaussian Noise
                A.Blur(blur_limit=(3, 7), p=1.0)
            ])
    
             # Define the elastic transform pipeline
            elastic_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                #A.VerticalFlip(p=0.5),
                A.ElasticTransform(alpha=2, sigma=20, p=1.0),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p = 1.0)
            ])
            
            
            """
            noise_transform = A.Compose([
                A.GaussNoise(var_limit=(0.002, 0.05), mean=0, p=1)  # Gaussian Noise  
            ])
    
            # Define the elastic transform pipeline
            elastic_transform = A.ElasticTransform(alpha=25, sigma=4, p=1)
            """
            # Normalize the image before applying Gaussian noise
            image_min = np.min(image_a)
            image_max = np.max(image_a)
    
    
            # Apply elastic deformation to both the image and the label
            # This ensures that both the image and its corresponding label are deformed similarly
            elastic_result = elastic_transform(image=image_a, mask=label_a)
            image_deformed_a = elastic_result["image"]
            label_deformed_a = elastic_result["mask"]
    
            #elastic augs only for now
            """
            # Normalize the image, handling edge cases where image_max == image_min
            if image_max != image_min:
                image_normalised_np = (image_a - image_min) / (image_max - image_min)
            else:
                image_normalised_a = np.zeros_like(image_a)  # Default to zero array
    
            # Apply Gaussian noise only to the image after deformation
            noise_result = noise_transform(image=image_deformed_a)
            image_augmented_a = noise_result["image"]
    
            # Revert normalization to restore the original intensity range of the image
            image_augmented_a = np.clip(image_augmented_a, 0, 1)  # Ensure values are within [0, 1]
            image_augmented_a = image_augmented_a * (image_max - image_min) + image_min
            """
    
            image_a = image_deformed_a
            label_a = label_deformed_a
        
        #if(self.one_hot_labels == True):
        # Apply value mapping to the label
        label_a = value_map(label_a)  
            
        # Convert Numpy arrays directly to torch tensors
        image_tensor = torch.as_tensor(image_a, dtype=torch.float32, device=device).unsqueeze(0)
        if(self.one_hot_labels == True):
            label_tensor = torch.as_tensor(label_a, dtype=torch.float32, device=device)
        else:
            label_tensor = torch.as_tensor(label_a, dtype=torch.float32, device=device).unsqueeze(0) 

        # Pad to max resolution of slice in dataset
        image_tensor = tensor_transforms.pad_to_resolution(image_tensor, [row_max, col_max], value = -1000)
        label_tensor = tensor_transforms.pad_to_resolution(label_tensor, [row_max, col_max], value = 0)

        # Normalize image tensor
        # Normalize image tensor, ensure epsilon to avoid division by zero
        eps = 1e-6
        image_tensor = (image_tensor - image_tensor_min) / (image_tensor_max - image_tensor_min + eps)


        if(self.one_hot_labels == True):
        # One-hot encode the label tensor
            label_tensor = nn.functional.one_hot(label_tensor.long(), num_classes=masks_no).float() #old iteration does 20 dim one hot encoding instead of 19
            # Permute axes of the label tensor
            label_tensor = torch.permute(label_tensor, (2, 0, 1))
            #label_tensor = one_hot_ignore_backround(label_tensor, num_classes=19)
     
        # Ensure tensors are on the correct device
        image_tensor = image_tensor.to(device)
        label_tensor = label_tensor.to(device)

        assert image_tensor.shape[1:] == label_tensor.shape[1:], \
            f"Image and label sizes do not match! Image shape: {image_tensor.shape}, Label shape: {label_tensor.shape}"

        return image_tensor, label_tensor
