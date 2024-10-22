#PREPROCESSING 7

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

value_map = one_hot.one_hot_encoding(masks_no = masks_no, masks_array= masks_array)

#Paths 
'''
train_images = pathlib.Path(r"D:/Spider Data/train_image_augmented_slices")
train_labels = pathlib.Path(r"D:/Spider Data/train_label_augmented_slices")
test_images = pathlib.Path(r"D:/Spider Data/test_image_slices")
test_labels = pathlib.Path(r"D:/Spider Data/test_label_slices")
'''

train_images = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/train_augmented_image_slices")
train_labels = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/train_augmented_label_slices")
test_images = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/test_image_slices")
test_labels = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/test_label_slices")

'''
train_image_numpy = pathlib.Path(r"D:/Spider Data/train_image_augmented_tensors")
train_label_numpy = pathlib.Path(r"D:/Spider Data/train_label_augmented_tensors")
test_image_numpy = pathlib.Path(r"D:/Spider Data/test_image_tensors")
test_label_numpy =  pathlib.Path(r"D:/Spider Data/test_label_tensors")
'''

train_image_numpy = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/train_image_numpy")
train_label_numpy = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/train_label_numpy")
test_image_numpy = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/test_image_numpy")
test_label_numpy = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/test_label_numpy")

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

# Loop for processing images
for idx in range(0, len(train_images_sitk_list)):
    img_path = train_images.joinpath(train_images_sitk_list[idx])
    numpy_filename = train_images_sitk_list[idx].split('.')[0]

    print("Processing image:", numpy_filename)

    # Load the image and get the image array
    image = mri_slice.Mri_Slice(img_path)
    image_a = image.hu_a

    # Save the processed image tensor
    image_numpy_path = train_image_numpy.joinpath(f"{numpy_filename}.npy")
    np.save(image_numpy_path, image_a)

# Loop for processing labels
for idx in range(0, len(train_labels_sitk_list)):
    label_path = train_labels.joinpath(train_labels_sitk_list[idx])
    numpy_filename = train_labels_sitk_list[idx].split('.')[0]

    print("Processing label:", numpy_filename)

    # Load the label and get the label array
    label = mri_slice.Mri_Slice(label_path)
    label_a = label.hu_a

    # Save the processed label tensor
    label_numpy_path = train_label_numpy.joinpath(f"{numpy_filename}.npy")
    np.save(label_numpy_path, label_a)


#test set
for idx in range(0, len(test_images_sitk_list)):

    img_path = test_images.joinpath(test_images_sitk_list[idx])
    label_path = test_labels.joinpath(test_labels_sitk_list[idx])

    numpy_filename = train_images_sitk_list[idx].split('.')[0]

    print("label", numpy_filename)

    image = mri_slice.Mri_Slice(img_path)
    label = mri_slice.Mri_Slice(label_path)

    image_a = image.hu_a
    label_a = label.hu_a

    image_numpy_path = test_image_numpy.joinpath(f"{numpy_filename}.npy")
    label_numpy_path = test_label_numpy.joinpath(f"{numpy_filename}.npy")

    np.save(image_numpy_path, image_a)
    np.save(label_numpy_path, label_a)

    

end_time = time.time()

total_time = end_time - start_time

print("total time for converting from sitk to numpy using np", total_time)