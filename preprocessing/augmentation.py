#PREPROCESSING SOMETHING

#Data Augmentation

#Generate X amount of slices from 1 slice to further increase the dataset size 

#Dependencies 
import albumentations as a
import SimpleITK as sitk
import numpy as np
import pathlib 
import os 
import sys
from natsort import natsorted

#Training Data Paths, only applying augmentation on training data 
train_img_slice_dir = pathlib.Path(r"D:/Spider Data/train_image_slices")
train_label_slice_dir = pathlib.Path(r"D:/Spider Data/train_label_slices")

#Get lists of the files in the directories 
image_train_dir_list = os.listdir(train_img_slice_dir) 
label_train_dir_list = os.listdir(train_label_slice_dir)

#Sort lists just to be safe
image_train_dir_list = natsorted(image_train_dir_list)
label_train_dir_list = natsorted(label_train_dir_list)

train_len = len(image_train_dir_list)
test_len = len(label_train_dir_list)

if (train_len != test_len):
    print("Error: Directories aren't of equal size")
    sys.exit()

dirlen = train_len

transform = a.Compose([
    #TODO add augmentations here 

])



for idx in range(0, dirlen):

    image_path = train_img_slice_dir.joinpath(image_train_dir_list[idx])
    label_path = train_label_slice_dir.joinpath(label_train_dir_list[idx])

    image_sitk = sitk.ReadImage(image_path)
    label_sitk = sitk.ReadImage(label_path)

    image_np = sitk.GetArrayFromImage(image_sitk)
    label_np = sitk.GetArrayFromImage(label_sitk)

    break


