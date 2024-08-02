#Dependencies 
import numpy as np 
from natsort import natsorted
import pathlib
import os
import SimpleITK as sitk
import sys

#Import MRI Slice class
from image import mri_slice
#Array transforms for cropping
from transforms import array_transforms

#Directories of uncropped slices train set 
#Change as needed 
#train_img_slice_dir = pathlib.Path(r"your/path/here")
#train_label_slice_dir = pathlib.Path(r"")

train_img_slice_dir = pathlib.Path(r"spider_toy_dset_slices/train_image_slices")
train_label_slice_dir = pathlib.Path(r"spider_toy_dset_slices/train_label_slices")

#Directories to write the cropped slices to 
#Change as needed 
#train_cropped_img_slice_dir = pathlib.Path(r"")
#train_cropped_label_slice_dir = pathlib.Path(r"")

train_cropped_img_slice_dir = pathlib.Path(r"spider_toy_dset_slices/train_image_cropped_slices")
train_cropped_label_slice_dir = pathlib.Path(r"spider_toy_dset_slices/train_label_cropped_slices")

image_path = train_img_slice_dir
label_path = train_label_slice_dir

image_dir_list = os.listdir(image_path)
label_dir_list = os.listdir(label_path)

#Sort directories so that each image has its corresponding label on same idx 
image_dir_list = natsorted(image_dir_list)
label_dir_list = natsorted(label_dir_list)

dirlen_image = len(os.listdir(image_path))
dirlen_label = len(os.listdir(label_path))

print("train images count", dirlen_image)

if(dirlen_image != dirlen_label):
  print("Error: image directory has", dirlen_image, "images not equal to label directory", dirlen_label, "images")
  sys.exit()
  
#After check pass the value to a single var since they're both the same value 
dirlen = dirlen_image

for idx in range(0, dirlen):

    #Get image and corresponding label paths
    img_path = image_path.joinpath(image_dir_list[idx])
    lbl_path = label_path.joinpath(label_dir_list[idx])#first part before joinpath is pathlib.Path, second part is the directory of the file 
    
    print(image_dir_list[idx])

    #Read image and label
    image = mri_slice.Mri_Slice(img_path)
    label = mri_slice.Mri_Slice(lbl_path)

    #Get arrays
    image_a = image.hu_a
    label_a = label.hu_a

    #Crop both arrays based on ROI of label 
    image_cropped_a, label_cropped_a = array_transforms.crop_zero(image_a, label_a)

    print("array shape", image_cropped_a.shape)

    #Get cropped images from arrays  
    image_cropped_mha = sitk.GetImageFromArray(image_cropped_a)
    label_cropped_mha = sitk.GetImageFromArray(label_cropped_a)

    print(train_cropped_img_slice_dir.joinpath(image_dir_list[idx]))

    image_cropped_path = train_cropped_img_slice_dir.joinpath(image_dir_list[idx])
    label_cropped_path = train_cropped_label_slice_dir.joinpath(label_dir_list[idx])

    #Write Images
    sitk.WriteImage(image_cropped_mha, image_cropped_path)
    sitk.WriteImage(label_cropped_mha, label_cropped_path)


 

    

