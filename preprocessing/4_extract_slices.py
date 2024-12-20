#PREPROCESSING 4

#Dependencies 
import SimpleITK as sitk
reader = sitk.ImageFileReader()
reader.SetImageIO("MetaImageIO")
import numpy as np
import os
import pathlib
from natsort import natsorted
import sys

#Import array transforms for cropping
from transforms import array_transforms
#import Mri class
from image import mri 

#MRI (.mha) file directories, replace paths as needed 
#train_img_dir = pathlib.Path(r"your/path/here")
#train_label_dir = pathlib.Path(r"")
#test_img_dir = pathlib.Path(r"")
#test_label_dir= pathlib.Path(r"")

#train_img_dir = pathlib.Path(r"D:/Spider Data/train_images")
#train_label_dir = pathlib.Path(r"D:/Spider Data/train_labels")
#test_img_dir = pathlib.Path(r"D:/Spider Data/test_images")
#test_label_dir = pathlib.Path(r"D:/Spider Data/test_labels")

train_img_dir = pathlib.Path(r"D:/Spider Data/train_images")
train_label_dir = pathlib.Path(r"D:/Spider Data/train_labels")
test_img_dir = pathlib.Path(r"D:/Spider Data/test_images")
test_label_dir = pathlib.Path(r"D:/Spider Data/test_labels")


#Directories to extract the 2D slices from the 3D images, replace paths as needed 
#NOTE: be careful to the paths set because the generated images will take up A LOT of space 
#train_img_slice_dir = pathlib.Path(r"your/path/here")
#train_label_slice_dir = pathlib.Path(r"")
#test_img_slice_dir = pathlib.Path(r"")
#test_label_slice_dir= pathlib.Path(r"")

train_img_slice_dir = pathlib.Path(r"D:/Spider Data/train_image_slices")
train_label_slice_dir = pathlib.Path(r"D:/Spider Data/train_label_slices")
test_img_slice_dir = pathlib.Path(r"D:/Spider Data/test_image_slices")
test_label_slice_dir= pathlib.Path(r"D:/Spider Data/test_label_slices")

#train_img_slice_dir = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/train_image_slices")
#train_label_slice_dir = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/train_label_slices")
#test_img_slice_dir = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/test_image_slices")
#test_label_slice_dir = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/test_label_slices")


#Get lists of the files in the directories 
image_train_dir_list = os.listdir(train_img_dir) 
label_train_dir_list = os.listdir(train_label_dir)
image_test_dir_list = os.listdir(test_img_dir) 
label_test_dir_list = os.listdir(test_label_dir)

#Sort the lists using natsort 
    # for sorting to format: 1_t1.mha, 1_t2.mha, 2_t1.mha ...so on
image_train_dir_list = natsorted(image_train_dir_list)
label_train_dir_list = natsorted(label_train_dir_list)
image_test_dir_list = natsorted(image_test_dir_list)
label_test_dir_list = natsorted(label_test_dir_list)

#Checking for same length for corresponding image/label lists on train/test
image_train_dirlen = len(image_train_dir_list)
label_train_dirlen = len(label_train_dir_list)
image_test_dirlen = len(image_test_dir_list)
label_test_dirlen = len(label_test_dir_list)

#sys.exit on length mismatch
if(image_train_dirlen != label_train_dirlen):
    sys.exit("Error: Training directories don't have the same amount of images")

if(image_test_dirlen != label_test_dirlen):
    sys.exit("Error: Validation directories don't have the same amount of images")

#Continuing after checks assign lengths to vars for iterating through each directory
train_dirlen = image_train_dirlen
test_dirlen = image_test_dirlen

print("train dirlen", train_dirlen)
print("test dirlen", test_dirlen)

#NOTE: iterating 2 loops, one for training data one for testing data, due to different lengths and pre-processing on training data 

#Extracting slices for TRAINING images/labels
for idx in range(0, train_dirlen):
    img_path = train_img_dir.joinpath(image_train_dir_list[idx])
    label_path = train_label_dir.joinpath(label_train_dir_list[idx])#first part before joinpath is pathlib.Path, second part is the directory of the file 

    #Get 3D array after pre-processing
    image = mri.Mri(img_path, is_label= False, is_train_set= True)
    label = mri.Mri(label_path, is_label= True, is_train_set= True) 

    #Copy
    image_a = image.hu_a
    label_a = label.hu_a
    print("train:", idx)
    
    #Remove slices with no corresponding mask in label 
    image_a, label_a = array_transforms.remove_empty_slices(image_a, label_a)
    
    

    #Extract slices after processing to corresponding directories 
    array_transforms.extract_slices(image_a, image_train_dir_list[idx], train_img_slice_dir) 
    array_transforms.extract_slices(label_a, label_train_dir_list[idx], train_label_slice_dir) 
    

#Extracting slices for TEST images/labels
for idx in range(0, test_dirlen):
    img_path = test_img_dir.joinpath(image_test_dir_list[idx])
    label_path = test_label_dir.joinpath(label_test_dir_list[idx]) #first part before joinpath is pathlib.Path, second part is the directory of the file 

    #Get 3D array after pre-processing
    image = mri.Mri(img_path, is_label= False, is_train_set= True)
    label = mri.Mri(label_path, is_label= True, is_train_set= True)  #NOTE going to end up resampling ALL images to common voxel spacing to get more data 

    #Copy
    image_a = image.hu_a
    label_a = label.hu_a
    print("test", idx)

    #Extract slices after processing to corresponding directories 
    array_transforms.extract_slices(image_a, image_test_dir_list[idx], test_img_slice_dir) 
    array_transforms.extract_slices(label_a, label_test_dir_list[idx], test_label_slice_dir) 
    