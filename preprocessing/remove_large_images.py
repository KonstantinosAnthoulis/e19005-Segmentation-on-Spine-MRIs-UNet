#PREPROCESSING 2 

#Now that we have our patient series, due to technical constraints, we are going to remove any images that
    #have resolution > 600 in any axis except index

#This number was picked upon dataset inspection but you can set it to anything you want/omit the step entirely if your
    #training setup can handle it 

#Dependencies 
import SimpleITK as sitk 
import pathlib
import os 
import numpy as np
from natsort import natsorted

#Directories with patient series
images_dir = pathlib.Path(r"C:/Users/Konstantinos/Desktop/Spider Data/images_series")
labels_dir = pathlib.Path(r"C:/Users/Konstantinos/Desktop/Spider Data/labels_series")

#Directories to write the lower res images to
images_lowres_dir = pathlib.Path(r"C:/Users/Konstantinos/Desktop/Spider Data/images_series_lowres")
labels_lowres_dir = pathlib.Path(r"C:/Users/Konstantinos/Desktop/Spider Data/labels_series_lowres")

#Set max res, adjust as needed
max_res = 550

#Count how many images we are removing from the dset 
rm_count  = 0

#Directories Lists
images_dir_list = os.listdir(images_dir)
labels_dir_list = os.listdir(labels_dir)

images_dir_list = natsorted(images_dir_list)
labels_dir_list = natsorted(labels_dir_list)

dirlen = len(images_dir_list)

for idx in range(0, dirlen):
    
    #Get image paths in directory
    img_path = images_dir.joinpath(images_dir_list[idx])    
    label_path = labels_dir.joinpath(labels_dir_list[idx]) 

    #Get sitk images
    img_sitk = sitk.ReadImage(img_path)
    label_sitk = sitk.ReadImage(label_path)

    #Get np arrays of image
    img_np = sitk.GetArrayFromImage(img_sitk)
   
    if any(dim > max_res for dim in img_np.shape):
        rm_count = rm_count + 1 
        continue 

    #If past here write the images to new directory 
    sitk.WriteImage(img_sitk, images_lowres_dir.joinpath(images_dir_list[idx]))
    sitk.WriteImage(label_sitk, labels_lowres_dir.joinpath(labels_dir_list[idx]))


print("total images removed", rm_count)

images_lowres_dir_list = os.listdir(images_lowres_dir)

print(len(images_lowres_dir_list))



    