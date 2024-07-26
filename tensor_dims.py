#Dependencies 
import numpy as np 
from natsort import natsorted

#Import MRI Slice class
from mri_slice import Mri_Slice
#Array transforms for cropping
from transforms import array_transforms

#In the script extract_slices.py we've cropped all the slices to only contain their own ROI, removing all the 
#data around the images with 0 label information.

#This script combs through the dataset to find the max dimensions of an image for creating tensors when training,
#as well as saving min and max pixel values from the image_train directory for normalization further down the line 

#2D Slice Directories, replace with path as needed 
#train_img_slice_dir = pathlib.Path(r"")
#train_label_slice_dir = pathlib.Path(r"")
#test_img_slice_dir = pathlib.Path(r"")
#test_label_slice_dir= pathlib.Path(r"")

row_list = []
col_list = []


#TODO
#images are already cropped so just go through the dset to find
#max image dims (row, col)
#min and max train_images pixel values for tensor normalisation 

for idx in range(0, dirlen):
  #print("dirlen", dirlen)
  
 #toy dset 
  
  img_path = image_path.joinpath(image_dir_list[idx])
  lbl_path = label_path.joinpath(label_dir_list[idx])#first part before joinpath is pathlib.Path, second part is the directory of the file 
  '''
  img_path = local_img_idr.joinpath(image_dir_list[idx])
  label_path = local_label_dir.joinpath(label_dir_list[idx]) #first part before joinpath is pathlib.Path, second part is the directory of hte file 
  '''
  image = Mri_Slice(img_path)
  label = Mri_Slice(lbl_path)

  image_a = image.hu_a
  label_a = label.hu_a

  #crop zero
  image_a_cropzero, label_a_cropzero = array_transforms.crop_zero(image_a, label_a)

  #find array min max values for normalising in Dataset class
  if(idx ==0):
    
    image_tensor_min = np.min(image_a_cropzero)
    image_tensor_max = np.max(label_a_cropzero)
    label_tensor_min = np.min(label_a_cropzero)
    label_tensor_max = np.max(label_a_cropzero)
    
    unique_masks_a = np.unique(label_a)
  else:
    
    if(np.min(image_a_cropzero) < image_tensor_min):
      image_tensor_min = np.min(image_a_cropzero)
      image_tensor_min_dir = img_path
    if(np.min(label_a_cropzero) < label_tensor_min):
      label_tensor_min = np.min(label_a_cropzero)
    if(np.max(image_a_cropzero) > image_tensor_max):
      image_tensor_max = np.max(image_a_cropzero)
    if(np.max(label_a_cropzero) > label_tensor_max):
      label_tensor_max = np.max(label_a_cropzero)
    
  #find amount of unique masks for one-hot encoding 
    current_masks_a = np.unique(label_a)
    if(len(current_masks_a) > len(unique_masks_a)):
      unique_masks_a = current_masks_a
  #print("image res", image_a_cropzero.shape)
  
  
  row_list.append(image_a_cropzero.shape[0]) #add row value to list
  #print(image_a_cropzero.shape[0])
  col_list.append(image_a_cropzero.shape[1]) #add col value to list 
  
  

#calculate max 

row_dim_max = max(row_list)
col_dim_max = max(col_list)

row_dim_max = ((row_dim_max + 15) // 16) * 16 #nearest multiple of 16 
col_dim_max = ((col_dim_max + 15) // 16) * 16 #nearest multiple of 16


print("row max:", max(row_list))
print("col max:", max(col_list))

print("image tensor min", image_tensor_min)
print("image tensor max", image_tensor_max)
print("label tensor min", label_tensor_min)
print("label tensor max", label_tensor_max)

print("amount of masks", len(unique_masks_a))
print("masks array", unique_masks_a)



