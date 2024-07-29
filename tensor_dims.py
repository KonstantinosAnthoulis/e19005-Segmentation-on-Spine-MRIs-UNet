#Dependencies 
import numpy as np 
from natsort import natsorted
import pathlib
import os
import sys
import json


#Import MRI Slice class
from mri_slice import Mri_Slice
#Array transforms for cropping
from transforms import array_transforms

#In the script extract_slices.py we've cropped all the slices to only contain their own ROI, removing all the 
#data around the images with 0 label information.

#This script combs through the dataset to find the max dimensions of an image for creating tensors when training,
#as well as saving min and max pixel values from the image_train directory for normalization further down the line 

#2D Slice Directories | Training set, replace with path as needed 
#train_img_slice_dir = pathlib.Path(r"")
#train_label_slice_dir = pathlib.Path(r"")

train_img_slice_dir = pathlib.Path(r"D:/Spider Mini Set Check Data Pipeline/cropped/train_images")
train_label_slice_dir = pathlib.Path(r"D:/Spider Mini Set Check Data Pipeline/cropped/train_labels")

image_path = train_img_slice_dir
label_path = train_label_slice_dir

image_dir_list = os.listdir(image_path)
label_dir_list = os.listdir(label_path)

#Sort directories so that each image has its corresponding label on same idx 
image_dir_list = natsorted(image_dir_list)
label_dir_list = natsorted(label_dir_list)

#Empty lists to hold row and col dimensions of images
row_list = []
col_list = []

dirlen_image = len(os.listdir(image_path))
dirlen_label = len(os.listdir(label_path))

print("train images count", dirlen_image)

if(dirlen_image != dirlen_label):
  print("Error: image directory has", dirlen_image, "images not equal to label directory", dirlen_label, "images")
  sys.exit()
  

#After check pass the value to a single var since they're both the same value 
dirlen = dirlen_image

#TODO
#images are already cropped so just go through the dset to find
#max image dims (row, col)
#min and max train_images pixel values for tensor normalisation 

for idx in range(0, dirlen):

  print("idx", idx)
  #Get image and corresponding label paths
  img_path = image_path.joinpath(image_dir_list[idx])
  lbl_path = label_path.joinpath(label_dir_list[idx])#first part before joinpath is pathlib.Path, second part is the directory of the file 
 
  #Read image and label
  image = Mri_Slice(img_path)
  label = Mri_Slice(lbl_path)

  #Get arrays
  image_a = image.hu_a
  label_a = label.hu_a

  #Comb through the dataset to find max ROI dimensions as well as min/max image values for tensor normalisation later 
  if(idx ==0):
    
    image_tensor_min = np.min(image_a)
    image_tensor_max = np.max(label_a)
    label_tensor_min = np.min(label_a)
    label_tensor_max = np.max(label_a)
    
    unique_masks_a = np.unique(label_a)
  else:
    
    if(np.min(image_a) < image_tensor_min):
      image_tensor_min = np.min(image_a)
      image_tensor_min_dir = img_path
    if(np.min(label_a) < label_tensor_min):
      label_tensor_min = np.min(label_a)
    if(np.max(image_a) > image_tensor_max):
      image_tensor_max = np.max(image_a)
    if(np.max(label_a) > label_tensor_max):
      label_tensor_max = np.max(label_a)
    
  #Find amount of unique masks for one-hot encoding 
    current_masks_a = np.unique(label_a)
    if(len(current_masks_a) > len(unique_masks_a)):
      unique_masks_a = current_masks_a
  
  #Add values to lists
  row_list.append(image_a.shape[0]) 
  col_list.append(image_a.shape[1])  
  
#Get max values from lists
row_dim_max = max(row_list)
col_dim_max = max(col_list)

#Nearest Multiples of 16 for Unet
row_dim_max = ((row_dim_max + 15) // 16) * 16 
col_dim_max = ((col_dim_max + 15) // 16) * 16

#Prints
'''
print("row max:", max(row_list))
print("col max:", max(col_list))

print("image tensor min", image_tensor_min)
print("image tensor max", image_tensor_max)
print("label tensor min", label_tensor_min)
print("label tensor max", label_tensor_max)

print("amount of masks", len(unique_masks_a))
print("masks array", unique_masks_a)
'''

#These are the mask values from the Spider GC Dataset, I'm just leaving this here explicitly because some images don't contain mask 209 
  #Comment the code out for different dataset/depending on your application
#unique_masks_a = np.array([0. ,   1. ,  2.   ,3.  , 4. ,  5. ,  6. ,  7. ,  8.,   9. ,100. ,201. ,202. ,203. ,204., 205. ,206. ,207. ,208. ,209.])

#Save to .json for easy access when training the model 
data = {
    "row_max": max(row_list),
    "col_max": max(col_list),
    "image_tensor_min": image_tensor_min,
    "image_tensor_max": image_tensor_max,
    "label_tensor_min": label_tensor_min,
    "label_tensor_max": label_tensor_max,
    "masks_no": len(unique_masks_a),
    "masks_array": unique_masks_a
}

# Print types of elements in the data dictionary to check for non-serializable types
for key, value in data.items():
    print(f"{key}: {type(value)}")

# Function to convert numpy data types to native Python types
def convert_to_native_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# Convert all data to native Python types
data = {key: convert_to_native_types(value) for key, value in data.items()}

# Set the path for the JSON file
file_path = "D:/Spider Mini Set Check Data Pipeline/tensor_data.json"

# Save the dictionary to a JSON file
with open(file_path, "w") as json_file:
    json.dump(data, json_file, indent=4)



