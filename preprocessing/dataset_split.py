#PREPROCESSING 3

#Now that we have our low-resolution 3D images it's time to separate them to train and test splits
    #Not applying this step to 2D images since the result will be the same plus this is an
    #easier way to ensure series from the same patient will only be present in either train or test

import SimpleITK as sitk
import natsort
import pathlib
import os
import shutil
import random
from sklearn.model_selection import train_test_split

#Full 
full_image_path = pathlib.Path(r"D:/Spider Data/images_series")
full_label_path = pathlib.Path(r"D:/Spider Data/labels_series")
#Directories to copy to
train_image_path = pathlib.Path(r"D:/Spider Data/train_images")
train_label_path = pathlib.Path(r"D:/Spider Data/train_labels")
test_image_path = pathlib.Path(r"D:/Spider Data/test_images")
test_label_path = pathlib.Path(r"D:/Spider Data/test_labels")

full_image_dir_list = os.listdir(full_image_path)
full_label_dir_list = os.listdir(full_label_path)

print("full image dir len", len(full_image_dir_list))

# Function to get patient IDs from filenames
def get_patient_id(filename):
    return filename.split('_')[0]


# Get list of image filenames
image_files = os.listdir(full_image_path)

# Split filenames into train and validation sets based on patient IDs
patient_ids = [get_patient_id(filename) for filename in image_files]
unique_patient_ids = list(set(patient_ids))
train_patient_ids, valid_patient_ids = train_test_split(unique_patient_ids, test_size=0.2, random_state=46)




# Copy images and masks to train or test directories
for filename in image_files:
    patient_id = get_patient_id(filename)
    is_train = patient_id in train_patient_ids
    source_image_path = os.path.join(full_image_path, filename)
    source_mask_path = os.path.join(full_label_path, filename)  # Assuming mask filenames are same as image filenames

    if is_train:
        destination_image_path = os.path.join(train_image_path, filename)
        destination_mask_path = os.path.join(train_label_path, filename)
    else:
        destination_image_path = os.path.join(test_image_path, filename)
        destination_mask_path = os.path.join(test_label_path, filename)

    shutil.copy(source_image_path, destination_image_path)
    shutil.copy(source_mask_path, destination_mask_path)
    

