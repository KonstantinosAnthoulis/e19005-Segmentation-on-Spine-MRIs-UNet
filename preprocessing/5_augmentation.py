# PREPROCESSING SOMETHING

# Data Augmentation

# Generate X amount of slices from 1 slice to further increase the dataset size 

# Dependencies 
import albumentations as A
import SimpleITK as sitk
import numpy as np
import pathlib 
import os 
import sys
from natsort import natsorted
import matplotlib.pyplot as plt
import shutil

# Training Data Paths, only applying augmentation on training data 
##train_img_slice_dir = pathlib.Path(r"D:/Spider Data/train_image_slices")
#train_label_slice_dir = pathlib.Path(r"D:/Spider Data/train_label_slices")

train_img_slice_dir = pathlib.Path(r"D:/Spider Data/dataset/train_image_slices")
train_label_slice_dir = pathlib.Path(r"D:/Spider Data/dataset/train_label_slices")

# Test directory to write in 
#train_img_augmented_slice_dir = pathlib.Path(r"D:/Spider Data/train_image_augmented_slices")
#train_label_augmented_slice_dir = pathlib.Path(r"D:/Spider Data/train_label_augmented_slices")

train_img_augmented_slice_dir = pathlib.Path(r"D:/Spider Data/dataset/train_augmented_image_slices")
train_label_augmented_slice_dir = pathlib.Path(r"D:/Spider Data/dataset/train_augmented_label_slices")

# Get lists of the files in the directories 
image_train_dir_list = os.listdir(train_img_slice_dir) 
label_train_dir_list = os.listdir(train_label_slice_dir)

# Sort lists just to be safe
image_train_dir_list = natsorted(image_train_dir_list)
label_train_dir_list = natsorted(label_train_dir_list)

train_len = len(image_train_dir_list)
test_len = len(label_train_dir_list)

'''
if (train_len != test_len):
    print("Error: Directories aren't of equal size")
    sys.exit()
'''
    
dirlen = train_len

# Set number of augmented images to generate per image 
# also applies for flipped so in total you get 2 x augmented_no instances per image 
augmented_no = 2

# Define the augmentation pipeline with Gaussian Noise and Elastic Transform
noise_transform = A.Compose([
    A.GaussNoise(var_limit=(0.001, 0.03), mean=0, p=1),  # Gaussian Noise  
    A.ElasticTransform(alpha=25, sigma=4, p=1)           # Random Elastic Deformation 
])

# Define the horizontal flip transform
flip_transform = A.HorizontalFlip(p=1)  # p=1 ensures the image is always flipped

# Loop through each image in the training set
for idx in range(0, dirlen):
    
    print("idx:", idx)

    image_path = train_img_slice_dir.joinpath(image_train_dir_list[idx])
    label_path = train_label_slice_dir.joinpath(label_train_dir_list[idx])

    image_sitk = sitk.ReadImage(image_path)
    label_sitk = sitk.ReadImage(label_path)

    image_np = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    label_np = sitk.GetArrayFromImage(label_sitk)

    """"
    # Loop over original and flipped versions
    for flip_type in ['original', 'flipped']:
        
        # Apply horizontal flip if needed
        if flip_type == 'flipped':
            flipped = flip_transform(image=image_np, mask = label_np)
            image_np = flipped['image']
            label_np = flipped['mask']
    """
    # Apply augmentations and generate the augmented images
    for aug_idx in range(augmented_no):

            # try normalising input image before feeding it to gauss 
            image_min = np.min(image_np)
            image_max = np.max(image_np)

            # Normalize the image, handling edge cases where image_max == image_min
            if image_max != image_min:
                image_normalised_np = (image_np - image_min) / (image_max - image_min)
            else:
                image_normalised_np = np.zeros_like(image_np)  # or keep it as is without normalization
    

            # augment
            augmented = noise_transform(image=image_normalised_np)
            image_augmented_np = augmented["image"]

            if image_augmented_np.shape[-1] == 1:
                image_augmented_np = np.squeeze(image_augmented_np, axis=-1)
                image_np = np.squeeze(image_np, axis=-1)

            # revert normalisation 
            image_augmented_np = np.clip(image_augmented_np, 0, 1)
            image_augmented_np = image_augmented_np * (image_max - image_min) + image_min 

            # Plotting code for checking if the augmentation is applied correctly
            # Please don't uncomment this and run the loop you'll flood with matplotlib plots 
            """
            # Plot the original and the augmented image side by side
            plt.figure(figsize=(10, 5))

            # Plot original image
            plt.subplot(1, 2, 1)
            plt.imshow(image_np, cmap='gray')
            plt.title("Original Image")
            plt.axis("off")

            # Plot image with Gaussian noise
            plt.subplot(1, 2, 2)
            plt.imshow(image_augmented_np, cmap='gray')
            plt.title("Image with Gaussian Noise")
            plt.axis("off")

            # Show the plots
            plt.tight_layout()
            plt.show()
            """

            image_augmented_sitk = sitk.GetImageFromArray(image_augmented_np)

            input_path_split = image_train_dir_list[idx].split(".")
            pre = input_path_split[0]  # '1_t1_0'
            post = input_path_split[1]  # '.mha'

            # Modify the filename to indicate if it's flipped and which augmentation
            """
            if flip_type == 'flipped':
                pre = pre + "_f"
            """
            pre_img = pre + "_" + str(aug_idx)

            print(pre_img)

            augmented_filename = pre_img + "." + post 
            
            image_augmented_path = train_img_augmented_slice_dir.joinpath(augmented_filename)
            #label_augmented_path = train_label_augmented_slice_dir.joinpath(augmented_filename)

            sitk.WriteImage(image_augmented_sitk, image_augmented_path)

            #if(aug_idx == 0):#
            #augmented_label_filename = pre + ".mha"
            label_augmented_sitk = sitk.GetImageFromArray(label_np)
            label_augmented_path = train_label_augmented_slice_dir.joinpath(augmented_filename)
            sitk.WriteImage(label_augmented_sitk, label_augmented_path)
"""
#Delete folders after processing to avoid cluttering disk space
def delete_folder(folder_path):
    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    else:
        print(f"Folder not found or already deleted: {folder_path}")


delete_folder(train_img_slice_dir)
delete_folder(train_label_slice_dir)
"""