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
import albumentations as A
from natsort import natsorted
import matplotlib.pyplot as plt

#Training Data Paths, only applying augmentation on training data 
train_img_slice_dir = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/train_image_slices")
train_label_slice_dir = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/train_label_slices")

#test dir to write in 
noise_dir = pathlib.Path(r"C:/Users/user/Desktop/Spider Data/noise_test")

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

gaussian_noise_transform = A.Compose([
    A.GaussNoise(var_limit=(0.001, 0.01), mean = 0, p=1), #Gaussian Noise  
    A.ElasticTransform(alpha=20, sigma=4, p=1) #Random Elastic Deformation 
])

for idx in range(0, dirlen):

    image_path = train_img_slice_dir.joinpath(image_train_dir_list[idx])
    label_path = train_label_slice_dir.joinpath(label_train_dir_list[idx])

    image_sitk = sitk.ReadImage(image_path)
    label_sitk = sitk.ReadImage(label_path)

    image_np = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    label_np = sitk.GetArrayFromImage(label_sitk).astype(np.int16) #explicitly setting float32 for albumentations to work

    #try normalising input image before feeding it to gauss 
    image_min = np.min(image_np)
    image_max = np.max(image_np)

    image_normalised_np = (image_np - image_min) / (image_max - image_min)


    #augment
    augmented = gaussian_noise_transform(image = image_normalised_np)
    image_augmented_np = augmented["image"]

    if image_augmented_np.shape[-1] == 1:
        image_augmented_np = np.squeeze(image_augmented_np, axis=-1)
        image_np = np.squeeze(image_np, axis=-1)

    #revert normalisation 
    image_augmented_np = np.clip(image_augmented_np, 0 ,1)

    image_augmented_np = image_augmented_np * (image_max - image_min) + image_min 
    
    #Plotting code for checking if the augmentation is applied correctly
    #Please don't uncomment this and run the loop you'll flood with matplotlib plots 
    
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
    

    image_augmented_sitk = sitk.GetImageFromArray(image_augmented_np)

    break


