# SPIDER Grand Challenge - 2D Segmentation UNet on spinal MR images (Bachelor's Thesis)
## Intro
This repo details the entire process of training a 2D segmentation Unet in the context of the [competition set by Radboud UMC](https://spider.grand-challenge.org/). The competition objective is to train a segmentation model to accurately detect anatomical structures of the spine, such as the vertebrae, intervertrebral discs (IVD) and the spinal canal within spine MR images. Due to the dimensionality of the images (analyzed further in the following chapters), 3D convnets are considered SOTA due to their ability to capture spatial information from all 3 axes, such as the template 3D NN-Unet trained by the competition hosts. The downside of 3D convnets is the exponential increase of computational demands during training compared to 2D convnets. The subject of the thesis was to test whether a 2D Unet, given the correct hyperparameters and extensive augmentation, can get close to replicating those results with a much smaller tech footprint. A lighter model such as this would also mean inference would be faster and less computationally demanding, rendering such a solution more easily deployable in a clinical setting. This readme can also serve as a quick primer if the reader is planning to create their own iteration, as the dataset will be covered in detail.
## MRI 
Before we dive into the data, a quick glance over what an MR image is and how they are portrayed on their multiple axes can help set everything in perspective. +
## Dataset
The original dataset is comprised of 447 MR images + 
## Preprocessing
Initially + 
## Model and Hyperparameters

## Training Results
Compared to + 
