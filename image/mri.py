#Dependencies 
import SimpleITK as sitk
reader = sitk.ImageFileReader()
reader.SetImageIO("MetaImageIO")
import numpy as np

#Import funtcions from transforms.py
#rom transforms import mri_transforms, array_transforms
import mri_transforms, array_transforms
#Here resampling and any other 3D pre-processing are applied before getting the 3D numpy array from the .mha file 
#Parameters:
    #path: file path to .mha image
    #is_label (T/F): checks if the file is image or masks to apply appropriate resampling method 
    #is_train_set (T/F): checks if the image will be part of the training or testing dataset
      #not resampling test datasets to better resemble real world data   
class Mri:
    def __init__(self, path, is_label, is_train_set):

        #Read 3D image from path, explicitly setting IO Reader just in case 
        mri_mha = sitk.ReadImage(path, imageIO = "MetaImageIO") 

        #transpose so that idx first 
        mri_mha = mri_transforms.transpose(mri_mha)

        #Resample accordingly if part of training set
        if(is_train_set):
          mri_mha = mri_transforms.resample_img(mri_mha, out_spacing= [2.0, 0.6, 0.6], is_label=is_label) 
      
        #Extract 3D array from the .mha file
        mri_a = np.array(sitk.GetArrayFromImage(mri_mha)) #mri_array

        #Transpose Array to common format [idx, row, col] 
        if(mri_a.shape[0] > mri_a.shape[1] or mri_a.shape[0] > mri_a.shape[2]): #if z axis isn't first
          mri_a = np.transpose(mri_a, (2, 0, 1)) #set axes to correct order

        #Clip values to the upper/lower limits of Hounsfield scale 
        mri_a = np.clip(mri_a, -1000, 4000)

        #note: for some reason sitk.GetArrayFromImage() takes an image with dims [slice index, row, col] and transforms it to[row, col, slice index]
            #this bit of code solves that issue, not sure why it happens but this is enough to make sure all the images are of format [slice, row, col]
        
        #Keeping this bit here commented out because a lot of the images were flipped 90 degrees after resampling, could be needed upon exported dset inspection 
        '''
        if(mri_a.shape[2] > mri_a.shape[1]): #bring images to vertical orientation
          mri_a = np.transpose(mri_a, (0, 2, 1))
        '''
       
        #Convert to float32, helps with creating tensors down the line 
        mri_a_float32 = mri_a.astype(dtype = np.float32)

        #TODO: Set bounds to [-1000, 2000] https://en.wikipedia.org/wiki/Hounsfield_scale
        #Note: could experiment with restricting the values to the range on the Hounsfield scale (scale used by MRI imaging) where bone appears
            #Not sure because of medical ambiguity, should test

        #get the final array from the class
        self.hu_a = mri_a_float32




