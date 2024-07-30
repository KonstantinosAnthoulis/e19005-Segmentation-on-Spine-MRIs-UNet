#Dependencies
import SimpleITK as sitk
import numpy as np
 
#Parameters:
    #path: file path to 2D .mha image 
class Mri_Slice:
    def __init__(self, path):

        #Read 2D image from path, explicitly setting IOReader just in case 
        mri_mha = sitk.ReadImage(path, imageIO = "MetaImageIO") 

        #Extract 2D numpy array from the .mha file 
        mri_a = np.array(sitk.GetArrayFromImage(mri_mha))

        #Convert to float32, helps with creating tensors down the line     
        mri_a_float32 = mri_a.astype(dtype = np.float32)
  
        self.hu_a = mri_a_float32