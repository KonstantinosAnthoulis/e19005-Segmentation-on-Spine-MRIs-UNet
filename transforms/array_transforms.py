import numpy as np
import SimpleITK as sitk
import pathlib

#--------------------3D ARRAYS-------------------------------------------------------------------
def remove_empty_slices(img_a, label_a):
    
    slice_num = -1 #negative 1 because +1 brings us to [0] as first slice 
    x = label_a.shape[1]
    y = label_a.shape[2]

    #print("size of array before trimming", label_a.shape[0])

    #loop to count non empty slices in file 
    for idx in range(label_a.shape[0]): #go through slices 
        #print("array shape before adding slice" , new_label.shape) #print array shape before removing slice 
        if (np.any(label_a[idx])): #slice is NOT empty
        #print("non empty slice at idx", idx)
        #print(np.any(label.hu_a[idx]))
            slice_num = slice_num +1 

    #print(slice_num)
    out_label_a = np.empty((slice_num+1, x, y))
    out_img_a = np.empty((slice_num+1, x, y))

    #print("size of array after trimming what it should be", out_label_a.shape[0])

    new_index = 0

    for idx in range(label_a.shape[0]): #go through slices of original array
        
        if (np.any(label_a[idx])): #slice is NOT empty
        #print(np.any(label.hu_a[idx]))
        #print("index", idx)
        #print("new index", new_index)
            out_label_a[new_index] = label_a[idx] #add non empty slice to array
            out_img_a[new_index] = img_a[idx]
        #print("slice added", new_label_hu[new_index].shape)
        #print("idx of slice being added", new_index)
        #print("idx of slices", idx)
            new_index = new_index +1


    #print(np.any(out_label_a[0]))
    #print(out_label_a.shape)
    
    return out_img_a, out_label_a


def extract_slices(arr, input_mri_path, target_slice_dir):

    #For every 2D slice in the 3D array
    for idx in range(arr.shape[0]):
        #Get 2D Array
        idx_arr = arr[idx, : , :]

        #Vertical and horizontal flips because resampling ode screws things up for some reason
        idx_arr = np.flipud(idx_arr) 
        idx_arr = np.fliplr(idx_arr)

        #Get 2D image
        idx_slice = sitk.GetImageFromArray(idx_arr)
        
        #Small debug print to ensure correct dimensions (should be triple digits in both X and Y)
        
        if(idx == 0):
            print("dims of exported slices:", idx_slice.GetSize(), "for" , input_mri_path)
       
        #Naming convention + path for new file
        input_path_split = input_mri_path.split(".")
        pre = input_path_split[0] #1_t1
        post = "_" + str(idx) + "." + input_path_split[1] #_0.mha for 1st slice of image 1_t1

        slice_path = pre + post
        target_dir = target_slice_dir.joinpath(slice_path)

        sitk.WriteImage(idx_slice, target_dir)

#--------------------2D ARRAYS-------------------------------------------------------------------
#remove all rows columns with just 0s 
import numpy as np

import numpy as np

def crop_zero(img_a, label_a):
    # Find the bounding box of the ROI in the label array
    rows = np.any(label_a, axis=1)
    cols = np.any(label_a, axis=0)
    
    # Check if there are any non-zero values in the label
    if not np.any(rows) or not np.any(cols):
        # No non-zero values found, return the original images
        return img_a, label_a
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Calculate the dimensions of the ROI
    roi_height = rmax - rmin + 1
    roi_width = cmax - cmin + 1

    # Ensure the dimensions are multiples of 16 for U-Net
    if roi_height % 16 != 0:
        out_row = ((roi_height + 15) // 16) * 16
    else:
        out_row = roi_height

    if roi_width % 16 != 0:
        out_col = ((roi_width + 15) // 16) * 16
    else:
        out_col = roi_width

    # Calculate the center of the bounding box
    center_row = (rmin + rmax) // 2
    center_col = (cmin + cmax) // 2

    # Calculate the crop start coordinates
    start_row = max(0, center_row - out_row // 2)
    start_col = max(0, center_col - out_col // 2)

    # Ensure the crop stays within the image boundaries
    end_row = min(start_row + out_row, img_a.shape[0])
    end_col = min(start_col + out_col, img_a.shape[1])
    start_row = end_row - out_row
    start_col = end_col - out_col

    # Crop the image and label arrays
    out_img_a = img_a[start_row:end_row, start_col:end_col]
    out_label_a = label_a[start_row:end_row, start_col:end_col]

    return out_img_a, out_label_a









    