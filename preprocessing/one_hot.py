import numpy as np 
import json 
import cupy as cp 

def one_hot_encoding(masks_no, masks_array): 
    val_range = np.arange(masks_no)
    mapping_dict = dict(zip(masks_array, val_range))
    
    # Define a function to perform the mapping using the dictionary
    def map_to_specified_set(value):
        return mapping_dict.get(value, np.nan)  # Return np.nan for values not found in the mapping_dict
    
    # Create a vectorized version of the mapping function
    value_map = np.vectorize(map_to_specified_set)
    
    return value_map

def one_hot_encoding_cp(masks_no, masks_array):
    masks_array = cp.asarray(masks_array)  # Ensure it's a CuPy array
    val_range = cp.arange(masks_no)
    unique_vals = cp.array(masks_array)  
    mapping_vals = cp.arange(masks_no)

    def map_to_specified_set(arr):
        output = cp.full(arr.shape, -1)  # Default all to -1
        for i, uval in enumerate(unique_vals):
            output[arr == uval] = mapping_vals[i]
        return output
    
    # Return the mapping function itself (not the output)
    return map_to_specified_set  # Return the function, not its output

