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
    val_range = cp.arange(masks_no)  # Use CuPy's arange
    mapping_dict = {k: v for k, v in zip(masks_array, val_range)}  # Mapping dictionary remains the same
    
    # Define a function to perform the mapping using the dictionary
    def map_to_specified_set(value):
        return mapping_dict.get(value, cp.nan)  # Return cp.nan for values not found in the mapping_dict
    
    # Create a vectorized version of the mapping function
    value_map = cp.vectorize(map_to_specified_set)
    
    return value_map
