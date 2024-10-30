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
    # Convert masks_array to a cupy array and ensure it's of integer type
    masks_array = cp.asarray(masks_array).astype(cp.int32)
    val_range = cp.arange(masks_no)

    # Create a lookup array large enough to hold max value in masks_array
    max_value = int(cp.max(masks_array)) + 1
    lookup_array = cp.full(max_value, cp.nan)

    # Populate the lookup array
    lookup_array[masks_array] = val_range

    # Define the mapping function
    def map_to_specified_set(value):
        if value < max_value:
            return lookup_array[value]
        else:
            return cp.nan

    # Vectorize the function
    value_map = cp.vectorize(map_to_specified_set)
    
    return value_map
