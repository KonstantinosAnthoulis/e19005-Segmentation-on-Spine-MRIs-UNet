import numpy as np 
import json 
import cupy as cp 

def value_map(masks_no, masks_array): 
    val_range = np.arange(masks_no)
    mapping_dict = dict(zip(masks_array, val_range))
    
    # Define a function to perform the mapping using the dictionary
    def map_to_specified_set(value):
        return mapping_dict.get(value, np.nan)  # Return np.nan for values not found in the mapping_dict
    
    # Create a vectorized version of the mapping function
    value_map = np.vectorize(map_to_specified_set)
    
    return value_map

def value_map_cp(masks_no, masks_array): 
    # Create a range of values on the GPU
    val_range = cp.arange(masks_no)
    
    # Create a mapping dictionary using CuPy
    mapping_dict = cp.array(masks_array)
    
    # Define a function to perform the mapping
    def map_to_specified_set(values):
        # Ensure values is a CuPy array
        values = cp.asarray(values)
        
        # Create an output array initialized with cp.nan
        mapped = cp.full(values.shape, cp.nan)  # Default to nan
        
        # Iterate over the unique values in the input
        for unique_value in cp.unique(values):
            # Find the index of the unique_value in mapping_dict
            index = cp.where(mapping_dict == unique_value)[0]
            if index.size > 0:
                # Map the found index to the corresponding value in val_range
                mapped[values == unique_value] = val_range[index[0]]
        
        return mapped
    
    # Return the mapping function directly (no need for vectorization here)
    return map_to_specified_set