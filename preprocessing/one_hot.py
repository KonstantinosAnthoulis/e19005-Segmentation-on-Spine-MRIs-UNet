import numpy as np 
import json 

def one_hot_encoding(masks_no, masks_array): 
    val_range = np.arange(masks_no)
    mapping_dict = dict(zip(masks_array, val_range))
    
    # Define a function to perform the mapping using the dictionary
    def map_to_specified_set(value):
        return mapping_dict.get(value, np.nan)  # Return np.nan for values not found in the mapping_dict
    
    # Create a vectorized version of the mapping function
    value_map = np.vectorize(map_to_specified_set)
    
    return value_map

