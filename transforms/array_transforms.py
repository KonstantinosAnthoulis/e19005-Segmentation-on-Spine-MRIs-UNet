import numpy as np


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



#remove all rows columns with just 0s 
def crop_zero(img_a, label_a):
 
    x_max_list = []
    y_max_list = []

    row_max_list = []
    col_max_list = []

    #TODO something goes wrong here in counting non zero rows cols, they end up higher than the image dimensions somehow 

    for idx in range(label_a.shape[0]): #go through non empty slices
        row_count = 0 #reset for each slice
        col_count = 0 #reset for each slice 

        #counting 0 rows, using this util for now but there is a chance it could find and delete cols/rows of 0 within the region of interest
        for idx_row in range(label_a.shape[1]): #for rows/y
            if  np.any(label_a[idx, idx_row, :]):
                row_count = row_count + 1
             

        for idx_col in range(label_a.shape[2]): #for columns/x
            if  np.any(label_a[idx, : ,idx_col]):
                col_count = col_count +1
                

        #max_list.append([idx, row_count, col_count]) #index of slice in mri, row count, col count  
        row_max_list.append(row_count)
        col_max_list.append(col_count)
        #print("slice and non 0 lines cols",max_list[idx])   
        
    #print("original image res", label_a.shape) 

    #max_list_arr = np.array(max_list)
    #print(max_list_arr)

    row_max_nonzero = max(row_max_list)
    col_max_nonzero = max(col_max_list)
    #print("y max nonzero", y_max_nonzero)
    #print("x max", x_max_nonzero)
    #print("y max", y_max_nonzero)

    if (row_max_nonzero % 16 != 0):
        out_row = ((row_max_nonzero + 15) // 16) * 16
        #print("row max going in div", row_max_nonzero)
        #print("row div", row_max_nonzero % 16)
        
    else:
        out_row = row_max_nonzero

    if(col_max_nonzero % 16 != 0): 
        out_col = ((col_max_nonzero + 15) // 16) * 16
        #print("col max going in div", col_max_nonzero)
        #print("col div", col_max_nonzero % 16)
    else:  
        out_col = col_max_nonzero
    
    center_row = label_a.shape[1]//2 - out_row//2
    center_col = label_a.shape[2]//2 - out_col//2

    out_img_a = np.empty([img_a.shape[0], out_row, out_col])
    out_label_a = np.empty([label_a.shape[0],out_row, out_col]) #return arrays will have x y dims multiple of 16 for unet 
    
    #print("out shape", out_img_a.shape)

    for idx in range(label_a.shape[0]): #go through non empty slices
        img_slice = img_a[idx]
        label_slice = label_a[idx]
        #print("slice from input array shape", img_slice.shape)
        
        #print("row dims to grab from", center_row, center_row + out_row)
        #TODO find how to get center crop of 
        out_img_a[idx] = img_slice[center_row:center_row + out_row, center_col:center_col + out_col]

        out_label_a[idx] =label_slice[center_row:center_row + out_row, center_col:center_col + out_col]
    
    #print(out_label_a.shape)
    
    return out_img_a, out_label_a 

    



    