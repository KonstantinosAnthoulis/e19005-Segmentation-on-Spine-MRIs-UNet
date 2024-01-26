import numpy as np

def remove_empty_slices(img_a, label_a):
    #creating new array instead of deleting from old one 
    slice_num = -1 #negative 1 because +1 brings us to [0] as first slice 
    x = label_a.shape[1]
    y = label_a.shape[2]

    print("size of array before trimming", label_a.shape[0])

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

    print("size of array after trimming what it should be", out_label_a.shape[0])

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


    print(np.any(out_label_a[0]))
    print(out_label_a.shape)
    
    return out_img_a, out_label_a



#remove all rows columns with just 0s 
def crop_zero(img_a, label_a):
    row_max = 0
    col_max = 0 #max of non empty rows and columns in the slices 

    #max_list = [] #will hold [idx, count of rows, count of cols] 
    x_max_list = []
    y_max_list = []

     #finding start and end of non-zero cols rows
    row_start_idx = 0 #reset for each mri
    row_end_idx = 0 #reset for each mri
    col_start_idx = 0 #reset for each mri
    col_end_idx = 0 #reset for each mri

    for idx in range(label_a.shape[0]): #go through non empty slices
        row_count = 0 #reset for each slice
        col_count = 0 #reset for each slice 

        #counting 0 rows, using this util for now but there is a chance it could find and delete cols/rows of 0 within the region of interest
        for idx_row in range(label_a.shape[1]): #for rows
            if  np.any(label_a[idx, idx_row, :]):
                row_count = row_count + 1
                if(row_start_idx == 0): #check if first non zero row 
                    row_start_idx = idx_row
                if np.any(label_a[idx, idx_row + 1, :]): #check if last non zero row
                    row_end_idx = idx_row 

        for idx_col in range(label_a.shape[2]): #for columns
            if  np.any(label_a[idx, : ,idx_col]):
                col_count = col_count +1
                if(col_start_idx == 0): #check if first non zero col 
                    col_start_idx = idx_col
                if np.any(label_a[idx, :, idx_col + 10]): #check if last non zero col
                    col_end_idx = idx_col


        #max_list.append([idx, row_count, col_count]) #index of slice in mri, row count, col count  
        x_max_list.append(row_count)
        y_max_list.append(col_count)
        #print("slice and non 0 lines cols",max_list[idx])   
        
    
    print("original image res", label_a.shape) 

    #max_list_arr = np.array(max_list)
    #print(max_list_arr)

    x_max_nonzero = max(x_max_list)
    y_max_nonzero = max(y_max_list)
    print("x max", x_max_nonzero)
    print("y max", y_max_nonzero)

    print("row start idx", row_start_idx)
    print("row end idx", row_end_idx)
    print("col start idx", col_start_idx )
    print("col end idx", col_end_idx)



    #TODO create the new arrays to return all the non_zero rows cols 

    #out_img_a = np.empty([img_a.shape[0],x_max_nonzero, y_max_nonzero])
    #out_label_a = np.empty([label_a.shape[0], x_max_nonzero, y_max_nonzero])

    print("/2 row", img_a.shape[1]/2)
    print("/2 col", img_a.shape[2]/2 )

    out_img_a = img_a[ img_a.shape[1]//2 - x_max_nonzero//2:img_a.shape[1] + x_max_nonzero//2 , img_a.shape[2]//2 - y_max_nonzero//2:img_a.shape[2] + y_max_nonzero//2 , img_a.shape[0]]

    print(out_img_a.shape)
    return x_max_nonzero, y_max_nonzero #temporary just to get max dims

    #TODO find the rows and cols where nonzeros start and end and from that index start adding to nonzero array



    