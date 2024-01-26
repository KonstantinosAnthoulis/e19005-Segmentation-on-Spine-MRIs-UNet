import numpy as np


def remove_empty_slices(img_a, label_a):
    #creating new array instead of deleting from old one 
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
    row_max = 0
    col_max = 0 #max of non empty rows and columns in the slices 

    #max_list = [] #will hold [idx, count of rows, count of cols] 
    x_max_list = []
    y_max_list = []

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
        x_max_list.append(col_count)
        y_max_list.append(row_count)
        #print("slice and non 0 lines cols",max_list[idx])   
        
    
    #print("original image res", label_a.shape) 

    #max_list_arr = np.array(max_list)
    #print(max_list_arr)

    x_max_nonzero = max(x_max_list)
    y_max_nonzero = max(y_max_list)
    print("y max nonzero", y_max_nonzero)
    #print("x max", x_max_nonzero)
    #print("y max", y_max_nonzero)

    if (x_max_nonzero % 16 != 0):
        out_x = ((x_max_nonzero + 15) // 16) * 16
    else:
        out_x = x_max_nonzero

    if(y_max_nonzero % 16 != 0): 
        out_y = ((y_max_nonzero + 15) // 16) * 16
        print("y max going in div", y_max_nonzero)
        print("y div", y_max_nonzero % 16)
    else:  
        out_y = y_max_nonzero
    
    center_row = label_a.shape[2]//2 - out_y//2
    center_col = label_a.shape[1]//2 - out_x//2

    out_img_a = np.empty([img_a.shape[0], out_x, out_y])
    out_label_a = np.empty([label_a.shape[0],out_x, out_y]) #return arrays will have x y dims multiple of 16 for unet 
    
    #print("out shape", out_img_a.shape)

    for idx in range(label_a.shape[0]): #go through non empty slices
        img_slice = img_a[idx]
        label_slice = label_a[idx]
        #print("slice from input array shape", img_slice.shape)
        
        #TODO find how to get center crop of 
        out_img_a[idx] = img_slice[center_col:center_col + out_x, center_row:center_row + out_y]

        out_label_a[idx] =label_slice[center_col:center_col + out_x, center_row:center_row + out_y]
    
    #print(out_label_a.shape)
    
    return out_img_a, out_label_a 

    



    