#PREPROCESSING 1 

#Each patient series in the dataset is comprised of these possible images:
    #t1/t2 image only
    #t1 and t2 pair
    #t1 t2 and t2_space images 

#We want the model to be unbiased towards either t1 or t2. This first step will only give us the series with
    #both t1 and t2 images excl space

#Dependencies
import SimpleITK as sitk 
import pathlib
import os 
from natsort import natsorted

#Full Dataset Paths 
#images_dir = pathlib.Path(r"D:/Spider Data/images")
#labels_dir = pathlib.Path(r"D:/Spider Data/labels")

#Laptop path, only for running the data pipeline and writing the report while the model is training 
images_dir = pathlib.Path(r"D:/Spider Data/images")
labels_dir = pathlib.Path(r"D:/Spider Data/labels")


#Paths to write the t1-t2 series to
#images_series_dir = pathlib.Path(r"D:/Spider Data/images_series")
#labels_series_dir = pathlib.Path(r"D:/Spider Data/labels_series")

#Laptop path
images_series_dir = pathlib.Path(r"D:/Spider Data/images_series")
labels_series_dir = pathlib.Path(r"D:/Spider Data/labels_series")

#Get lists of full dset
images_dir_list = os.listdir(images_dir)
labels_dir_list = os.listdir(labels_dir)

#Sort lists to make sure we get the correct pairs every time and that we're not going through the dset randomly 
images_dir_list = natsorted(images_dir_list)
labels_dir_list = natsorted(labels_dir_list)

#Some metrics to count how many images we've excluded from our final dset
space_excl_count = 0 
t1_excl_count = 0
t2_excl_count = 0

#Print lengths to make sure they're the same and see how many 3D images we have 
    #as well as to make sure they're the same length
print("images dset length", len(images_dir_list))
print("masks dset length", len(labels_dir_list))

#If same, comb through the dataset 
dirlen = len(images_dir_list)

for idx in range (0,dirlen):
    
    print(idx)

    #Get image paths in directory
    img_path = images_dir.joinpath(images_dir_list[idx])    
    label_path = labels_dir.joinpath(labels_dir_list[idx]) 

    #if the file name has SPACE in it, skip it since we're excluding SPACE images due to resolution
    if("SPACE" in img_path.name):
        space_excl_count = space_excl_count + 1
        continue 

    #if image without its t1/t2 counterpart also skip over

    #first split the file name at character _ to get series no and t1 or t2 separately 
    filename_split = images_dir_list[idx].split('_')

    #if the image is t1 without t2 counterpart 
    if("t1" in filename_split[1]):
        if((filename_split[0] + "_t2.mha") not in images_dir_list):
            t2_excl_count = t2_excl_count + 1
            continue 
    else: #if t2 without t1 counterpart 
        if((filename_split[0] + "_t1.mha") not in images_dir_list):
            t1_excl_count = t1_excl_count + 1
            continue  


    #if we are here in the loop it means the idx is at a t1-t2 pair 
    #write the pair of mha images to directory 
    image_sitk = sitk.ReadImage(img_path)
    label_sitk = sitk.ReadImage(label_path)

    sitk.WriteImage(image_sitk, images_series_dir.joinpath(images_dir_list[idx]))
    sitk.WriteImage(label_sitk, labels_series_dir.joinpath(labels_dir_list[idx]))
    

print("total SPACE images excluded", space_excl_count)
print("total t1 images excluded", t1_excl_count)
print("total t2 images excluded", t2_excl_count)
    

        

