import SimpleITK as sitk
import numpy as np 


#--------------3D MHA---------------------------------------------
#https://gist.github.com/mrajchl/ccbd5ed12eb68e0c1afc5da116af614a
def resample_img(itk_image, out_spacing, is_label):

    # Resample images to 2 0.6 0.6 spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor) #Label interpolation 
    else:
        resample.SetInterpolator(sitk.sitkBSpline) #Image Interpolation 

    return resample.Execute(itk_image)

#Transpose 3d image to correct coordinates 
def transpose(image):
    size = image.GetSize()
    if size[0] > size[1] or size[0] > size[2]:
        if size[1] < size[2]:
            return sitk.PermuteAxes(image, [1, 0, 2])
        elif size[2] < size[1]:
            return sitk.PermuteAxes(image, [2, 0, 1])
    return image



#--------------2D MHA---------------------------------------------


    