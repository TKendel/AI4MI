import itk
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



# # Import Default Parameter Map
# parameter_object = itk.ParameterObject.New()
# parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
# parameter_object.AddParameterMap(parameter_map_rigid)


# Import Default Parameter Map
parameter_object = itk.ParameterObject.New()
parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
parameter_object.AddParameterMap(parameter_map_rigid)
parameter_map_affine= parameter_object.GetDefaultParameterMap('affine')
parameter_object.AddParameterMap(parameter_map_affine)
# parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')
# parameter_object.AddParameterMap(parameter_map_bspline)

# parameter_object.AddParameterFile('Transform_1 (-).txt')
# parameter_object.SetParameter(0, "WriteResultImage", "true")

# transforms = itk.transformread('Transform_1.tfm')

# fixed_image = itk.imread('data\SEGTHOR_tmp\\train\gt\Patient_27_0077.png')
# fixed_image = itk.imread('test1.png', itk.F)

fixed_mask = itk.imread('3DModels\GT2.nii.gz_1.nii', itk.UC)
moving_mask = itk.imread('3DModels\GT.nii.gz_1.nii', itk.UC)

test_mask = itk.imread('3DModels\GT.nii.gz_6_1.nii', itk.UC)



# ## Convert to HSV
# hsv = cv.cvtColor(fixed_image, cv.COLOR_BGR2HSV)

# mask = cv.inRange(hsv, (0, 0, 74), (0,0,150))

# ## Slice the heart
# imask = mask > 0
# heartT = np.zeros_like(fixed_image, np.uint8)
# heartT[imask] = fixed_image[imask]
# heart_fixed = np.asarray(heartT)
# heart_fixed = itk.GetImageFromArray(heartT)

# plt.imshow(heart_fixed)
# plt.show()

# moving_image = itk.imread('data\SEGTHOR\\train\gt\Patient_27_0028.png')
# moving_image = itk.imread('test2.png', itk.F)

# ## Convert to HSV
# hsv = cv.cvtColor(moving_image, cv.COLOR_BGR2HSV)

# mask = cv.inRange(hsv, (0, 0, 74), (0,0,150))

# ## Slice the heart
# imask = mask > 0
# heartF = np.zeros_like(moving_image, np.uint8)
# heartF[imask] = moving_image[imask]
# heart_moving = np.asarray(heartT)
# heart_moving = itk.GetImageFromArray(heartF)

# plt.imshow(heart_moving)
# plt.show()
# registered_image, params = itk.elastix_registration_method(fixed_image, moving_imag )

# Call registration function
result_image, result_transform_parameters = itk.elastix_registration_method(
    fixed_mask, moving_mask,
    parameter_object=parameter_object,
    log_to_console=True)


# result_image_transformix = itk.transformix_filter(
#     moving_mask,
#     transforms)

result_image_transformix = itk.transformix_filter(
    test_mask,
    result_transform_parameters)

# image = itk.rescale_intensity_image_filter(result_image_transformix, output_minimum=0, output_maximum=255)
itk.imwrite(result_image_transformix, "binary_thinning2.nii")
# plt.imshow(result_image)
# plt.show()

# result_image_transformix = itk.transformix_filter(
#     test,
#     params)

# plt.imshow(result_image_transformix)
# plt.show()