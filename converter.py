
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from torchvision import transforms

import nibabel as nib
img = cv.imread('data\segthor_train\\train\Patient_01\GT.nii\GT.nii')

image = nib.load('data\segthor_train\\train\Patient_01\GT.nii\GT.nii')

# update data type:
new_dtype = np.uint8  # for example to cast to int8.
image.set_data_dtype(new_dtype)

nib.save(image, 'my_image_new_datatype.nii')

image = cv.imread('my_image_new_datatype.nii').dtype


