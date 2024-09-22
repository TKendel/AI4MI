import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2

# load the image
img = cv.imread('data\SEGTHOR\\train\img\Patient_03_0010.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read!" # Check if the image was successfully loaded

# Apply median blur
img = cv.medianBlur(img, 1)
 
# Apply global thresholding
ret, th1 = cv.threshold(img, 10, 255, cv.THRESH_BINARY)

titles = ['Grayscaled OG', 'Global Thresholding']
images = [img, th1]
 
# for i in range(2):
#     plt.subplot(2,1,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()


## adding Gaussian noise to the thresholded image (th1)
## Gaussian noise can help prevent overfitting by making the model generalize better

# Generate Gaussian noise
mean = 0
std_dev = 25 # don't know if this the correct std
gaussian_noise = np.random.normal(mean, std_dev, img.shape)

# Add the Gaussian noise to the image
noisy_img = img.astype(np.float32) + gaussian_noise

# Clip the values to stay within valid range and convert back to uint8
noisy_th1 = np.clip(noisy_img, 0, 255).astype(np.uint8)

# Display the original, thresholded, and noisy images
titles_gaussian = ['Original (Grayscaled)', 'Thresholded Image (th1)', 'Noisy Thresholded Image']
images_gaussian = [img, th1, noisy_th1]

for i in range(3):
    plt.subplot(3, 1, i + 1), plt.imshow(images_gaussian[i], 'gray')
    plt.title(titles_gaussian[i])
    plt.xticks([]), plt.yticks([])

plt.show()

## Display the original, thresholded and noisy images
# cv.imshow('Original image', img_blur)
# cv.imshow('Thresholded Image', th1)
# cv.imshow('Noisy Thresholded Image', noisy_th1)
# cv.waitKey(0)
# cv.destroyAllWindows()






