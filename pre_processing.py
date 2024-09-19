import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from torchvision import transforms


img = cv.imread('data\SEGTHOR\\train\img\Patient_03_0010.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read!"
img = cv.medianBlur(img, 1)
 
ret, th1 = cv.threshold(img, 50, 255, cv.THRESH_BINARY)

titles = ['Grayscaled OG', 'Global Thresholding']
images = [img, th1]
 
for i in range(2):
    plt.subplot(2,1,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()