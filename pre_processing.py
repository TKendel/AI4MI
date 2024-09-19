import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from torchvision import transforms


ground_truth = cv.imread('data\SEGTHOR\\train\gt\Patient_10_0055.png', cv.IMREAD_GRAYSCALE)

for i in range(20, 162):
    iterator = 0
    if i < 10:
        iterator = f'00{i}'
    if i > 9 and i < 100:
        iterator = f'0{i}'
    if  i > 100:
        iterator = f'{i}'
    original_image = cv.imread(f'data\SEGTHOR\\train\img\Patient_10_0{iterator}.png')

    # convert the image to grayscale
    gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)


    # for salt and pepper noise removal
    # blurred_image = cv.medianBlur(original_image, 3)

    # kernel = np.ones((5,5), np.uint8)
    erosion = cv.erode(gray, cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)), iterations = 1)

    # cv.imshow('image', erosion)
    # cv.waitKey(0)

    # binary thresholding, input should be grayscaled
    ret, threshold = cv.threshold(erosion, 60, 255, cv.THRESH_BINARY)
    edges = cv.Canny(threshold,100,200)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cv.imshow('test', edges)
    cv.waitKey(0)

    big_contours = []

    # # Compute the convex hull of the contour
    # for contour in contours:
    #     rectangle = cv.boundingRect(contour)
    #     if rectangle[2] < 100 or rectangle[3] < 100: 
    #         big_contours.append(rectangle)

    for c in contours:
        convexHull = cv.convexHull(c)
        cv.drawContours(original_image, [convexHull], 2, (255, 0, 0), 2)
        

    cv.imshow('image', original_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# titles = ['Original Image', 'Morphed Image']
# images = [original_image, threshold]
 
# for i in range(2):
#     plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()