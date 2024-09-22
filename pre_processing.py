import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from torchvision import transforms


# ground_truth = cv.imread('data\SEGTHOR\\train\gt\Patient_10_0055.png', cv.IMREAD_GRAYSCALE)

for i in range(20, 202):
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

    crop = gray[66:194]

    # for salt and pepper noise removal
    # blurred_image = cv.medianBlur(original_image, 3)

    # kernel = np.ones((5,5), np.uint8)
    erosion = cv.erode(crop, cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)), iterations = 1)


    cv.imshow('image', erosion)
    cv.waitKey(0)

    # binary thresholding, input should be grayscaled
    ret, threshold = cv.threshold(erosion, 60, 255, cv.THRESH_BINARY)

    cv.imshow('thresh', threshold)
    cv.waitKey(0)
    
    edges = cv.Canny(threshold,100,200)

    cv.imshow('first edges', edges)
    cv.waitKey(0)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv.contourArea, reverse=True)
    rect_areas = []
    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        rect_areas.append(w * h)
    avg_area = np.mean(rect_areas)

    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        cnt_area = w * h
        if cnt_area < 0.6 * avg_area:
            edges[y:y + h, x:x + w] = 0

    dilate = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_CROSS,(5,5)), iterations = 1)

    edges = cv.Canny(dilate,100,200)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


    cnt = sorted(contours, key=cv.contourArea, reverse=True)[0:3]
    # cv.drawContours(crop, cnt, -1, (0, 255, 255), 2)
    if cnt != None:
        spine_found = False	
        for contour in cnt:
            x,y,w,h = cv.boundingRect(contour)
            if y < 90 and y > 35 and x > 55 and x < 90:
                cv.rectangle(crop, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        print('No contours found! ')

    cv.imshow('Drawn', crop)
    cv.waitKey(0)

    big_contours = []

    # # Compute the convex hull of the contour
    # for contour in contours:
    #     rectangle = cv.boundingRect(contour)
    #     if rectangle[2] < 100 or rectangle[3] < 100: 
    #         big_contours.append(rectangle)
            

# titles = ['Original Image', 'Morphed Image']
# images = [original_image, threshold]
 
# for i in range(2):
#     plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()