import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

'''
**Deprecated**

Used for manual segmentation of the heart. Works alright for current patient but we end up into problems when trying to extrapolate it over rest of patients.
'''

for i in range(20, 202):
    iterator = 0
    if i < 10:
        iterator = f'00{i}'
    if i > 9 and i < 100:
        iterator = f'0{i}'
    if  i > 100:
        iterator = f'{i}'
        
    original_image = cv.imread(f'data\SEGTHOR\\train\img\Patient_10_0{iterator}.png')

    # Convert the image to grayscale
    gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    #TODO Instead crop using a dialated heart mask for better focus maybe
    # Crop the image.
    crop = gray[66:194]

    # For salt and pepper noise removal
    # blurred_image = cv.medianBlur(original_image, 3)

    # Binary thresholding, input should be grayscaled
    ret, threshold = cv.threshold(crop, 60, 255, cv.THRESH_BINARY)

    # kernel = np.ones((5,5), np.uint8)
    erosion = cv.erode(threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)), iterations = 1)
    
    # Extract edges using Canny
    edges = cv.Canny(erosion,100,200)

    # Find big edges and sort sort them starting from the biggest
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv.contourArea, reverse=True)
    rect_areas = []
    # If bounding box around a contour is in the heart area save it
    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        rect_areas.append(w * h)
    avg_area = np.mean(rect_areas)

    # Apply a bounding box around the contours
    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        cnt_area = w * h
        if cnt_area < 0.6 * avg_area:
            edges[y:y + h, x:x + w] = 0

    # Dilate the contours so the connect
    dilate = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_RECT, (5,5)), iterations = 1)

    # Agagin get the contour and find 
    edges = cv.Canny(dilate,100,200)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cnt = sorted(contours, key=cv.contourArea, reverse=True)[0:3]
    if cnt != None:
        spine_found = False	
        for contour in cnt:
            x,y,w,h = cv.boundingRect(contour)
            if y < 90 and y > 35 and x > 55 and x < 90:
                cv.rectangle(crop, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        print('No contours found! ')

    titles = ['Threshold', 'Erosion', 'Edges', 'Final result']
    images = [threshold, erosion, edges, crop]
    
    for i in range(4):
        plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
