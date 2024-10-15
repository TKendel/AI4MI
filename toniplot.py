import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread("presentation\Patient_10_0050.png", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("presentation\\beranPatient_10_0050.png", cv.IMREAD_GRAYSCALE)
img3 = cv.imread("presentation\\berPatient_10_0050.png", cv.IMREAD_GRAYSCALE)
img4 = cv.imread("presentation\\ergc2bPatient_10_0050.png", cv.IMREAD_GRAYSCALE)
img5 = cv.imread("presentation\\ergcbPatient_10_0050.png", cv.IMREAD_GRAYSCALE)
img6 = cv.imread("presentation\\erPatient_10_0050.png", cv.IMREAD_GRAYSCALE)



titles = ['Original', 'BERAN', 'BER', 'ERGC2B', 'ERGC15B', 'ER']
images = [img1, img2, img3, img4, img5, img6]

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
