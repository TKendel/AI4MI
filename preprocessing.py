import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms, adjust_gamma, adjust_log, equalize_adapthist

class Preprocessing:

    def __init__(self, img, path):
        self.img = img
        self.path = path

    def normalize(self):
        '''
        Normalize image by adding a low and high value which should be included in the grayscale colour range,
        If it is a 0, 1 the output will have the darkest black from the inital and whites whites typically resulting in bones! 
        '''
        self.img = cv.normalize(self.img, None, 0, 255, cv.NORM_MINMAX)

    def equalize(self):
        '''
        Spread out the colour histogram of the image resulting in a higher contrast image
        '''
        self.img = cv.equalizeHist(self.img)

    def adaptiveEqualize(self, clip_limit=0.03):
        '''

        https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
        '''
        self.img = equalize_adapthist(self.img, clip_limit=clip_limit)
 
    def bilateralFilter(self):
        '''
        Apply a filter that blures content to remove noise but keeps edges. More on it can be read here:
        https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
        '''
        self.img = cv.bilateralFilter(self.img, 2, 3, 3)

    def __drawBox(self, lines):
        '''
        Function for drawing boxed given lines.
        '''
        a,b,c = lines.shape
        for i in range(a):
            if lines[i][0][2] > 174:
                self.img = cv.rectangle(self.img, (lines[i][0][2]+5, 0), (255,255), (0, 0, 0), -1)

        return lines

    def removeTable(self, tableXY):
        '''
        Removing table by finding edges and then its biggest vertical lines which are used as coordinates to cut out the table
        '''
        if not tableXY.any():
            edges = cv.Canny(self.img, 50, 150)
            minLineLength=180
            tableXY = cv.HoughLinesP(image=edges, rho=1, theta=np.pi, threshold=60, lines=np.array([]), minLineLength=minLineLength, maxLineGap=60)

            tableXY = self.__drawBox(tableXY)

            return tableXY
        else:
            return self.__drawBox(tableXY)
        
    def histogramMatching(self, reference):
        '''
        Tries to match the colour histogram by using the reference image as a reference on image.
        Tricky to use since the refernce should in a way move with the given image.
        '''
        self.img = match_histograms(self.img, reference)

    def gammaCorrection(self, gamma_value):
        '''
        Turn down the contrast 
        '''
        self.img = np.uint8(adjust_gamma(self.img, gamma_value))

    def logCorrection(self, log_value):
        '''
        Turn up the contrast 
        '''
        self.img = np.uint8(adjust_log(self.img, log_value))

    def save(self):
        '''
        Save image
        '''
        plt.imsave(self.path, self.img, cmap='gray')