import cv2 as cv
import glob
import numpy as np

from preprocessing import Preprocessing


tableXY = np.array([False])
current_patient = '03'

print('Processing images')

try:
    for filepath in glob.iglob('data/SEGTHOR/train/img/*.png'):
        if filepath[-11:-9] == current_patient:
            ct_scan = cv.imread(filepath, cv.IMREAD_GRAYSCALE)

            pp = Preprocessing(ct_scan, f'data/SEGTHOR_tmp/train/img/{filepath[-19:]}')

            pp.bilateralFilter()
            pp.equalize()
            tableXY = pp.removeTable(tableXY)
            pp.save()
            
        else:
            tableXY = np.array([False])

        current_patient = filepath[-11:-9]

    print('Done with the processing of the images')

except:
    print("An error occured while loading the preprocessing pipeline")



