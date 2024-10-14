import cv2 as cv
import glob
import numpy as np
import os
from preprocessing import Preprocessing


tableXY = np.array([False])
current_patient = '03'

print('Processing images')
try:
    for filepath in glob.iglob('data/SEGTHOR/train/img/*.png'):
        new_patient = filepath[-11:-9]
        if new_patient != current_patient:
            tableXY = np.array([False])
        current_patient = new_patient
        ct_scan = cv.imread(filepath, cv.IMREAD_GRAYSCALE)

        pp = Preprocessing(ct_scan, f'data/SEGTHOR_tmp/train/img/{filepath[-19:]}')

        try:
            # pp.bilateralFilter()               # Noise reduction
            pp.equalize()
            # pp.adaptiveEqualize()              # Contrast enhancement
            pp.normalize()                     # Intensity normalization
            tableXY = pp.removeTable(tableXY)  # Remove table (if present)
            # pp.logCorrection(1)
            pp.gammaCorrection(2)
            pp.normalize()                     # Intensity normalization
            # pp.closing()
            # pp.CLAHEClipping()
            pp.save()                          # Save the processed image
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    print('Done with the processing of the images')

except:
    print("An error occured while loading the preprocessing pipeline")
