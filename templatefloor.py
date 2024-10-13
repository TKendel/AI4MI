import cv2 as cv
import glob
import numpy as np
import os
from preprocessing import Preprocessing


tableXY = np.array([False])
current_patient = '03'

print('Processing images')
try:
    for filepath in glob.iglob('data/SEGTHOR/val/img/*.png'):
        new_patient = filepath[-11:-9]
        if new_patient != current_patient:
            tableXY = np.array([False])
        current_patient = new_patient
        ct_scan = cv.imread(filepath, cv.IMREAD_GRAYSCALE)

        pp = Preprocessing(ct_scan, f'data/SEGTHOR_tmp2/val/img/{filepath[-19:]}')

        try:
            pp.bilateralFilter()               # Noise reduction
            pp.equalize()
            tableXY = pp.removeTable(tableXY)  # Remove table (if present)
            pp.normalize()                     # Intensity normalization
            pp.adaptiveEqualize()              # Contrast enhancement
            pp.save()                          # Save the processed image
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    print('Done with the processing of the images')

except:
    print("An error occured while loading the preprocessing pipeline")


# Check if the number of slices is the same in both directories
# original_files = glob.glob(os.path.join('data/SEGTHOR/val/img/', '*.png'))
# processed_files = glob.glob(os.path.join('data/SEGTHOR_tmp/val/img/', '*.png'))

# original_count = len(original_files)
# processed_count = len(processed_files)

# if original_count == processed_count:
#     print(f"Success! Both directories have the same number of slices: {original_count}")
# else:
#     print(f"Warning! Mismatch in slice count: {original_count} original vs {processed_count} processed")
