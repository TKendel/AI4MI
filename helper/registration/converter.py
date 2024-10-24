import numpy as np
import nibabel as nib

from os import walk, remove, rename
from os.path import join, isfile


'''
Converter used for creating compressed nifti files
'''

path = 'data\segthor_train\\train'
# Iterate over files in directory
for subdir, dirs, files in walk(path):
    for file in files:
        # Look for new files 
        if file[-4:] == '.nii':
            #TODO: Uncomment to rename the old file and make space for the new one
            if isfile(join(subdir, 'GT.nii.gz')):
                rename(join(subdir, 'GT.nii.gz'), join(subdir, 'GT_OG.nii.gz'))

            image = nib.load(join(subdir, file))

            # Update data type:
            new_dtype = np.uint8
            image.set_data_dtype(new_dtype)

            # Compress to gzip
            nib.save(image, join(subdir, 'GT.nii.gz'))

            # Remove files
            remove(join(subdir, file))
