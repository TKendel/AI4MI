#Format data for nnunet raw fodler

import os
import shutil



import numpy as np
import nibabel as nib
import os

from scipy.ndimage import center_of_mass
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import rotate










#EXAMPLE FOR A SINGLE FILE
"""
# Reads and saves the same nii file
reference = nib.load('data/segthor_train/train/Patient_27/Patient_27.nii.gz')  #load volume .nii.gz
ref_np = np.array(reference.dataobj)    # to np array

print(np.ptp(ref_np))
print(np.amin(ref_np))
print(np.max(ref_np))

upper = 400
lower = 20
windowed = ref_np.copy()

windowed[windowed > upper] = upper
windowed[windowed < lower] = lower

print(np.ptp(windowed))
print(np.amin(windowed))
print(np.max(windowed))

output = nib.Nifti1Image(windowed.astype(np.uint8), reference.affine)
nib.save(output,  'data_test/windowed.nii.gz' )
"""


reference = nib.load('data_test/windowed.nii.gz')  #loaad reference .nii.gz
ref_np = np.array(reference.dataobj)    # to np arraay





window_pairs = [[400, 20],[300, 0], [600, 0], [200, 100], [320,30], [1800,400]]      #add pairs as lists, with upper followerd by lower

patient_list = list(range(1,41)) #there are 40 patients

for patient in patient_list:
        print('Processig patient', patient, '/40')
        volume_file = os.path.join('data/segthor_train/train/Patient_'+f"{patient:02}", 'Patient_'+f"{patient:02}"+'.nii.gz')
        volume_destination = os.path.join('Dataset010_SEGT/nnUNet_raw_data_base/nnUNet_raw_data/Dataset600_segthor/imagesTr/', 'SEGT_'+f"{patient:03}"+'_0000.nii.gz')
        shutil.copyfile(volume_file, volume_destination)      #source --> destination

        volume = nib.load(volume_file)  #loaad reference .nii.gz
        volume_np = np.array(volume.dataobj)    # to np arraay
        #apply all windows
        for i in range(len(window_pairs)): #equivalent to "for window in window_pairs:"
                print('window', i+1,'/',len(window_pairs))

                upper = window_pairs[i][0]
                lower = window_pairs[i][1]
                windowed = volume_np.copy()

                windowed[windowed > upper] = upper
                windowed[windowed < lower] = lower

                output = nib.Nifti1Image(windowed, volume.affine)      #windowed.astype(np.uint8) The model is using uint8
                nib.save(output,  os.path.join('Dataset010_SEGT/nnUNet_raw_data_base/nnUNet_raw_data/Dataset600_segthor/imagesTr/', 'SEGT_'+f"{patient:03}"+f'_000{i+1}.nii.gz') )



        segmentation_file = os.path.join('data/segthor_train/train/Patient_'+f"{patient:02}", 'gt.nii.gz')
        segmentation_destination = os.path.join('Dataset010_SEGT/nnUNet_raw_data_base/nnUNet_raw_data/Dataset600_segthor/labelsTr/', 'SEGT_'+f"{patient:03}"+'.nii.gz')
        shutil.copyfile(segmentation_file, segmentation_destination)      #source --> destination

        

print("All files have been copied and formatted")
print("windowed profiles have been added")






