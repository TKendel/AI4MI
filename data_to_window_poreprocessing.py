#Format data for nnunet raw fodler

import os
import shutil



import numpy as np
import nibabel as nib
import os

from scipy.ndimage import center_of_mass
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import rotate



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #np will give a DeprecationWarning on "volume_np = np.array(volume.dataobj)"






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









"""
# Reads and saves the same nii file
reference = nib.load('data/segthor_train/train/Patient_27/Patient_27.nii.gz')  #load volume .nii.gz
ref_np = np.array(reference.dataobj)    # to np array

print("range:",np.ptp(ref_np))
print(np.amin(ref_np))
print(np.max(ref_np))



# Reads and saves the same nii file
reference = nib.load('data/segthor_train/train/Patient_24/Patient_24.nii.gz')  #load volume .nii.gz
ref_np = np.array(reference.dataobj)    # to np array

print("range:",np.ptp(ref_np))
print(np.amin(ref_np))
print(np.max(ref_np))



# Reads and saves the same nii file
reference = nib.load('data/segthor_train/train/Patient_13/Patient_13.nii.gz')  #load volume .nii.gz
ref_np = np.array(reference.dataobj)    # to np array

print("range:",np.ptp(ref_np))
print(np.amin(ref_np))
print(np.max(ref_np))

"""







def normalize_array(a):
        min = np.amin(a)
        max = np.max(a)
        a = (a-min)/(max-min)
        return a

"""
def shift_avrg(a):
        avg = np.average(a)
        shape = a.shape

        #can change later, we will assume 3D
        size = shape[0]*shape[1]*shape[2]

        shift = -1*avg/size

        a = a+shift
        return a
"""




def shift_avrg(a):      #I understand this does not normalize them, but it helps greatly
        avg = np.average(a)
        shape = a.shape

        shift = -1*avg

        a = a+shift
        return a



patient_list = list(range(1,41)) #there are 40 patients

"""
for patient in patient_list:
        print('Processig patient', patient, '/40')
        volume_file = os.path.join('data/segthor_train/train/Patient_'+f"{patient:02}", 'Patient_'+f"{patient:02}"+'.nii.gz')
        reference = nib.load(volume_file)  #load volume .nii.gz
        ref_np = np.array(reference.dataobj)    # to np array

        print("range:",np.ptp(ref_np))
        print(np.amin(ref_np))
        print(np.max(ref_np))
        print('shape', ref_np.shape)

        b = normalize_array(ref_np)
        print("range:",np.ptp(b))
        print(np.amin(b))
        print(np.max(b))
        print('shape', b.shape)
"""










#window_pairs = [[400, 20],[300, 0], [600, 0], [200, 100], [320,30], [1800,400]]      #add pairs as lists, with upper followerd by lower
#window_pairs = [[100, 0],[200, 100], [300, 200], [400, 300], [500,400], [1000,500],[0,-500],[100, -100], [100, -200]]      #range tests, [300, 200] bone for patient 1
window_pairs = [[1000, 600]]    


patient_list = list(range(1,41)) #there are 40 patients

for patient in patient_list:
        print('Processig patient', patient, '/40')
        volume_file = os.path.join('data/segthor_train/train/Patient_'+f"{patient:02}", 'Patient_'+f"{patient:02}"+'.nii.gz')
        volume_destination = os.path.join('Dataset010_SEGT/nnUNet_raw_data_base/nnUNet_raw_data/Dataset600_segthor/imagesTr/', 'SEGT_'+f"{patient:03}"+'_0000.nii.gz')
        shutil.copyfile(volume_file, volume_destination)      #source --> destination

        volume = nib.load(volume_file)  #loaad reference .nii.gz
        volume_np = np.array(volume.dataobj)    # to np arraay

        #debug "normalization"
        """
        print('average original', np.average(volume_np))
        print("range:",np.ptp(volume_np))
        print('min',np.amin(volume_np))
        print('max',np.max(volume_np))
        print('shape', volume_np.shape)
        shape =  volume_np.shape 

        norm = normalize_array(volume_np)       #normalize betweeon 0 and 1
        norm = norm*1000         # stretch normalization to 0-1000
        print('average nborm', np.average(norm))
        print("range:",np.ptp(norm))
        print('min',np.amin(norm))
        print('max',np.max(norm))
        print('shape', norm.shape)

        shift = shift_avrg(volume_np)
        print('average shifted', np.average(shift))
        print("range:",np.ptp(shift))
        print('min',np.amin(shift))
        print('max',np.max(shift))
        print('shape', shift.shape)

        output = nib.Nifti1Image(norm, volume.affine)      #windowed.astype(np.uint8) The model is using uint8
        nib.save(output,  os.path.join('Dataset010_SEGT/nnUNet_raw_data_base/nnUNet_raw_data/Dataset600_segthor/imagesTr/', 'SEGT_'+f"{patient:03}"+'_0000_norm.nii.gz') )

        output = nib.Nifti1Image(shift, volume.affine)      #windowed.astype(np.uint8) The model is using uint8
        nib.save(output,  os.path.join('Dataset010_SEGT/nnUNet_raw_data_base/nnUNet_raw_data/Dataset600_segthor/imagesTr/', 'SEGT_'+f"{patient:03}"+'_0000_shift.nii.gz') )
        """

        shift = shift_avrg(volume_np)
        #apply all windows
        for i in range(len(window_pairs)): #equivalent to "for window in window_pairs:"
                print('window', i+1,'/',len(window_pairs))

                upper = window_pairs[i][0]
                lower = window_pairs[i][1]
                windowed = shift.copy()

                windowed[windowed > upper] = upper
                windowed[windowed < lower] = lower

                output = nib.Nifti1Image(windowed, volume.affine)      #windowed.astype(np.uint8) The model is using uint8
                nib.save(output,  os.path.join('Dataset010_SEGT/nnUNet_raw_data_base/nnUNet_raw_data/Dataset600_segthor/imagesTr/', 'SEGT_'+f"{patient:03}"+f'_000{i+1}.nii.gz') )



        segmentation_file = os.path.join('data/segthor_train/train/Patient_'+f"{patient:02}", 'gt.nii.gz')
        segmentation_destination = os.path.join('Dataset010_SEGT/nnUNet_raw_data_base/nnUNet_raw_data/Dataset600_segthor/labelsTr/', 'SEGT_'+f"{patient:03}"+'.nii.gz')
        shutil.copyfile(segmentation_file, segmentation_destination)      #source --> destination

        

print("All files have been copied and formatted")
print("windowed profiles have been added")






