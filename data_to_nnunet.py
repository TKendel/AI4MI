import os
import shutil


'''
Format data for nnunet raw fodler

This file reads files from the format we utlized in the project and outputs copies
in the format expected by nnU-Net.
It was not expected that this would be ran multiple times, so no much fexibility 
was built into it and requiers editing if the segthor file format changes 
(unlikely within our project spawn, but a concern for a longer project)
'''

id = 550        #ID for the dataset folder (destination). Format needed by nnU-Net
print(f"{id:03}")

patient_list = list(range(1,41)) #there are 40 patients

for patient in patient_list:
        volume_file = os.path.join('data/segthor_train/train/Patient_'+f"{patient:02}", 'Patient_'+f"{patient:02}"+'.nii.gz')
        volume_destination = os.path.join(f'Dataset010_SEGT/nnUNet_raw_data_base/nnUNet_raw_data/Dataset{id:03}_segthor/imagesTr/', 'SEGT_'+f"{patient:03}"+'_0000.nii.gz')
        shutil.copyfile(volume_file, volume_destination)      #source --> destination

        segmentation_file = os.path.join('data/segthor_train/train/Patient_'+f"{patient:02}", 'gt.nii.gz')
        segmentation_destination = os.path.join(f'Dataset010_SEGT/nnUNet_raw_data_base/nnUNet_raw_data/Dataset{id:03}_segthor/labelsTr/', 'SEGT_'+f"{patient:03}"+'.nii.gz')
        shutil.copyfile(segmentation_file, segmentation_destination)      #source --> destination

        print('Patient', patient, '/40')

print("All files have been copied and formatted")
