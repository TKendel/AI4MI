import os
import shutil


"""
Format data for nnunet raw fodler
"""
patient_list = list(range(1,41)) #there are 40 patients

for patient in patient_list:
        volume_file = os.path.join('data/segthor_train/train/Patient_'+f"{patient:02}", 'Patient_'+f"{patient:02}"+'.nii.gz')
        volume_destination = os.path.join('Dataset001_SEGT/nnUNet_raw/Dataset001_segthor/imagesTr/', 'SEGT_'+f"{patient:03}"+'_0000.nii.gz')
        shutil.copyfile(volume_file, volume_destination)      #source --> destination

        segmentation_file = os.path.join('data/segthor_train/train/Patient_'+f"{patient:02}", 'GT.nii.gz')
        segmentation_destination = os.path.join('Dataset001_SEGT/nnUNet_raw/Dataset001_segthor/labelsTr/', 'SEGT_'+f"{patient:03}"+'.nii.gz')
        shutil.copyfile(segmentation_file, segmentation_destination)      #source --> destination

        print('Patient', patient, '/40')

print("All files have been copied and formatted")
