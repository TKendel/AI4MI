#Format data for nnunet raw fodler

import os
import shutil


id = 550        
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








