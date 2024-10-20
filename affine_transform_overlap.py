#Afffine transform on segmentations

#import cv2
import numpy as np
import nibabel as nib
import os

from scipy.ndimage import center_of_mass
#from scipy.spatial.transform.Rotation import align_vectors
#from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import rotate

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 



# Reads and saves the same nii file
reference = nib.load('data/OLD DATA/segthor_train_og/train/Patient_27/GT2.nii.gz')  #loaad reference (correct) .nii.gz
ref_np = np.array(reference.dataobj)    # to np arraay
heart_ref = np.where(ref_np == 2, ref_np, 0) #np.where(conndition, if yes, if not) # HEART - MASK 2

#Shifted heart (broken)
broken = nib.load('data/OLD DATA/segthor_train_og/train/Patient_27/GT.nii.gz')  #load image to fix .nii.gz
broken_np = np.array(broken.dataobj)    # to np arraay
broken_heart = np.where(broken_np == 2, broken_np, 0) # HEART - MASK 2




#First rotate, as it is done at the center of the volume, not heart
broken_heart = rotate(broken_heart, angle=-26, reshape=False)   #Rotate matrix #reshape=False makes sure not to expand matrix (crop when rotating)
#https://stackoverflow.com/questions/53171057/numpy-matrix-rotation-for-any-degrees

#Calculate euclidean distance between centers of mass
center_ref = center_of_mass(heart_ref)
center_broken = center_of_mass(broken_heart)

x = center_ref[0]-center_broken[0]
y = center_ref[1]-center_broken[1]
z = center_ref[2]-center_broken[2]

print('Distance:', x,y,z)

#Transform broken heart
broken_heart = np.roll(broken_heart, round(x), axis=0)
broken_heart = np.roll(broken_heart, round(y), axis=1)
broken_heart = np.roll(broken_heart, round(z), axis=2)


# Solution - given transformation
solution = nib.load('data/OLD DATA/segthor_train_og/train/Patient_27/GT2.nii.gz')  #loaad reference (correct) .nii.gz
sol_np = np.array(solution.dataobj)    # to np arraay
heart_sol = np.where(sol_np == 2, sol_np, 0) #np.where(conndition, if yes, if not) # HEART - MASK 2


def mesure_overlap(gt, prediction):
        if gt.shape != prediction.shape:
                print('ERROR: the dimention of the volumes do not match')
                return None

        #one hot
        one1 = np.where(gt != 0, 1, 0)
        #one2 = np.where(prediction != 0, 1, 0)
        five2 = np.where(prediction != 0, 5, 0)

        #Math trick
        #one2 = one2*5       #vol array of 0 and 5
        result = one1 - five2
        tp = np.sum(np.where(result == -4, 1, 0))    #if both true, 1-5 = -4
        tn = np.sum(np.where(result == 0, 1, 0))   #if both false 0-0 = 0
        fp = np.sum(np.where(result == -5, 1, 0))  #if pred is wrongfully true 0-5 = -5
        fn = np.sum(np.where(result == 1, 1, 0))  #if pred is wrongfully false 1-0 = 1

        return tp, tn, fp, fn

tp, tn, fp, fn = mesure_overlap(heart_ref, broken_heart)
print('\n')
print('Reference vs Broken')
print('True Positives:', tp)
print('Total cases:', tp+tn+fp+fn)
print('Broken TP/Reference heart volume', tp/(np.sum(np.where(heart_ref == 2, 1, 0))))  #one hot encodes the ref heart mask and then sums results
iou = (tp)/(tp+fp+fn)
print('IoU:', iou)


tp, tn, fp, fn = mesure_overlap(heart_sol, broken_heart)
print('\n')
print('Solution vs Broken')
print('True Positives:', tp)
print('Total cases:', tp+tn+fp+fn)
print('Broken TP/Solution heart volume', tp/(np.sum(np.where(heart_sol == 2, 1, 0))))  #one hot encodes the ref heart mask and then sums results
iou = (tp)/(tp+fp+fn)
print('IoU:', iou)









"""
patient_list = list(range(1,41))        #there are 40 patients

#f"{patient_list[0]:02}
#https://stackoverflow.com/questions/3505831/in-python-how-do-i-convert-a-single-digit-number-into-a-double-digits-string

for patient in patient_list:
        broken = nib.load( os.path.join('data/segthor_train/train/Patient_'+f"{patient:02}", 'GT.nii.gz') )  #loaad image to fix .nii.gz
        broken_np = np.array(broken.dataobj)    # to np arraay
        broken_heart = np.where(broken_np == 2, broken_np, 0) # HEART - MASK 2
        broken_but_heart = np.where(broken_np != 2, broken_np, 0) # Errase the heart

        broken_heart = rotate(broken_heart, angle=-26, reshape=False)   #Rotate matrix
        #We don't want partial classes (between 0 and 2), so we round everything that is not 0 to 2
        broken_heart =  np.where(broken_heart > 0, 2, 0)#.astype(np.int8)  #because we are reseting to 2, we get an error for sending int64, use .astype(np.int8) [used later on]

        #Transform broken heart
        broken_heart = np.roll(broken_heart, round(x), axis=0)
        broken_heart = np.roll(broken_heart, round(y), axis=1)
        broken_heart = np.roll(broken_heart, round(z), axis=2)

        reconstruction = broken_but_heart
        reconstruction = np.where(broken_heart != 2, reconstruction, 2).astype(np.uint8)    #where broken_heart does not have a 2, keep it as is, else change to a 2

        output = nib.Nifti1Image(reconstruction, broken.affine)
        nib.save(output,  os.path.join('data/segthor_train/train/Patient_'+f"{patient:02}", 'gt.nii.gz') )
        print('Patient', patient, '/40')

print("All exports have been saved")
"""