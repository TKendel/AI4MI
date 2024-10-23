import numpy as np
import nibabel as nib
import os

from scipy.ndimage import center_of_mass
from scipy.ndimage import rotate

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


'''
This file loads three (3) GT of patient 27, the initial correct gt, the gt with the shifted heart out of place
and that calculated with the given transformation matrix. The shifted heart is positioned through our methods
and the overlap is calculated.

This script is not meant to be run mulstiple times, and was only used to better understand the behaviour of
our implementation
'''

# ++READ CORRECT GT AND SHIFTED GT++
# Reads and saves the same nii file
reference = nib.load('data/OLD DATA/segthor_train_og/train/Patient_27/GT2.nii.gz')  #loaad reference (correct) .nii.gz
ref_np = np.array(reference.dataobj)    # to np arraay
heart_ref = np.where(ref_np == 2, ref_np, 0) #np.where(conndition, if yes, if not) # HEART - MASK 2

# Shifted heart (broken)
broken = nib.load('data/OLD DATA/segthor_train_og/train/Patient_27/GT.nii.gz')  #load image to fix .nii.gz
broken_np = np.array(broken.dataobj)    # to np arraay
broken_heart = np.where(broken_np == 2, broken_np, 0) # HEART - MASK 2

# ++ALLIGN SHIFTED HEART++
# First rotate, as it is done at the center of the volume, not heart
broken_heart = rotate(broken_heart, angle=-26, reshape=False)   #Rotate matrix #reshape=False makes sure not to expand matrix (crop when rotating)
#https://stackoverflow.com/questions/53171057/numpy-matrix-rotation-for-any-degrees

# Calculate euclidean distance between centers of mass
center_ref = center_of_mass(heart_ref)
center_broken = center_of_mass(broken_heart)

x = center_ref[0]-center_broken[0]
y = center_ref[1]-center_broken[1]
z = center_ref[2]-center_broken[2]

print('Distance:', x,y,z)

# Transform broken heart
broken_heart = np.roll(broken_heart, round(x), axis=0)
broken_heart = np.roll(broken_heart, round(y), axis=1)
broken_heart = np.roll(broken_heart, round(z), axis=2)

# ++LOAD HEART WITH SOLUTION MATRIX TRANSFORMATION++
solution = nib.load('data/OLD DATA/segthor_train_og/train/Patient_27/GT2.nii.gz')  #loaad reference (correct) .nii.gz
sol_np = np.array(solution.dataobj)    # to np arraay
heart_sol = np.where(sol_np == 2, sol_np, 0) #np.where(conndition, if yes, if not) # HEART - MASK 2

# ++CALCULATE TP, TN, FP, FN OF TWO VOLUMES - VOLUME: 0 BACKGROUND, ELSE TRUTH++
def mesure_overlap(gt, prediction):
        if gt.shape != prediction.shape:
                print('ERROR: the dimention of the volumes do not match')
                return None

        # One hot
        one1 = np.where(gt != 0, 1, 0)
        #one2 = np.where(prediction != 0, 1, 0)
        five2 = np.where(prediction != 0, 5, 0)

        # Math trick
        #one2 = one2*5       #vol array of 0 and 5
        result = one1 - five2
        tp = np.sum(np.where(result == -4, 1, 0))    #if both true, 1-5 = -4
        tn = np.sum(np.where(result == 0, 1, 0))   #if both false 0-0 = 0
        fp = np.sum(np.where(result == -5, 1, 0))  #if pred is wrongfully true 0-5 = -5
        fn = np.sum(np.where(result == 1, 1, 0))  #if pred is wrongfully false 1-0 = 1

        return tp, tn, fp, fn

# ++REPORT OVERLAP OF HEARTS++
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
