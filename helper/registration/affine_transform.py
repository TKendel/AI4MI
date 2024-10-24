import numpy as np
import nibabel as nib
import os

from scipy.ndimage import center_of_mass
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import rotate


'''
Afffine transform on segmentations.

This is our own implementation of the aligment of the heart volumes.

This file reads both GT of patient 27, and utilizes the correct gt to allign the other, calculating
the distance to then apply it to the rest of the gt of all other patients.
'''

# ++LOAD NII.GZ GT VOLUMES++
reference = nib.load('data/segthor_train/train/Patient_27/GT2.nii.gz')  #loaad reference (correct) .nii.gz
ref_np = np.array(reference.dataobj)    # to np arraay

# ++EXAMPLE ON HOW TO SAVE THE VOLUME TO FILE++
#output = nib.Nifti1Image(ref_np, reference.affine)
#https://stackoverflow.com/questions/28330785/creating-a-nifti-file-from-a-numpy-array
#np.eye
#https://numpy.org/devdocs/reference/generated/numpy.eye.html
# Save image
#nib.save(output,  os.path.join('data/segthor_train/train/Patient_27', 'test.nii.gz') )

# Check for unique labels, print if needed
#uniq = np.unique(ref_np)

# ++DIVIDES THE LAYERS OF EACH MASK - THE OTHER LAYERS AND SAVING HAVE BEEN COMMENTED OUT, BUT LEFT FOR REFERENCE++
# Save individual layers
#single_layer = np.where(ref_np == 1, ref_np, 0) #np.where(conndition, if yes, if not) # ESOPHAGUS - MASK 1
#output = nib.Nifti1Image(single_layer, reference.affine)
#nib.save(output,  os.path.join('data/segthor_train/train/Patient_27', 'esophagus.nii.gz') )

heart_ref = np.where(ref_np == 2, ref_np, 0) #np.where(conndition, if yes, if not) # HEART - MASK 2
#output = nib.Nifti1Image(heart_ref, reference.affine)
#nib.save(output,  os.path.join('data/segthor_train/train/Patient_27', 'heart.nii.gz') )

#single_layer = np.where(ref_np == 3, ref_np, 0) #np.where(conndition, if yes, if not) # TRACHEA - MASK 3
#output = nib.Nifti1Image(single_layer, reference.affine)
#nib.save(output,  os.path.join('data/segthor_train/train/Patient_27', 'mask3.nii.gz') )

#single_layer = np.where(ref_np == 4, ref_np, 0) #np.where(conndition, if yes, if not) # AORTA - MASK 4
#output = nib.Nifti1Image(single_layer, reference.affine)
#nib.save(output,  os.path.join('data/segthor_train/train/Patient_27', 'mask4.nii.gz') )

# ++LOADS SHIFTED HEART++
broken = nib.load('data/segthor_train/train/Patient_27/GT.nii.gz')  #load image to fix .nii.gz
broken_np = np.array(broken.dataobj)    # to np arraay
broken_heart = np.where(broken_np == 2, broken_np, 0) # HEART - MASK 2

# ++ALIGNMENT PROCESS++
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

# ++AFTER CALCULATING TRANSFORMATION, APPLY SHIFTING TO ALL PATIENTS++
patient_list = list(range(1,41))        #there are 40 patients

#f"{patient_list[0]:02} #some f string and regex refreshing, used when creating the file path
#https://stackoverflow.com/questions/3505831/in-python-how-do-i-convert-a-single-digit-number-into-a-double-digits-string

for patient in patient_list:
        broken = nib.load( os.path.join('data/segthor_train/train/Patient_'+f"{patient:02}", 'GT.nii.gz') )  #loaad image to fix .nii.gz
        broken_np = np.array(broken.dataobj)    # to np arraay
        broken_heart = np.where(broken_np == 2, broken_np, 0) # HEART - MASK 2
        broken_but_heart = np.where(broken_np != 2, broken_np, 0) # Errase the heart

        broken_heart = rotate(broken_heart, angle=-26, reshape=False)   #Rotate matrix
        # We don't want partial classes (between 0 and 2), so we round everything that is not 0 to 2
        broken_heart =  np.where(broken_heart > 0, 2, 0)#.astype(np.int8)  #because we are reseting to 2, we get an error for sending int64, use .astype(np.int8) [used later on]

        # Transform broken heart
        broken_heart = np.roll(broken_heart, round(x), axis=0)
        broken_heart = np.roll(broken_heart, round(y), axis=1)
        broken_heart = np.roll(broken_heart, round(z), axis=2)

        reconstruction = broken_but_heart
        reconstruction = np.where(broken_heart != 2, reconstruction, 2).astype(np.uint8)    #where broken_heart does not have a 2, keep it as is, else change to a 2

        output = nib.Nifti1Image(reconstruction, broken.affine)
        nib.save(output,  os.path.join('data/segthor_train/train/Patient_'+f"{patient:02}", 'gt.nii.gz') )
        print('Patient', patient, '/40')

print("All exports have been saved")


# ++FOR REFERENCE, BELOW YOU MAY FIND THE PROCESS DESCRIBED ABOVE APPLIED TO A SINGLE SHIFTED VOLUME++

# Transform broken heart
'''
broken_heart = np.roll(broken_heart, round(x), axis=0)
broken_heart = np.roll(broken_heart, round(y), axis=1)
broken_heart = np.roll(broken_heart, round(z), axis=2)
'''
# Translate np
#https://stackoverflow.com/questions/44874512/how-to-translate-shift-a-numpy-array

# Save heart
'''
output = nib.Nifti1Image(broken_heart, reference.affine)
nib.save(output,  os.path.join('data/segthor_train/train/Patient_27', 'broken_heart.nii.gz') )
'''

# Example on patient - heart shift and rotation - full reconstruction export
'''
#17
broken = nib.load('data/segthor_train/train/Patient_17/GT.nii.gz')  #loaad image to fix .nii.gz
broken_np = np.array(broken.dataobj)    # to np arraay
broken_heart_17 = np.where(broken_np == 2, broken_np, 0) # HEART - MASK 2
broken_but_heart = np.where(broken_np != 2, broken_np, 0) # Errase the heart
print('SHAPES')
print(broken_heart_17.shape)
print(broken_but_heart.shape)

broken_heart_17 = rotate(broken_heart_17, angle=-26, reshape=False)   #Rotate matrix
#We don't want partial classes (between 0 and 2), so we round everything that is not 0 to 2
print(broken_heart_17.shape)

#roubd all values to two
broken_heart_17 =  np.where(broken_heart_17 > 0, 2, 0)#.astype(np.int8)  #because we are reseting to 2, we get an error for sending int64, use .astype(np.int8)

#Transform broken heart
broken_heart_17 = np.roll(broken_heart_17, round(x), axis=0)
broken_heart_17 = np.roll(broken_heart_17, round(y), axis=1)
broken_heart_17 = np.roll(broken_heart_17, round(z), axis=2)

reconstruction = broken_but_heart
reconstruction = np.where(broken_heart_17 != 2, reconstruction, 2).astype(np.int8)        #where broken_heart_17 does not have a 2, keep it as is, else change to a 2

output = nib.Nifti1Image(reconstruction, broken.affine)
nib.save(output,  os.path.join('data/segthor_train/train/Patient_17', 'broken_heart.nii.gz') )
'''
