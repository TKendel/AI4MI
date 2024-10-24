import os
import shutil
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from scipy.ndimage import affine_transform


'''
To correct the heart segmentation using the GT2 as a reference, you will need to apply an affine transformation to align the heart segment 
in GT2 with the heart segmentation in the GT for other patients. 

Steps:
1) Load the NIfTI files: Use Nibabel to load the GT and GT2 segmentations for Patient 27 and the segmentations for the other patients.

2) Identify the heart segmentation in both GT and GT2: You'll need to extract only the heart segmentation (label corresponding to the heart) 
from both GT and GT2. The heart label is 2 -->  use NumPy to extract the segmentation.

3) Register (Align) GT and GT2: Use SimpleITK for image registration to compute the affine transformation between GT and GT2. 
Apply the affine transformation to the heart segmentation of GT.

4) Apply the transformation to other patients: Once you have the affine transformation that aligns the heart in GT2, apply 
this transformation to the heart segmentation in the other patients' GT.

5) Replace the misaligned heart with the corrected one: For each patient, after applying the affine transformation, replace the misaligned 
heart segmentation with the transformed one from GT.
'''

original_dir = './data/segthor_train/train'
transformed_dir = './data/transformed_segthor_train/train'

# Create a copy of the segthor_train/train directory
if not os.path.exists(transformed_dir):
    shutil.copytree(original_dir, transformed_dir)

def load_patient_data(patient_id, gt_filename='GT.nii.gz'):
    '''
    This function loads the ground truth NIfTI file for a patient and returns the data.
    '''
    patient_path = os.path.join(f"{transformed_dir}/Patient_{str(patient_id).zfill(2)}", gt_filename)
    return nib.load(patient_path).get_fdata()

def visualize_segmentations(gt2_heart_sitk, transformed_gt_heart_sitk):
    '''
    Visualize segmentation
    '''
    # Convert both images to numpy arrays
    gt2_heart_array = sitk.GetArrayFromImage(gt2_heart_sitk)
    transformed_gt_heart_array = sitk.GetArrayFromImage(transformed_gt_heart_sitk)

    # Choose the middle slice to visualize
    middle_slice_index = gt2_heart_array.shape[0] // 2
    
    # Plotting
    plt.figure(figsize=(12, 6))

    # First subplot: GT2 (fixed image) with the transformed GT overlayed
    plt.subplot(1, 2, 1)
    plt.imshow(gt2_heart_array[middle_slice_index], cmap="Reds", alpha=0.5 )
    plt.imshow(transformed_gt_heart_array[middle_slice_index], cmap="hot", alpha=0.5)  # Overlay the transformed GT heart
    plt.title("GT2 Heart (Fixed) with GT Heart (Moving) overlay")

    # Second subplot: GT Heart only after transformation
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_gt_heart_array[middle_slice_index], cmap="gray")
    plt.title("GT Heart (Moving Image after Initial Transform)")

    plt.show()

def calc_affine_transform(gt_data, gt2_data):
    '''
    Compute the affine transformation between GT and GT2 for Patient 27.
    '''
    # Extract heart segmentations
    gt_heart = (gt_data == 2).astype(np.uint8)    
    gt2_heart = (gt2_data == 2).astype(np.uint8)

    # Convert heart segmentations to SimpleITK images
    gt_heart_sitk = sitk.GetImageFromArray(gt_heart.astype(np.float32))
    gt2_heart_sitk = sitk.GetImageFromArray(gt2_heart.astype(np.float32))

    # Perform registration to get the affine transformation
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    affine_transform = sitk.AffineTransform(gt_heart_sitk.GetDimension())
    registration_method.SetInitialTransform(affine_transform)
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=0.01, numberOfIterations=200)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
  
    final_transform = registration_method.Execute(gt2_heart_sitk, gt_heart_sitk)

    # Debugging: Print the affine matrix and translation after registration
    print("Affine transformation matrix:\n", final_transform.GetMatrix())
    print("Affine transformation translation:\n", final_transform.GetTranslation())

    # Extract and return the affine transformation matrix and translation
    affine_matrix = np.array(final_transform.GetMatrix()).reshape(3, 3)  # Convert tuple to NumPy array and reshape to 3x3
    translation = np.array(final_transform.GetTranslation())  # Convert tuple to NumPy array
    
    return affine_matrix, translation


def apply_affine_to_patient(patient_id, affine_matrix, translation_vector):
        padded_patient_id = str(patient_id).zfill(2)
        patient_data = load_patient_data(padded_patient_id, "GT.nii.gz")
        heart_segmentation = (patient_data == 2).astype(np.uint8)

        print("shape affine", affine_matrix.shape)
        print(" affine", affine_matrix)

        transformed_volume = affine_transform(
            heart_segmentation,
            affine_matrix,  # Invert the matrix for affine_transform
            offset=translation_vector,
            order=1  # Use linear interpolation for smooth results, order=0 would be nearest-neighbor
        )
        
        # Create a copy of the original segmentation and replace the heart segmentation
        transformed_data = np.copy(patient_data)
        transformed_data[patient_data == 2] = 0
        transformed_data[transformed_volume > 0.5] = 2  # Threshold to keep values as binary

        # Save the transformed volume back into a NIfTI file
        save_as = 'GT3.nii.gz' if patient_id == 27 else 'GT.nii.gz'
        volume_file_path = os.path.join(f"./data/transformed_segthor_train/train/Patient_{str(padded_patient_id).zfill(2)}/Patient_{patient_id}.nii.gz")  # Path to the volume file
        nifti_img =  nib.load(volume_file_path)

        transformed_nifti_img = nib.Nifti1Image(transformed_data, nifti_img.affine)
        nib.save(transformed_nifti_img, f'./data/transformed_segthor_train/train/Patient_{padded_patient_id}/{save_as}')

        print(f"Transformed NIfTI file saved for Patient {patient_id:02d}")

# Load data for Patient 27 and compute the affine transformation
gt_patient_27_data = load_patient_data(27, 'GT.nii.gz')
gt2_patient_27_data = load_patient_data(27, 'GT2.nii.gz')
affine_transform_matrix, affine_translation= calc_affine_transform(gt_patient_27_data, gt2_patient_27_data)
print(affine_transform)
# Apply the transformation to all patients
for patient_id in range(1, 41):
    if patient_id == 27: # for debugging i only use patient 27 rather than all the patients
        apply_affine_to_patient(patient_id, affine_transform_matrix, affine_translation)
