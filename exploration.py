import os
from collections import defaultdict
import SimpleITK as sitk
from collections import Counter

def count_segmentations_per_patient(image_dir):
    """
    Count the number of GT segmentation for each patient based on filenames in a directory.
    Args:
    - image_dir (str): Path to the directory containing patient slice images.
    Returns:
    - gt_per_patient (dict): Dictionary where keys are patient IDs and values are the number of segmentations.
    """
    gt_per_patient = defaultdict(int)
    for patient_folder in  os.listdir(image_dir):
        patient_path = os.path.join(image_dir, patient_folder)
        if os.path.isdir(patient_path):
            for filename in os.listdir(patient_path):
                if filename.startswith("GT"):
                    gt_per_patient[patient_folder] += 1
                    

    return gt_per_patient
# Example usage:
# image_directorygt = os.path.join("data", "segthor_train", "train")
# nr_segmentations = count_segmentations_per_patient(image_directorygt)
# print(nr_segmentations)



def count_patients_in_set(image_dir):
    """
    Count the number of unique patients based on filenames in a directory.
    Args:
    - image_dir (str): Path to the directory containing patient slice images.
    Returns:
    - num_patients (int): Number of unique patients in the directory.
    """
    patient_ids = set()

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            # Assuming the format is like "Patient_03_000.png"
            patient_id = '_'.join(filename.split('_')[:2])
            patient_ids.add(patient_id)

    return len(patient_ids)

# Example usage:
# image_directory = "./data/SEGTHOR/train/img"
# num_patients = count_patients_in_set(image_directory)
# print(f"Number of unique patients in train: {num_patients}")



def count_slices_in_set(image_dir):
    """
    Count the number of slices based on filenames in a directory.
    Args:
    - image_dir (str): Path to the directory containing patient sliced images.
    Returns:
    - num_patients (int): Number of  slices in the directory.
    """
    count = 0

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            count+=1
    return count
# Example usage:
# image_directory = "./data/SEGTHOR/train/img"
# num_slices = count_slices_in_set(image_directory)
# print(f"Number of slices: {num_slices}")

# from utils import count_slices_per_patient, count_slices_per_patient
# x  = count_slices_per_patient(os.path.join("data", "SEGTHOR", "train", "img"))
# print(max(x.values()))



def classdistribution(base_path, output_file):
    # Initialize an empty counter for class distribution
    class_distribution = Counter()

    # Process each patient one by one to avoid memory issues
    for patient_id in range(1, 41):
        padded_patient_id = str(patient_id).zfill(2) 
        gt_path = os.path.join(base_path, f"Patient_{padded_patient_id}", "GT.nii.gz")    

        if os.path.exists(gt_path):
            # Load the GT segmentation using SimpleITK
            gt_image = sitk.ReadImage(gt_path)
            gt_array = sitk.GetArrayFromImage(gt_image)

            # Flatten the array and update the class distribution
            flattened_gt = gt_array.flatten()
            class_distribution.update(flattened_gt)
        else:
            print(f"GT file not found for Patient_{padded_patient_id}")

    # Write the class distribution to a text file
    with open(output_file, 'w') as f:
        f.write("Class Distribution:\n")
        for organ_class, count in class_distribution.items():
            f.write(f"Class {organ_class}: {count} voxels\n")
    
    return class_distribution

# Define the base path and output file
base_path = "./data/segthor_train/train/"
output_file = './class_distribution.txt'

# Run the function and write the output to a text file
# classdist = classdistribution(base_path, output_file)

def calculate_class_distribution(class_voxels, exclude_background=False):
    if exclude_background:
        class_voxels = {key: value for key, value in class_voxels.items() if key != 0}
    
    # Total number of voxels
    total_voxels = sum(class_voxels.values())
    
    # Calculate percentage distribution
    class_distribution = {key: (value / total_voxels) * 100 for key, value in class_voxels.items()}
    
    return class_distribution

def read_class_distribution_from_txt(file_path):
    """
    Reads class distribution from a text file and returns a dictionary of class voxel counts.
    
    Parameters:
    file_path (str): The path to the text file containing class distribution.
    
    Returns:
    dict: A dictionary where keys are class labels and values are voxel counts.
    """
    class_voxels = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Class') and not line.startswith('Class Distribution'):
                parts = line.split(':')
                class_id = int(parts[0].strip().split()[1])
                voxel_count = int(parts[1].strip().split()[0])
                class_voxels[class_id] = voxel_count
    
    return class_voxels

# Now read from the file and return the class distribution
class_voxels_from_file = read_class_distribution_from_txt("class_distribution.txt")
class_perc = calculate_class_distribution(class_voxels_from_file, exclude_background=True)
print(class_perc)