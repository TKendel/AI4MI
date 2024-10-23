import os
import random
import matplotlib.pyplot as plt
import cv2


def get_image_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]

original_folder = 'data/SEGTHOR/train/img'
transformed_folder = 'data/SEGTHOR_tmp/train/img'
original_paths = get_image_paths(original_folder)
transformed_paths = get_image_paths(transformed_folder)

assert len(original_paths) == len(transformed_paths), "Folders do not have the same number of images."

nr_samples = 5
random_indices = random.sample(range(len(original_paths)), nr_samples)
fig, axes = plt.subplots(2, nr_samples, figsize=(15, 6))

for i, idx in enumerate(random_indices):
    original_image = cv2.imread(original_paths[idx], cv2.IMREAD_GRAYSCALE)
    transformed_image = cv2.imread(transformed_paths[idx], cv2.IMREAD_GRAYSCALE)
    
    patient_id = original_paths[idx][-11:]
    
    axes[0, i].imshow(original_image, cmap='gray')
    axes[0, i].set_title(f'Original {patient_id}')
    axes[0, i].axis('off')
    axes[1, i].imshow(transformed_image, cmap='gray')
    axes[1, i].set_title(f'Transformed {patient_id}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()



def get_image_paths0000(folder):
    # Filter only files ending with '000.png'
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('000.png')]

original_folder = 'data/SEGTHOR/train/img'
transformed_folder = 'data/SEGTHOR_tmp/train/img'
original_paths = get_image_paths0000(original_folder)
transformed_paths = get_image_paths0000(transformed_folder)

assert len(original_paths) == len(transformed_paths), "Folders do not have the same number of images."

nr_samples = 5
random_indices = random.sample(range(len(original_paths)), nr_samples)
fig, axes = plt.subplots(2, nr_samples, figsize=(15, 6))

for i, idx in enumerate(random_indices):
    original_image = cv2.imread(original_paths[idx], cv2.IMREAD_GRAYSCALE)
    transformed_image = cv2.imread(transformed_paths[idx], cv2.IMREAD_GRAYSCALE)
    
    patient_id = original_paths[idx][-11:]
    
    axes[0, i].imshow(original_image, cmap='gray')
    axes[0, i].set_title(f'Original {patient_id}')
    axes[0, i].axis('off')
    axes[1, i].imshow(transformed_image, cmap='gray')
    axes[1, i].set_title(f'Transformed {patient_id}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
