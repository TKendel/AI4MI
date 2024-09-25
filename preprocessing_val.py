import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image in grayscale
# img = cv.imread('data\SEGTHOR\\train\img\Patient_03_0010.png', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read!"  # Check if the image was successfully loaded

val_image_folder = r"C:\Users\mirth\OneDrive\Documenten\Studie AI\Master AI\AI for Medical Imaging\Group Project Medical Imaging\AI4MI\data\SEGTHOR_fixed\val\img"

# List all files in the validation folder
image_files = [f for f in os.listdir(val_image_folder) if f.endswith('.png')]

# Loop through each image file in the folder
for image_file in image_files:
    # Construct the full file path
    image_path = os.path.join(val_image_folder, image_file)

    # Load the image in grayscale
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, f"Image {image_file} could not be read!"

    # # Apply median blur (optional, you can skip this if not needed)
    # img_blurred = cv.medianBlur(img, 1)

    # === Apply Contrast Enhancement (CLAHE) ===
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # clipLimit controls the amount of contrast enhancement
    img_contrast = clahe.apply(img)  # Enhance contrast

    # === Apply Global Thresholding ===
    ret, th1 = cv.threshold(img_contrast, 10, 255, cv.THRESH_BINARY)

    # Normalize the image (to range [0, 1])
    img_normalized = img_contrast.astype(np.float32) / 255.0

    # Save the preprocessed image
    new_filename = os.path.splitext(image_file)[0] + '_processed.png'  # Add suffix to the filename
    new_image_path = os.path.join(val_image_folder, new_filename)  # Full path for saving
    cv.imwrite(new_image_path, rotated_img)  # Save the rotated (and preprocessed) image

    # === Generate Gaussian noise ===
    # mean = 0
    # std_dev = 0.05  # Small standard deviation for subtle noise
    # gaussian_noise = np.random.normal(mean, std_dev, img.shape)

    # Add the Gaussian noise to the normalized image
    #noisy_img = img_normalized + gaussian_noise

    # Clip the values to stay within valid range [0, 1] and convert back to [0, 255]
    noisy_img_clipped = np.clip(img_normalized, 0, 1) * 255.0
    noisy_th1 = noisy_img_clipped.astype(np.uint8)

    # === Apply Slight Rotation to Simulate Patient Orientation Variations ===
    # angle = np.random.uniform(-10, 10)  # Random small rotation angle (-10 to +10 degrees)
    # (h, w) = img.shape[:2]
    # center = (w // 2, h // 2)  # Rotation around the center of the image

    # # Get the rotation matrix and apply the affine transformation
    # rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    # rotated_img = cv.warpAffine(img_normalized, rotation_matrix, (w, h))

    # === Display the original, enhanced, and rotated images ===
    titles = ['Original Image', 'Contrast Enhanced Image']
    images = [img, img_contrast]

    for i in range(2):
        plt.subplot(2, 1, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()