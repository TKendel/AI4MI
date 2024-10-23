import os
import shutil
import numpy as np


def moveGTfiles(seg_path):
    '''
    Move GT files
    '''
    val_list = [1,2,13,16,21,22,28,30,35,39]

    for root, dirs, files in os.walk(seg_path):
        for file in files:
            if file.endswith("GT.nii.gz") and int(root[-2:]) in val_list:
                shutil.copyfile(os.path.join(root, file), f'group-10\\val\gt\Patient_{root[-2:]}.nii.gz')
                print(root)
                print(file)


def updateNpyShape(seg_path):
    '''
    Transpose npy file dimensions
    '''
    for root, dirs, files in os.walk(seg_path):
        for file in files:
            if file.endswith("val.npy"):
                data = np.load(os.path.join(root, file))
                if data.shape[1] == 10:

                    print(file)
                    test = np.transpose(data, (0, 2, 1))

                    assert test.shape[2] == 10

                    np.save(os.path.join(root, file), test)
                    print(f"Saved {file} with the shape {test.shape}")

                elif data.shape[1] == 1967:
                    print(file)
                    test = np.transpose(data, (0, 2, 1))

                    assert test.shape[2] == 1967

                    np.save(os.path.join(root, file), test)
                    print(f"Saved {file} with the shape {test.shape}")


seg_path = "group-10\\val"

#moveGTfiles(seg_path)
#updateNpyShape(seg_path)

# Sanity check
for root, dirs, files in os.walk(seg_path):
    for file in files:
        if file.endswith("val.npy"):
            print(root, file)
            data = np.load(os.path.join(root, file))
            print(data.shape)
