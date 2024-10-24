import os
import shutil
import numpy as np


def moveGTfiles(root, file):
    '''
    Move GT files
    '''
    valList = [1,2,13,16,21,22,28,30,35,39]

    if int(file[-2:]) in valList:
        dest = f'group-10\\val\gt\Patient_{root[-2:]}.nii.gz'
        shutil.copyfile(os.path.join(root, file), dest)
        print(f"Moved file {file} to {dest}")


def updateNpyShape(root, file):
    '''
    Transpose npy file dimensions
    '''
    data = np.load(os.path.join(root, file))

    print(f"Transposing {file} with shape {data.shape}")
    newDataShape = np.transpose(data, (1, 2, 0))

    assert newDataShape.shape[2] == 100

    np.save(os.path.join(root, file), newDataShape)
    print(f"Saved {file} with the shape {newDataShape.shape}")


def getBestEpoch(root, file, best_epoch):
    '''
    Get best epoch of volume per class
    '''
    data = np.load(os.path.join(root, file))
    bestEpochVolume = data[:, :, best_epoch]

    addDimension = np.expand_dims(bestEpochVolume, axis=2)

    np.save(os.path.join(root, file), addDimension)
    print(f"Took out the best epoch given {best_epoch}, saved to {os.path.join(root, file)}")


updateShape = False
moveFiles = False
getBestMetrics = True
segPath = "group-10\\val"

f = open("data\BER\\best_epoch.txt", "r")
bestEpoch = int(f.readline())

for root, dirs, files in os.walk(segPath):
    for file in files:
        if file.endswith("val.npy"):
            if moveFiles == True:
                # seg_path should be the folder from which we want to transfer metrics
                moveGTfiles(root, file)
            if updateShape == True:
                updateNpyShape(root, file)
            if getBestMetrics == True:
                getBestEpoch(root, file, bestEpoch)

# Sanity check
for root, dirs, files in os.walk(segPath):
    for file in files:
        if file.endswith("val.npy"):
            print(root, file)
            data = np.load(os.path.join(root, file))
            print(data.shape)
