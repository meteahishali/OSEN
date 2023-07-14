import SimpleITK as sitk
import numpy as np
import os

def nii_to_npy(inPath, slices):
    files = os.listdir(inPath) # nii.fz files
    xx = []
    for i in range(0, len(files)):
        vol = sitk.ReadImage(inPath + files[i])
        mri_vol = sitk.GetArrayFromImage(vol)

        fileName = files[i].split('.')[0]
        slice_indices = slices[fileName]
        
        for j in slice_indices:
            j = int(j)
            mri_slice = mri_vol[j]

            mri_slice = np.array(mri_slice, dtype = 'float32')
            mri_slice = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min())

            xx.append(mri_slice)
    return xx

# Training and testing MRIs.
input_trainPath = 'diencephalon/training-images/'
input_testPath = 'diencephalon/testing-images/'

# Load training and testing slice indices.
train_slices = np.load('train_slices.npy', allow_pickle='TRUE').item()
test_slices = np.load('test_slices.npy', allow_pickle='TRUE').item()

# nii to npy.
x_train = np.array(nii_to_npy(input_trainPath, train_slices))
x_test = np.array(nii_to_npy(input_testPath, test_slices))

# Storing in npy.
np.save('x_train.npy', x_train)
np.save('x_test.npy', x_test)