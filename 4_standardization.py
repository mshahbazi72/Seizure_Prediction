import sys
import numpy as np

np.random.seed(49)

import h5py
from natsort import natsorted
from glob import glob
from os.path import join
import os
import matplotlib.pyplot as plt

patient_id = sys.argv[1]

images_path = ['../images/chb' + str(patient_id) + '/interictal_test/',
                    '../images/chb' + str(patient_id) + '/train/',
                    '../images/chb' + str(patient_id) + '/seizures_test/',
                    '../images/chb' + str(patient_id) + '/validation/']
normalized_images_path = ['../standardized_images/chb' + str(patient_id) + '/interictal_test/',
                    '../standardized_images/chb' + str(patient_id) + '/train/',
                    '../standardized_images/chb' + str(patient_id) + '/seizures_test/',
                    '../standardized_images/chb' + str(patient_id) + '/validation/']

for p in normalized_images_path:
    if not os.path.exists(p):
        os.makedirs(p)
# for visualization
with h5py.File("../images/images_axises.h5", 'r') as hf:
    f = np.array(hf.get('f'))
    t = np.array(hf.get('t'))

#####################################################################################################################


print("Calculating the mean o Patient", patient_id)

images_cnt = 0
means = []
stds=[]
sizes = []

mats_path = natsorted(glob(join(images_path[1], "*.h5")))
for ip in mats_path:
    print(ip)
    with h5py.File(ip, 'r') as f:
        images = np.array(f.get('images'))
    n, seq, ch, r, c = images.shape
    means.append(np.mean(images, axis=(0, 1, 2, 4)))
    stds.append(np.std(images, axis=(0, 1, 2, 4)))
    sizes.append(n * seq * ch * c)
means = np.array(means)
stds = np.array(stds)
sizes = np.array(sizes)
sums = sizes[:, np.newaxis] * means
patient_mean = np.sum(sums, axis=0) / np.sum(sizes)
sums = sizes[:, np.newaxis] * stds
patient_std = np.sum(sums, axis=0) / np.sum(sizes)

#####################################################################################################################

print("normalizing images of Patient", patient_id)

patient_mean = patient_mean[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
patient_std = patient_std[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

for q in range(len(images_path)):
    mats_path = natsorted(glob(join(images_path[q], "*.h5")))
    for b, ip in enumerate(mats_path):
        print(ip)
        with h5py.File(ip, 'r') as f:
            images = np.array(f.get('images'))
            labels = np.array(f.get('labels'))

        images = (images - patient_mean) / patient_std

        with h5py.File(join(normalized_images_path[q], 'Batch' + str(b + 1) + '.h5'), 'w') as f:
            f.create_dataset('images', data=images)
            f.create_dataset('labels', data=labels)
