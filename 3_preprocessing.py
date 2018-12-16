import parameters as params
import sys
import h5py
from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
import scipy
from glob import glob
from os.path import join
import os
from natsort import natsorted

patient_id = sys.argv[1]

np.random.seed(736)

segments_path = ['../selected_segments/chb' + str(patient_id) + '/interictal_test/',
                    '../selected_segments/chb' + str(patient_id) + '/train/',
                    '../selected_segments/chb' + str(patient_id) + '/seizures_test/',
                    '../selected_segments/chb' + str(patient_id) + '/validation/']

images_path = ['../images/chb' + str(patient_id) + '/interictal_test/',
                    '../images/chb' + str(patient_id) + '/train/',
                    '../images/chb' + str(patient_id) + '/seizures_test/',
                    '../images/chb' + str(patient_id) + '/validation/']
for p in images_path:
    if not os.path.exists(p):
        os.makedirs(p)

#################################################################################################################

sampling_f = params.sampling_f
subsampling_rate = params.subsampling_rate
sampling_f = int(sampling_f / subsampling_rate)

window_size = params.window_size * sampling_f
window_overlap = int(params.window_overlap * sampling_f)

########################v########################## train set ###################################################################

for q in range(len(segments_path)):
    mats_path = natsorted(glob(join(segments_path[q], "*.h5")))
    for b, s in enumerate(mats_path):

        print('Batch', b+1)

        with h5py.File(s, 'r') as hf:
            segments = np.array(hf.get('segments'))
            labels = np.array(hf.get('labels'))

        segments_n, segments_seq_n, segments_channel_n, segment_samples_n = segments.shape

        images = []
        for i in range(segments_n):
            sequence = []
            for s in range(segments_seq_n):
                multi_channel_image = []
                for ch in range(segments_channel_n):
                    seg = segments[i, s, ch]
                    f, t, img = stft(seg, sampling_f, nperseg= window_size, noverlap= window_overlap)
                    img = np.abs(img)
                    img = np.vstack((img[1:57, :], img[64: 117, :], img[124:, :]))  # Power line noise removal
                    f = np.hstack((f[1:57], f[64: 117], f[124:]))
                    multi_channel_image.append(img)
                sequence.append(multi_channel_image)
            images.append(sequence)

        images = np.asarray(images)
        print images.shape

        with h5py.File(join(images_path[q], 'Batch'+str(b)+'.h5'), 'w') as hf:
            hf.create_dataset('images', data=images)
            hf.create_dataset('labels', data=labels)

with h5py.File("../images/images_axises.h5", 'w') as hf:
    hf.create_dataset('f', data=f)
    hf.create_dataset('t', data=t)

#plt.pcolormesh(t, f, images[10][10])
#plt.colorbar()
# plt.show()
