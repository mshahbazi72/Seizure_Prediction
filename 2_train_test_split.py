import parameters as params
import sys
import h5py
import numpy as np
from glob import glob
from os.path import join
import os
from natsort import natsorted

patient_id = sys.argv[1]
seizure_index = int(sys.argv[2])
seizure_index -= 1
np.random.seed(736)

segments_interictal_train_path = '../segments/chb' + str(patient_id) + '/interictal_train/'
segments_interictal_test_path = '../segments/chb' + str(patient_id) + '/interictal_test/'
segments_seizures_path = '../segments/chb' + str(patient_id) + '/seizures/'

interictal_test_path = '../selected_segments/chb' + str(patient_id) + '/interictal_test/'
train_path = '../selected_segments/chb' + str(patient_id) + '/train/'
val_path = '../selected_segments/chb' + str(patient_id) + '/validation/'
seizures_test_path = '../selected_segments/chb' + str(patient_id) + '/seizures_test/'

if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(val_path):
    os.makedirs(val_path)
if not os.path.exists(interictal_test_path):
    os.makedirs(interictal_test_path)
if not os.path.exists(seizures_test_path):
    os.makedirs(seizures_test_path)

#################################################################################################################

seizures_mats_pathes = natsorted(glob(join(segments_seizures_path, '*.h5')))
interictal_train_mats_path = natsorted(glob(join(segments_interictal_train_path, '*.h5')))

seizures_mats_pathes[seizure_index]
os.system("cp " + seizures_mats_pathes[seizure_index] + " " + seizures_test_path)

os.system("cp " + segments_interictal_test_path + "* " + interictal_test_path)

#################################################################################################################
val_split = params.val_split

train_preictal_segments = []
train_preictal_labels = []
for m in seizures_mats_pathes:
    print m
    with h5py.File(m, 'r') as f:
        segments = np.array(f.get('segments'))
        labels = np.array(f.get('labels'))

    segments = list(segments[labels == 1])
    labels = list(labels[labels == 1])
    train_preictal_segments += segments
    train_preictal_labels += labels
preictal_n = len(train_preictal_labels)

val_preictal_segments = train_preictal_segments[int(val_split * len(train_preictal_segments)) :]
val_preictal_labels = train_preictal_labels[int(val_split * len(train_preictal_labels)) :]
train_preictal_segments = train_preictal_segments[: int(val_split * len(train_preictal_segments))]
train_preictal_labels = train_preictal_labels[: int(val_split * len(train_preictal_labels))]

#################################################################################################################

train_inter_labels = []
for m in interictal_train_mats_path:
    with h5py.File(m, 'r') as f:
        labels = list(np.array(f.get('labels')))
    train_inter_labels += labels
train_inter_labels = np.array(train_inter_labels)

indices = range(len(train_inter_labels))

np.random.shuffle(indices)
indices = indices[:preictal_n]
indices.sort()
train_indices = indices[:int(val_split * len(indices))]
val_indices = indices[int(val_split * len(indices)):]
val_inter_labels = train_inter_labels[val_indices]
train_inter_labels = train_inter_labels[train_indices]
del indices


train_inter_segments = []
val_inter_segments = []

offset = 0
train_p = 0
val_p = 0
for m in interictal_train_mats_path:
    print m
    with h5py.File(m, 'r') as f:
        segments = np.array(f.get('segments'))
    for i in range(len(segments)):
        if train_p < len(train_indices):
            if i + offset == train_indices[train_p]:
                train_inter_segments.append(segments[i])
                train_p += 1
        if val_p < len(val_indices):
            if i + offset == val_indices[val_p]:
                val_inter_segments.append(segments[i])
                val_p += 1
        if train_p == len(train_indices) and val_p == len(val_indices):
            break
    if train_p == len(train_indices) and val_p == len(val_indices):
        break
    offset += len(segments)

#################################################################################################################

val_segments = np.array(val_inter_segments + val_preictal_segments)
val_labels = np.array(list(val_inter_labels) + val_preictal_labels)

batch_size = params.batch_size
n = int(len(val_labels)/batch_size) + 1
for b in range(n):
    print(b)
    if val_segments[b * batch_size : min(len(val_labels), (b+1) * batch_size)].shape[0]:
        with h5py.File(val_path + 'Batch'+str(b)+'.h5', 'w') as f:
            f.create_dataset('segments', data=val_segments[b * batch_size : min(len(val_labels), (b+1) * batch_size)])
            f.create_dataset('labels', data=val_labels[b * batch_size : min(len(val_labels), (b+1) * batch_size)])

print val_segments.shape, val_labels.shape
del val_labels, val_segments, val_preictal_labels, val_inter_labels, val_preictal_segments, val_inter_segments

train_inter_segments = train_inter_segments[: int(val_split * len(train_inter_segments))]
train_inter_labels = train_inter_labels[: int(val_split * len(train_inter_labels))]

train_segments = np.array(train_inter_segments + train_preictal_segments)
del train_inter_segments, train_preictal_segments
train_labels = np.array(list(train_inter_labels) + train_preictal_labels)
del train_inter_labels, train_preictal_labels

indices = range(len(train_labels))
np.random.shuffle(indices)
train_segments = train_segments[indices]
train_labels = train_labels[indices]
print train_segments.shape, train_labels.shape

n = int(len(train_labels)/batch_size) + 1
for b in range(n):
    print(b)
    if train_segments[b * batch_size : min(len(train_labels), (b+1) * batch_size)].shape[0]:
        with h5py.File(train_path + 'Batch'+str(b)+'.h5', 'w') as f:
            f.create_dataset('segments', data=train_segments[b * batch_size : min(len(train_labels), (b+1) * batch_size)])
            f.create_dataset('labels', data=train_labels[b * batch_size : min(len(train_labels), (b+1) * batch_size)])
