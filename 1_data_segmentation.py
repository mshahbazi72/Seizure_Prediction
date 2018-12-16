import parameters as params
import sys
import numpy as np
np.random.seed(569)

from chb_edf_file import ChbEdfFile
from chb_label_wrapper import ChbLabelWrapper
from glob import glob
from os.path import join
import os
import h5py
from natsort import natsorted
from sklearn.model_selection import train_test_split


def get_segment_label(eeg_label):
    s = sum(eeg_label) * 1.0 / len(eeg_label)
    if s >= 0.5:
        return 1
    else:
        return 0

def subsampling(signal, rate):
    return signal[:, ::rate]



dataset_path = '../../dataset/'
patient_id = sys.argv[1]
segments_interictal_train_path = '../segments/chb' + str(patient_id) + '/interictal_train/'
segments_interictal_test_path = '../segments/chb' + str(patient_id) + '/interictal_test/'
segments_seizures_path = '../segments/chb' + str(patient_id) + '/seizures/'
patient_path = join(dataset_path, 'chb' + patient_id)
if not os.path.exists(segments_interictal_train_path):
    os.makedirs(segments_interictal_train_path)
if not os.path.exists(segments_interictal_test_path):
    os.makedirs(segments_interictal_test_path)
if not os.path.exists(segments_seizures_path):
    os.makedirs(segments_seizures_path)
# print(patient_path)

################################ Select seizures and interictal files ###########################################
# print("Patient", patient_id, ':\n')
files_pathes = natsorted(glob(join(patient_path, '*.edf')))
label_path = glob(join(patient_path, '*.txt'))[0]
labels = ChbLabelWrapper(label_path)
seizure_intervals = labels.get_seizure_interval()
seizures_idx = [i for i in range(len(seizure_intervals)) if len(seizure_intervals[i])]
seizure_intervals = [seizure_intervals[i] for i in range(len(seizure_intervals)) if len(seizure_intervals[i])]
seizure_files = [files_pathes[i] for i in range(len(files_pathes)) if i in seizures_idx]
interictal_files = [files_pathes[i] for i in range(len(files_pathes)) if i not in seizures_idx]
inter_test_index = range(len(interictal_files))
np.random.shuffle(inter_test_index)
inter_test_index = inter_test_index[:max(min(len(seizure_files), len(interictal_files)), int(0.4 * len(interictal_files)))]
test_interictal_files = [interictal_files[i] for i in range(len(interictal_files)) if i in inter_test_index]
train_interictal_files = [interictal_files[i] for i in range(len(interictal_files)) if i not in inter_test_index]
#################################################################################################################

sampling_f = params.sampling_f
segment_duration = params.segment_duration          # seconds
preictal_ioverlap = params.preictal_ioverlap          # 0.75 overlap
intericatl_ioverlap = params.intericatl_ioverlap         # 0.5 overlap
preictal_duration = params.preictal_duration     # seconds
preictal_samples_num = preictal_duration * sampling_f
subsampling_rate = params.subsampling_rate
sampling_f = int(sampling_f / subsampling_rate)
n_segment_samples = segment_duration * sampling_f
seqNum = params.seqNum
#################################################################################################################

# print 'creating seizures segments...\n'

for i, f in enumerate(seizure_files):

    # print(f)

    segments = []
    labels = []
    seizure_start, seizure_end = seizure_intervals[i]

    eeg_file = ChbEdfFile(f)
    eeg_data = eeg_file.get_data()[:,:seizure_start]
    ch, signal_l = eeg_data.shape
    label_signal = np.zeros(signal_l)
    label_signal[max(0, len(label_signal) - preictal_samples_num) : len(label_signal)] = 1

    ss_eeg_data = eeg_data[:, ::subsampling_rate]
    ss_label_sig = label_signal[::subsampling_rate]

    signal_l = ss_eeg_data.shape[1]
    n_segments = int(signal_l / n_segment_samples)

    if patient_id in ['11', '14', '16', '17', '18', '19', '20', '21', '22']:
        ss_eeg_data = np.delete(ss_eeg_data, [4, 9, 12, 17, 22], axis=0)
    elif patient_id == '15':
        ss_eeg_data = np.delete(ss_eeg_data, [4, 9, 13, 18, 23, 29], axis=0)

    ch, _= ss_eeg_data.shape

    for n in range(preictal_ioverlap * (n_segments - 1) +1):

        eeg_multichannel_segment = ss_eeg_data[:, int(n_segment_samples/preictal_ioverlap) * n : int(n_segment_samples/preictal_ioverlap) * n + n_segment_samples]       # (n_channel, n_segment_samples)
        eeg_multichannel_segment = eeg_multichannel_segment.reshape([ch, seqNum, -1])
        label = get_segment_label(ss_label_sig[int(n_segment_samples/preictal_ioverlap) * n : int(n_segment_samples/preictal_ioverlap) * n + n_segment_samples])
        segments.append(eeg_multichannel_segment.transpose([1, 0, 2]))
        labels.append(label)

    segments = np.array(segments)

    with h5py.File(join(segments_seizures_path, 'seizure'+str(i)+'.h5'), 'w') as f:
        f.create_dataset('segments', data=segments)
        f.create_dataset('labels', data=labels)

    eeg_file.close()
    del eeg_file

#################################################################################################################

# print 'creating interictal train segments...\n'

for i, f in enumerate(train_interictal_files):

    # print(f)

    segments = []

    eeg_file = ChbEdfFile(f)
    eeg_data = eeg_file.get_data()
    ch, signal_l = eeg_data.shape

    ss_eeg_data = eeg_data[:, ::subsampling_rate]

    signal_l = ss_eeg_data.shape[1]
    n_segments = int(signal_l / n_segment_samples)

    if patient_id in ['11', '14', '16', '17', '18', '19', '20', '21', '22']:
        ss_eeg_data = np.delete(ss_eeg_data, [4, 9, 12, 17, 22], axis=0)
    elif patient_id == '15':
        ss_eeg_data = np.delete(ss_eeg_data, [4, 9, 13, 18, 23, 29], axis=0)

    ch,_= ss_eeg_data.shape

    for n in range(intericatl_ioverlap * (n_segments - 1) +1):
        eeg_multichannel_segment = ss_eeg_data[:, int(n_segment_samples/intericatl_ioverlap) * n : int(n_segment_samples/intericatl_ioverlap) * n + n_segment_samples]       # (n_channel, n_segment_samples)
        eeg_multichannel_segment = eeg_multichannel_segment.reshape([ch, seqNum, -1])
        segments.append(eeg_multichannel_segment.transpose([1, 0, 2]))

    segments = np.array(segments)
    labels = np.zeros(len(segments))
    # print segments.shape
    with h5py.File(join(segments_interictal_train_path, 'interictal_train'+str(i)+'.h5'), 'w') as f:
        f.create_dataset('segments', data=segments)
        f.create_dataset('labels', data=labels)

    eeg_file.close()
    del eeg_file

#################################################################################################################

# print 'creating interictal test segments...\n'

for i, f in enumerate(test_interictal_files):

    # print(f)

    segments = []

    eeg_file = ChbEdfFile(f)
    eeg_data = eeg_file.get_data()
    ch, signal_l = eeg_data.shape

    ss_eeg_data = eeg_data[:, ::subsampling_rate]

    signal_l = ss_eeg_data.shape[1]
    n_segments = int(signal_l / n_segment_samples)

    if patient_id in ['11', '14', '16', '17', '18', '19', '20', '21', '22']:
        ss_eeg_data = np.delete(ss_eeg_data, [4, 9, 12, 17, 22], axis=0)
    elif patient_id == '15':
        ss_eeg_data = np.delete(ss_eeg_data, [4, 9, 13, 18, 23, 29], axis=0)

    ch, _= ss_eeg_data.shape

    for n in range(intericatl_ioverlap * (n_segments - 1) +1):
        eeg_multichannel_segment = ss_eeg_data[:, int(n_segment_samples/intericatl_ioverlap) * n : int(n_segment_samples/intericatl_ioverlap) * n + n_segment_samples]       # (n_channel, n_segment_samples)
        eeg_multichannel_segment = eeg_multichannel_segment.reshape([ch, seqNum, -1])
        segments.append(eeg_multichannel_segment.transpose([1, 0, 2]))

    segments = np.array(segments)
    # print segments.shape
    labels = np.zeros(len(segments))

    with h5py.File(join(segments_interictal_test_path, 'interictal_test'+str(i)+'.h5'), 'w') as f:
        f.create_dataset('segments', data=segments)
        f.create_dataset('labels', data=labels)

    eeg_file.close()
    del eeg_file

with open('../seizures_num.txt', 'w') as f:
    f.write(str(len(seizures_idx)))
