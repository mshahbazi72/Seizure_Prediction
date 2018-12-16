import parameters as params
import sys
import random as rn
import numpy as np
# np.random.seed(2 * int(sys.argv[2]))
# rn.seed(2*int(sys.argv[2]))
from keras.models import load_model
import h5py
from natsort import natsorted
from glob import glob
from os.path import join
import tensorflow as tf
import os
# os.environ['PYTHONHASHSEED'] = '0'
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# from keras import backend as K
# tf.set_random_seed(2*int(sys.argv[2]))
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

def seq_to_seg_labels(labels, seqNum):
    labels = [[l]*seqNum for l in labels]
    return np.array(labels).ravel()

def one_hot(labels, depth):
    labels1h = np.zeros((len(labels), depth))
    for i in range(len(labels)):
        labels1h[i, int(labels[i])] = 1
    return labels1h
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN

def get_smoothed_predictions(predictions, window_size, min_preictal_num):

    smoothed_predictions = []
    for i in range(smoothing_window, len(predictions)):
        window = predictions[i - smoothing_window:i]
        if sum(window) >= min_preictal_num:
            smoothed_predictions.append(1)
        else:
            smoothed_predictions.append(0)

    return np.array(smoothed_predictions)

def decrease_fp(smoothed_predictions, n_segment_samples, intericatl_ioverlap, preictal_samples_num):
    n_segments = int(preictal_samples_num / n_segment_samples)
    sph = preictal_ioverlap * (n_segments - 1) +1

    for i in range(len(smoothed_predictions)):
        if smoothed_predictions[i] == 1:
            smoothed_predictions[i+1 : i + sph] = 0
    return smoothed_predictions

def get_SPH(alarm_idx, total_seg_num, n_segment_samples):
    l = total_seg_num - alarm_idx
    sph = (l - 1) * int(n_segment_samples / preictal_ioverlap) + n_segment_samples
    return sph / (sampling_f * 60.0)    # Minutes


patient_id = sys.argv[1]

seizure_test_mats = natsorted(glob(join('../standardized_images/chb' + str(patient_id) + '/seizures_test/', '*.h5')))
interictal_test_mats = natsorted(glob(join('../standardized_images/chb' + str(patient_id) + '/interictal_test/', '*.h5')))
patient_CNN_model_path = join('../cnn_models', 'chb' + patient_id) + '/model.h5'
patient_LSTM_model_path = join('../lstm_models', 'chb' + patient_id) + '/model.h5'


class_num = params.class_num
smoothing_window = params.smoothing_window
min_preictal_num = params.min_preictal_num
sampling_f = params.sampling_f
segment_duration = params.segment_duration          # seconds
preictal_ioverlap = params.preictal_ioverlap          # 0.75 overlap
intericatl_ioverlap = params.intericatl_ioverlap         # 0.5 overlap
subsampling_rate = params.subsampling_rate
sampling_f = int(sampling_f / subsampling_rate)
n_segment_samples = segment_duration * sampling_f
seqNum = params.seqNum
preictal_duration = params.preictal_duration     # seconds
preictal_samples_num = preictal_duration * sampling_f

if not os.path.exists('../logs'):
    os.makedirs('../logs')

################################################## Network ###################################################################

cnn = load_model(patient_CNN_model_path)
lstm = load_model(patient_LSTM_model_path)

################################################ seizure test ###################################################################

seizure_segments = []
seizures_labels = []
print 'loading test seizure data...'
for p in seizure_test_mats:
    with h5py.File(p, 'r') as f:
        images = np.array(f.get('images'))
    seizure_segments = np.vstack((seizure_segments, images)) if len(seizure_segments) else images

seizure_segments = np.transpose(seizure_segments, [0, 1, 3, 4, 2])
_, s, r, c, ch = seizure_segments.shape
print seizure_segments.shape


################################################ lstm seizure ###################################################################

lstm_predicted_seizure = 0
lstm_SPH = 0

lstm_predictions = lstm.predict(seizure_segments)
lstm_predictions = np.argmax(lstm_predictions, axis=1)
smoothed_lstm_predictions = get_smoothed_predictions(lstm_predictions, smoothing_window, min_preictal_num)


if len(np.where(smoothed_lstm_predictions == 1)[0]):
    alarm_index = np.where(smoothed_lstm_predictions == 1)[0][0]
    lstm_SPH = get_SPH(alarm_index, len(smoothed_lstm_predictions), n_segment_samples)
    if len(smoothed_lstm_predictions) - preictal_ioverlap * (int(preictal_samples_num / n_segment_samples) - 1) +1 <= alarm_index:
        lstm_predicted_seizure = 1


smoothed_predictions = decrease_fp(smoothed_lstm_predictions, n_segment_samples, preictal_ioverlap, preictal_samples_num)


if np.sum(smoothed_predictions[len(smoothed_predictions) - preictal_ioverlap * (int(preictal_samples_num / n_segment_samples) - 1) +1 : -1]):
    lstm_predicted_seizure = 1
    print 'LSTM:', lstm_predicted_seizure, lstm_SPH
else:

    print 'LSTM: ', lstm_predicted_seizure, lstm_SPH

with open('../logs/seizure_'+patient_id + '.txt', 'a') as f:
    f.write(str(lstm_predicted_seizure)+'\n')
    f.write(str(lstm_SPH)+'\n')

################################################ interictal test ###################################################################
fP_num = 0
interictal_l = 0
print 'loading test interictal data...'
for p in interictal_test_mats:
    print p
    with h5py.File(p, 'r') as f:
        interictal_segments = np.array(f.get('images'))
    interictal_segments = np.transpose(interictal_segments, [0, 1, 3, 4, 2])

    predictions = lstm.predict(interictal_segments)
    predictions = np.argmax(predictions, axis=1)
    smoothed_predictions = get_smoothed_predictions(predictions, smoothing_window, min_preictal_num)
    smoothed_predictions = decrease_fp(smoothed_predictions, n_segment_samples, intericatl_ioverlap, preictal_samples_num)

    fP_num += sum(smoothed_predictions)

    interictal_l += (len(interictal_segments) - 1) * int(n_segment_samples / intericatl_ioverlap) + n_segment_samples

FPR = fP_num / (interictal_l / (sampling_f * 3600.0))

print 'LSTM FPR:', FPR


with open('../logs/interictal_'+patient_id + '.txt', 'a') as f:
        f.write(str(FPR)+'\n')
