import parameters as params
import sys
import random as rn
import numpy as np
np.random.seed(2 * int(sys.argv[2]))
rn.seed(2*int(sys.argv[2]))
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, Activation, Reshape, BatchNormalization, Dropout, LSTM, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
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

patient_id = sys.argv[1]

patient_train_mats = glob(join('../standardized_images/chb' + str(patient_id) + '/train/', '*.h5'))
patient_val_mats = glob(join('../standardized_images/chb' + str(patient_id) + '/validation/', '*.h5'))
patient_CNN_model_path = join('../cnn_models', 'chb' + patient_id) + '/model.h5'
patient_LSTM_model_path = join('../lstm_models', 'chb' + patient_id)


class_num = params.class_num
batch_s = params.batch_s
epochs = params.epochs
dropout_rate = params.dropout_rate

if not os.path.exists(patient_LSTM_model_path):
    os.makedirs(patient_LSTM_model_path)


X_val = []
y_val = []
print 'loading val set...'
for p in patient_val_mats:
    with h5py.File(p, 'r') as f:
        images = np.array(f.get('images'))
        lbs = np.array(f.get('labels')).ravel()
    X_val = np.vstack((X_val, images)) if len(X_val) else images
    y_val = np.hstack((y_val, lbs)) if len(y_val) else lbs

X_val = np.transpose(X_val, [0, 1, 3, 4, 2])
y_val = y_val.ravel()
y_val = one_hot(y_val, class_num)
print X_val.shape, y_val.shape

_, t, r, c, ch = X_val.shape
########################v########################## Network ###################################################################
cnn = load_model(patient_CNN_model_path)
model = Sequential()
model.add(TimeDistributed(cnn.get_layer('input'), input_shape=(t, r, c, ch)))
model.add(TimeDistributed(cnn.get_layer('c1')))
model.add(TimeDistributed(cnn.get_layer('bn1')))
model.add(Activation('relu'))
model.add(TimeDistributed(cnn.get_layer('do1')))
model.add(TimeDistributed(cnn.get_layer('mp1')))
model.add(TimeDistributed(cnn.get_layer('c2')))
model.add(TimeDistributed(cnn.get_layer('bn2')))
model.add(Activation('relu'))
model.add(TimeDistributed(cnn.get_layer('do2')))
model.add(TimeDistributed(cnn.get_layer('mp2')))
model.add(TimeDistributed(cnn.get_layer('c3')))
model.add(TimeDistributed(cnn.get_layer('bn3')))
model.add(Activation('relu'))
model.add(TimeDistributed(cnn.get_layer('do3')))
model.add(TimeDistributed(cnn.get_layer('mp3')))
model.add(TimeDistributed(cnn.get_layer('fl')))
model.add(TimeDistributed(cnn.get_layer('fc1')))
model.add(TimeDistributed(cnn.get_layer('fc2')))

model.add(LSTM(units=512))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

opt = Adam()
loss = 'categorical_crossentropy'
model.compile(optimizer= opt, loss= loss, metrics= ['accuracy', 'categorical_crossentropy'])

print model.summary()
########################v########################## Training ###################################################################


early_stopping = params.early_stopping
max_val = 0
max_val_acc = 0
stop_cnt = early_stopping

for e in range(epochs):
    for i, p in enumerate(patient_train_mats):
        # print 'epochs:', e+1, ' / file:',i
        with h5py.File(p, 'r') as f:
            data = np.array(f.get('images'))
            labels = np.array(f.get('labels')).ravel()

        data = np.transpose(data, [0, 1, 3, 4, 2])
        labels = one_hot(labels, class_num)

        model.fit(x= data, y= labels, batch_size= batch_s, epochs= 1, shuffle=False, verbose=0)

    val_loss, val_acc, _ = model.evaluate(X_val, y_val, verbose=0)
    print 'Epoch', e, 'Val LossL', val_loss, 'Val Acc:', val_acc
    if val_acc >= max_val:
        max_val = val_acc
        max_val_acc = val_acc
        stop_cnt = early_stopping
        model.save(join(patient_LSTM_model_path, 'model.h5'))
    elif stop_cnt == 0:
        break
    else:
        stop_cnt -= 1
model = load_model(join(patient_LSTM_model_path, 'model.h5'))
