import parameters as params
import sys
import random as rn
import numpy as np
np.random.seed(int(sys.argv[2]))
rn.seed(int(sys.argv[2]))
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
import h5py
from natsort import natsorted
from glob import glob
from os.path import join
import tensorflow as tf
import os
import keras.backend as K
from keras.models import load_model

# os.environ['PYTHONHASHSEED'] = '0'
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# from keras import backend as K
# tf.set_random_seed(int(sys.argv[2]))
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



patient_id = sys.argv[1]

patient_train_mats = glob(join('../standardized_images/chb' + str(patient_id) + '/train/', '*.h5'))
patient_val_mats = glob(join('../standardized_images/chb' + str(patient_id) + '/validation/', '*.h5'))

patient_model_path = join('../cnn_models', 'chb' + patient_id)
if not os.path.exists(patient_model_path):
    os.makedirs(patient_model_path)

class_num = params.class_num
batch_s = params.batch_s
epochs = params.epochs
dropout_rate = params.dropout_rate

X_val = []
y_val = []
print 'loading val set...'
for p in patient_val_mats:
    with h5py.File(p, 'r') as f:
        images = np.array(f.get('images'))
        lbs = np.array(f.get('labels')).ravel()

    _, s, ch, r, c = images.shape
    images = images.reshape((-1, ch, r, c))
    X_val = np.vstack((X_val, images)) if len(X_val) else images
    y_val = np.hstack((y_val, lbs)) if len(y_val) else lbs

X_val = np.transpose(X_val, [0, 2, 3, 1])
y_val = seq_to_seg_labels(y_val, s)
y_val = one_hot(y_val, class_num)
print X_val.shape, y_val.shape

################################################## Network ###################################################################

x = Input(shape=(r, c, ch), name='input')
c1 = Conv2D(filters= 16, kernel_size= (3, 3), strides= (2, 2), padding= 'same', name='c1')(x)
c1_n = BatchNormalization(name='bn1')(c1)
c1 = Activation('relu')(c1_n)
do1 = Dropout(rate = dropout_rate, name='do1')(c1)
mp1 = MaxPool2D((2,2), padding= 'same', name='mp1')(do1)
c2 = Conv2D(filters= 32, kernel_size= (3, 3), strides= (1, 1), padding= 'same', name='c2')(mp1)
c2_n = BatchNormalization(name='bn2')(c2)
c2 = Activation('relu')(c2_n)
do2 = Dropout(rate = dropout_rate, name='do2')(c2)
mp2 = MaxPool2D((2,2), padding= 'same', name='mp2')(do2)
c3 = Conv2D(filters= 64, kernel_size= (3, 3), strides= (1, 1), padding= 'same', name='c3')(mp2)
c3_n = BatchNormalization(name='bn3')(c3)
c3 = Activation('relu')(c3_n)
do3 = Dropout(rate = dropout_rate, name='do3')(c3)
mp3 = MaxPool2D((2,2), padding= 'same', name='mp3')(do3)
fl = Flatten(name='fl')(mp3)
fc1 = Dense(units= 512, activation= 'relu', name='fc1')(fl)
fc2 = Dense(units= 256, activation= 'relu', name='fc2')(fc1)
out = Dense(units= class_num, activation= 'softmax')(fc2)
model = Model(inputs= x, outputs= out)

opt = Adam()
loss = 'categorical_crossentropy'
model.compile(optimizer= opt, loss= loss, metrics= ['accuracy', 'categorical_crossentropy'])

print model.summary()
########################v########################## Training ###################################################################


early_stopping = params.early_stopping
max_val = 0
max_val_acc = 0
stop_cnt = early_stopping
print 'training...'
for e in range(epochs):
    for i, p in enumerate(patient_train_mats):
        with h5py.File(p, 'r') as f:
            data = np.array(f.get('images')).reshape((-1, ch, r, c))
            labels = np.array(f.get('labels')).ravel()

        labels = seq_to_seg_labels(labels, s)
        data = np.transpose(data, [0, 2, 3, 1])
        labels = one_hot(labels, class_num)

        model.fit(x=data, y=labels, batch_size=batch_s, epochs=1, shuffle=False, verbose=0)

    val_loss, val_acc, _ = model.evaluate(X_val, y_val, verbose=0)
    print 'Epoch', e, 'Val LossL', val_loss, 'Val Acc:', val_acc
    if val_acc >= max_val:
        max_val = val_acc
        max_val_acc = val_acc
        stop_cnt = early_stopping
        model.save(join(patient_model_path, 'model.h5'))
    elif stop_cnt == 0:
        break
    else:
        stop_cnt -= 1

model = load_model(join(patient_model_path, 'model.h5'))
