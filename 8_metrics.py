import sys
import numpy as np
from glob import glob
from os.path import join




patient_id = sys.argv[1]

with open('../logs/seizure_'+patient_id + '.txt', 'r') as f:
    lines = map(float, f.readlines())
    predicted_seizures = lines[::2]
    prediction_time = np.array(lines[1::2])
    prediction_time = prediction_time[prediction_time != 0]
    sensitivity = np.mean(predicted_seizures)
    mean_prediction_time = np.mean(prediction_time)

with open('../logs/interictal_'+patient_id + '.txt', 'r') as f:
    lines = map(float, f.readlines())
    mean_FPR = np.mean(lines)


with open('../logs/metrics.txt', 'a') as f:
    f.write('\nPatient {}'.format(patient_id) + ' Sen: '+ "{:.4f}".format(sensitivity) +' / FPR: '+"{:.4f}".format(mean_FPR)+ ' / Time: '+ "{:.4f}".format(mean_prediction_time)+'\n\n')
