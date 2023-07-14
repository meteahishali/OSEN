import tensorflow as tf
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

# FUNCTIONS

###############
# Data loading.
def loadData(param):

  data = {'x_train': None, 'x_val' : None, 'x_test': None,
          'y_train': None, 'y_val' : None, 'y_test': None}
  
  x_train = []
  x_val = []
  x_test = []
  y_train = []
  y_val = []
  y_test = []

  for i in range(0, 5):

    (x_train_tmp, y_train_tmp), (x_test_tmp, y_test_tmp) = tf.keras.datasets.mnist.load_data()
    x_train_tmp = np.array(x_train_tmp, dtype='float32')
    x_test_tmp = np.array(x_test_tmp, dtype='float32')
  

    x_train_tmp, x_val_tmp, y_train_tmp, y_val_tmp = train_test_split(
        x_train_tmp, y_train_tmp, test_size=1/6, random_state=i)
    print('Data loaded!')
    print('Train: ', x_train_tmp.shape)
    print('Validation: ', x_val_tmp.shape)
    print('Test: ', x_test_tmp.shape)

    # Segmentation masks
    y_train_tmp = np.array(x_train_tmp)
    y_val_tmp = np.array(x_val_tmp)
    y_test_tmp = np.array(x_test_tmp)

    y_train_tmp[y_train_tmp > 0] = 1.0
    y_val_tmp[y_val_tmp > 0] = 1.0
    y_test_tmp[y_test_tmp > 0] = 1.0

    y_train_tmp[y_train_tmp == 0] = 0.0
    y_val_tmp[y_val_tmp == 0] = 0.0
    y_test_tmp[y_test_tmp == 0] = 0.0

    x_train.append(x_train_tmp)
    x_val.append(x_val_tmp)
    x_test.append(x_test_tmp)
    y_train.append(y_train_tmp)
    y_val.append(y_val_tmp)
    y_test.append(y_test_tmp)

  data['x_train'] = x_train
  data['x_val'] = x_val
  data['x_test'] = x_test
  data['y_train'] = y_train
  data['y_val'] = y_val
  data['y_test'] = y_test

  return data
###############

###############
# Taking measurements.
def takeMeas(data, param):

  MeasurementRate = param['MeasurementRate']

  train_meas = []
  val_meas = []
  test_meas = []
  M_norms = []

  for i in range(0, len(data['x_train'])):

    sizeM = data['x_train'][i].shape[1]
    sizeN = data['x_train'][i].shape[2]
    K = sizeM * sizeN # Signal size.
  
    x_train_tmp = np.reshape(data['x_train'][i], [len(data['x_train'][i]), K])
    x_val_tmp = np.reshape(data['x_val'][i], [len(data['x_val'][i]), K])
    x_test_tmp = np.reshape(data['x_test'][i], [len(data['x_test'][i]), K])
  
    m = round(MeasurementRate * K) # Number of measurements.
    
    # The measurement matrix.
    M = np.random.randn(m, K)
    # Normalize the measurement matrix.
    temp = np.sqrt(np.sum(M ** 2, axis = 0))
    coeff = np.tile(temp,(m, 1))
    M_norm = np.divide(M, coeff)

    x_train_meas = np.matmul(M_norm, x_train_tmp.T)
    x_val_meas= np.matmul(M_norm, x_val_tmp.T)
    x_test_meas = np.matmul(M_norm, x_test_tmp.T)

    if param['noise'] == 'True':
      # Desired SNR in dB
      SNR_dB = 0
      snr = 10.0**(SNR_dB/10.0) # linear SNR.

      ###### x_train ######
      p1 = np.var(x_train_meas, axis = 0)
      n = p1/snr

      noise_signal = np.zeros(x_train_meas.shape, dtype = 'float32')
      for j in range(0, len(n)):
        noise_signal[:, j] = np.random.normal(0, np.sqrt(n[j]), len(x_train_meas))

      # Noise up the original signal
      x_train_meas_noisy = x_train_meas + noise_signal
      x_train_meas = np.array(x_train_meas_noisy)

      ###### x_val ######
      p1 = np.var(x_val_meas, axis = 0)
      n = p1/snr

      noise_signal = np.zeros(x_val_meas.shape, dtype = 'float32')
      for j in range(0, len(n)):
        noise_signal[:, j] = np.random.normal(0, np.sqrt(n[j]), len(x_val_meas))

      # Noise up the original signal
      x_val_meas_noisy = x_val_meas + noise_signal
      x_val_meas = np.array(x_val_meas_noisy)

      ###### x_test ######
      p1 = np.var(x_test_meas, axis = 0)
      n = p1/snr

      noise_signal = np.zeros(x_test_meas.shape, dtype = 'float32')
      for j in range(0, len(n)):
        noise_signal[:, j] = np.random.normal(0, np.sqrt(n[j]), len(x_test_meas))

      # Noise up the original signal
      x_test_meas_noisy = x_test_meas + noise_signal
      x_test_meas = np.array(x_test_meas_noisy)

    x_train_new = np.matmul(M_norm.T, x_train_meas).T
    x_val_new = np.matmul(M_norm.T, x_val_meas).T
    x_test_new = np.matmul(M_norm.T, x_test_meas).T

    ### Normalization of the proxies.
    scaler = Normalizer().fit(x_train_new)
    x_train_new = scaler.transform(x_train_new)
    x_train_new = np.reshape(x_train_new, [len(x_train_new), sizeM, sizeN])

    scaler = Normalizer().fit(x_val_new)
    x_val_new = scaler.transform(x_val_new)
    x_val_new = np.reshape(x_val_new, [len(x_val_new), sizeM, sizeN])

    scaler = Normalizer().fit(x_test_new)
    x_test_new = scaler.transform(x_test_new)
    x_test_new = np.reshape(x_test_new, [len(x_test_new), sizeM, sizeN])

    data['x_train'][i] = np.expand_dims(x_train_new, axis=3)
    data['x_val'][i] = np.expand_dims(x_val_new, axis=3)
    data['x_test'][i] = np.expand_dims(x_test_new, axis=3)
    data['y_train'][i] = np.expand_dims(data['y_train'][i], axis=3)
    data['y_val'][i] = np.expand_dims(data['y_val'][i], axis=3)
    data['y_test'][i] = np.expand_dims(data['y_test'][i], axis=3)

    ### Normalization of the measurements.
    scaler = Normalizer().fit(x_train_meas.T)
    x_train_meas = scaler.transform(x_train_meas.T)

    scaler = Normalizer().fit(x_val_meas.T)
    x_val_meas = scaler.transform(x_val_meas.T)

    scaler = Normalizer().fit(x_test_meas.T)
    x_test_meas = scaler.transform(x_test_meas.T)

    train_meas.append(x_train_meas)
    val_meas.append(x_val_meas)
    test_meas.append(x_test_meas)
    M_norms.append(M_norm)

    print('\nThe proxy signals are obtained for set ' + str(i + 1) + '.')
    print('Train: ', data['x_train'][i].shape)
    print('Validation: ', data['x_val'][i].shape)
    print('Test: ', data['x_test'][i].shape)

  data['x_train_meas'] = train_meas
  data['x_val_meas'] = val_meas
  data['x_test_meas'] = test_meas
  data['M_norm'] = M_norms

  return data
###############

###############
def loss_func(y_true, y_pred):
  loss = tf.nn.l2_loss(y_true - y_pred)
  return loss
###############

###############
def performThreshold(y_pred, y_gtd, thres):
  sizeM = y_gtd.shape[1]
  sizeN = y_gtd.shape[2]
  specificity = 0
  precision = 0
  recall = 0
  F1score = 0
  CE = 0
  TPR = 0
  FPR = 0

  for i in range(0, len(y_pred)):

    imge =np.array(y_pred[i, :, :, 0], dtype = 'float32')
    gtd = np.array(y_gtd[i, :, :, 0], dtype = 'float32')

    imge = np.interp(imge, (imge.min(), imge.max()), (0, +255))
    imge = np.array(imge, dtype = 'float32')
    
    imge_new = np.zeros(imge.shape, dtype = 'float32')
    imge_new[imge >= thres] = 1.0
    imge_new[imge < thres] = 0.0
    
    gtd[gtd == 255] = 1.0

    TP = imge_new[gtd == 1].sum()
    FN = gtd[imge_new == 0].sum()
    FP = imge_new.sum() - gtd[imge_new == 1].sum()
    TN = (imge_new == 0).sum() - gtd[imge_new == 0].sum()

    CE = CE + ((FP + FN) / (sizeM * sizeN))
    TPR = TPR + (TP / (TP + FN)) # Sensitivity
    FPR = FPR + (1 - (TN / (FP + TN)))# 1 - Specificity
    precision = precision + (TP / imge_new.sum())
    specificity = specificity + (TN / (FP + TN))
    recall = recall + (TP / (TP + FN))
  
  precision  = precision / y_gtd.shape[0]
  specificity  = specificity / y_gtd.shape[0]
  recall  = recall / y_gtd.shape[0]
  CE = CE / y_gtd.shape[0]
  TPR = TPR / y_gtd.shape[0]
  FPR = FPR / y_gtd.shape[0]
  F1score = 2*((precision * recall) / (precision + recall))
  F2score = (5*(precision * recall)) / ((4 * precision) + recall)

  return precision, specificity, recall, F1score, F2score, CE, TPR, FPR
###############

###############
def calculatePerformance(param, model, x_test, y_test, x_val, y_val):

  y_pred_val = model.predict(x_val, batch_size=32)

  # Do thresholding and save segmentation results
  thresholds = range(0, 256)
  specificity = np.zeros(len(thresholds))
  precision = np.zeros(len(thresholds))
  recall = np.zeros(len(thresholds))
  F1score = np.zeros(len(thresholds))
  F2score = np.zeros(len(thresholds))
  CE = np.zeros(len(thresholds))
  TPR = np.zeros(len(thresholds))
  FPR = np.zeros(len(thresholds))

  for j in range(0, len(thresholds)):
    print(j)
    precision[j], specificity[j], recall[j], F1score[j], F2score[j], CE[j], TPR[j], FPR[j] = performThreshold(y_pred_val, y_val, thresholds[j])

  thr = np.argmax(F1score)

  print('Obtained Validation F1 score: ', F1score[thr], ' using THR: ', thr)

  start = time.time()
  y_pred_test = model.predict(x_test, batch_size=32)
  end = time.time()
  print('Execution time: ', (end - start) / len(x_test))

  precision, specificity, recall, F1score, F2score, CE, TPR, FPR = performThreshold(y_pred_test, y_test, thr)

  print('Obtained Test F1 score: ', F1score, ' using THR: ', thr)
  print('Obtained Test F2 score: ', F2score, ' using THR: ', thr)
  print('Obtained Test Precision: ', precision, ' using THR: ', thr)
  print('Obtained Test Specificity: ', specificity, ' using THR: ', thr)
  print('Obtained Test Recall: ', recall, ' using THR: ', thr)
  print('Obtained Test CE: ', CE, ' using THR: ', thr)
  print('Obtained Test TPR: ', TPR, ' using THR: ', thr)
  print('Obtained Test FPR: ', FPR, ' using THR: ', thr)
  
  return precision, specificity, recall, F1score, F2score, CE, TPR, FPR
###############

# END FUNCTIONS