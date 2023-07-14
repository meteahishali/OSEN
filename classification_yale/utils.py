import tensorflow as tf
import numpy as np
import time
import scipy.io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, Normalizer

# FUNCTIONS

###############
# Data loading.
def loadData(param):

  data = {'x_train': None, 'x_val' : None, 'x_test': None,
          'y_train': None, 'y_val' : None, 'y_test': None, 
          'x_train_meas': None, 'x_val_meas': None, 'x_test_meas': None,
          'm_train': None, 'm_val' : None, 'm_test': None, 'M_norms': None}
          
  x_train = []
  x_train_meas = []
  y_train = []
  m_train = []

  x_test = []
  x_test_meas = []
  y_test = []
  m_test = []

  x_val = []
  x_val_meas = []
  y_val = []
  m_val = []

  M_norms = []

  for i in range(0, 5):
    dataPath = 'OSENData/data_dic_' + str(param['MeasurementRate']) + '_' + str(i + 1) + '.mat' # Modify
    Data = scipy.io.loadmat(dataPath)

    # Proxies.
    x_dic_tmp = Data['x_dic'].astype('float32')
    x_train_tmp = Data['x_train'].astype('float32')
    x_test_tmp = Data['x_test'].astype('float32')

    # Measurements.
    x_dic_meas_tmp = Data['Y0'].astype('float32').T
    x_train_meas_tmp = Data['Y1'].astype('float32').T
    x_test_meas_tmp = Data['Y2'].astype('float32').T

    # Labels.
    M_norms_tmp = Data['Proj_M'].astype('float32').T
    y_dic_tmp = Data['l_dic'].astype('float32')
    y_train_tmp = Data['l_train'].astype('float32')
    y_test_tmp = Data['l_test'].astype('float32')

    # Masks.
    m_dic_tmp = Data['y_dic'].astype('float32')
    m_train_tmp = Data['y_train'].astype('float32')
    m_test_tmp = Data['y_test'].astype('float32')

    onehot_encoder = OneHotEncoder(sparse=False)
    y_dic_tmp = onehot_encoder.fit_transform(y_dic_tmp)
    y_train_tmp = onehot_encoder.fit_transform(y_train_tmp)
    y_test_tmp = onehot_encoder.fit_transform(y_test_tmp)

    print('\n\n\n')
    print('Loaded dataset:')
    print(len(x_train_tmp), ' train')
    print(len(x_test_tmp), ' test')
    
    # Partition for the validation.
    x_train_tmp, x_val_tmp, x_train_meas_tmp, x_val_meas_tmp, y_train_tmp, y_val_tmp, m_train_tmp, m_val_tmp = train_test_split(x_train_tmp, 
                                                                                                                                x_train_meas_tmp,
                                                                                                                                y_train_tmp,
                                                                                                                                m_train_tmp,
                                                                                                                                test_size = 0.2,
                                                                                                                                random_state = 1)

    x_train_tmp = np.concatenate((x_dic_tmp, x_train_tmp), axis = 0)
    x_train_meas_tmp = np.concatenate((x_dic_meas_tmp, x_train_meas_tmp), axis = 0)
    y_train_tmp = np.concatenate((y_dic_tmp, y_train_tmp), axis = 0)
    m_train_tmp = np.concatenate((m_dic_tmp, m_train_tmp), axis = 0)
    
    # Data normalization for proxies.
    m =  x_train_tmp.shape[1]
    n =  x_train_tmp.shape[2]

    x_train_tmp = np.reshape(x_train_tmp, [len(x_train_tmp), m * n])
    x_val_tmp = np.reshape(x_val_tmp, [len(x_val_tmp), m * n])
    x_test_tmp = np.reshape(x_test_tmp, [len(x_test_tmp), m * n])
    
    scaler = Normalizer().fit(x_train_tmp)
    x_train_tmp = scaler.transform(x_train_tmp)
    scaler = Normalizer().fit(x_val_tmp)
    x_val_tmp = scaler.transform(x_val_tmp)
    scaler = Normalizer().fit(x_test_tmp)
    x_test_tmp = scaler.transform(x_test_tmp)

    x_train_tmp = np.reshape(x_train_tmp, [len(x_train_tmp), m, n])
    x_val_tmp = np.reshape(x_val_tmp, [len(x_val_tmp), m, n])
    x_test_tmp = np.reshape(x_test_tmp, [len(x_test_tmp), m, n])

    y_train.append(y_train_tmp)
    y_val.append(y_val_tmp)
    y_test.append(y_test_tmp)

    x_train.append(np.expand_dims(x_train_tmp, axis=-1))
    x_val.append(np.expand_dims(x_val_tmp, axis=-1))
    x_test.append(np.expand_dims(x_test_tmp, axis=-1))
    
    print("\n")
    print('Partitioned.')
    print(len(x_train_tmp), ' Train')
    print(len(x_val_tmp), ' Validation')
    print(len(x_test_tmp), ' Test\n')

    # Data normalization for measurements.

    scaler = Normalizer().fit(x_train_meas_tmp)
    x_train_meas_tmp = scaler.transform(x_train_meas_tmp)
    scaler = Normalizer().fit(x_val_meas_tmp)
    x_val_meas_tmp = scaler.transform(x_val_meas_tmp)
    scaler = Normalizer().fit(x_test_meas_tmp)
    x_test_meas_tmp = scaler.transform(x_test_meas_tmp)
    
    x_train_meas.append(x_train_meas_tmp)
    x_val_meas.append(x_val_meas_tmp)
    x_test_meas.append(x_test_meas_tmp)
    M_norms.append(M_norms_tmp)
    m_train.append(m_train_tmp)
    m_test.append(m_test_tmp)
    m_val.append(m_val_tmp)

  data['x_train'] = x_train
  data['x_val'] = x_val
  data['x_test'] = x_test

  data['x_train_meas'] = x_train_meas
  data['x_val_meas'] = x_val_meas
  data['x_test_meas'] = x_test_meas

  data['y_train'] = y_train
  data['y_val'] = y_val
  data['y_test'] = y_test

  data['m_train'] = m_train
  data['m_test'] = m_test
  data['m_val'] = m_val

  data['M_norm'] = M_norms



  data['x_train'] = x_train
  data['x_val'] = x_val
  data['x_test'] = x_test
  data['y_train'] = y_train
  data['y_val'] = y_val
  data['y_test'] = y_test

  return data
###############

###############
def calculatePerformance(param, model, x_test, y_test, x_val, y_val):

  start = time.time()
  y_pred = model.predict(x_test, batch_size = 32)
  if len(y_pred) == 2: # Only predictions.
    y_pred = y_pred[0]
  end = time.time()
  print('Execution time: ', (end - start) / len(x_test))
  y_pred = np.argmax(y_pred, axis = 1)
  y_test = np.argmax(y_test, axis = 1)
  accuracy = np.sum(y_pred == y_test) / len(y_test)
  print('Accuracy: ', accuracy)
  
  return accuracy

###############
# END FUNCTIONS