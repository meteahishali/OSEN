import tensorflow as tf
import numpy as np
import time
import sys
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, Normalizer

# FUNCTIONS
###############
def PSNR(x, x_predicted):
  psnrs = []
  PIXEL_MAX = 1.0

  for i in range(0, len(x_predicted)):
    image = x_predicted[i, :, :]
    recon = x[i, :, :]
    mse = np.mean((image - recon) ** 2)    
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    psnrs.append(psnr)
  
  print('Number of images: ', len(psnrs))
  print('Average PSNR: ', np.mean(psnrs))
###############

###############
# Data loading.
def loadData(param):

  data = {'x_train': None, 'x_val' : None, 'x_test': None,
          'x_train_meas': None, 'x_val_meas': None, 'x_test_meas': None,
          'm_train': None, 'm_val' : None, 'm_test': None, 'M_norms': None}

  x_train_tmp = np.load('x_train.npy')
  x_test_tmp = np.load('x_test.npy')

  x_train_tmp, x_val_tmp = train_test_split(x_train_tmp, test_size = 1/6, random_state = param['nu'] - 1)
  print('\n\nData loaded!')
  print('Train: ', x_train_tmp.shape)
  print('Validation: ', x_val_tmp.shape)
  print('Test: ', x_test_tmp.shape)

  data['x_train'] = x_train_tmp
  data['x_val'] = x_val_tmp
  data['x_test'] = x_test_tmp

  return data
###############

###############
# Taking measurements.
def takeMeas(data, param):

  def computeGradients(xx):
    # Segmentation masks, gradients.
    d_u = np.roll(xx, -1, 1)
    d_v = np.roll(xx, -1, 2)
    x_du = np.abs(xx - d_u)
    x_dv = np.abs(xx - d_v)

    return x_du, x_dv
  
  def computeSupports(xx):
    # Segmentation masks, gradients.
    d_u = np.roll(xx, -1, 1)
    d_v = np.roll(xx, -1, 2)
    y_du = np.abs(xx - d_u)
    y_dv = np.abs(xx - d_v)

    y_du[y_du > 0.04] = 1.0
    y_du[y_du <= 0.04] = 0.0
    y_dv[y_dv > 0.04] = 1.0
    y_dv[y_dv <= 0.04] = 0.0

    return y_du, y_dv

  def zero_fill_recon(xx, mask):
    # Zero-filling reconstruction.
    sizeM = xx.shape[1]
    sizeN = xx.shape[2]
    x_recon = np.zeros(xx.shape, dtype =xx.dtype)

    for i in range(0, len(xx)):
      image = xx[i, :, :]
      I = np.fft.fftshift(np.fft.fft2(image))[mask == 1] # FFT domain measurements, vectorized.
      y = I
      z_ = np.zeros((sizeM, sizeN), y.dtype)
      z_[mask == 1] = y
      x_hat_zero = np.fft.ifft2(np.fft.fftshift(z_))
      
      x_recon[i, :, :] = np.real(x_hat_zero)

      '''
      import matplotlib.pyplot as plt
      plt.figure(figsize=(24, 8))
      plt.subplot(1, 3, 1)
      plt.imshow(image, cmap='gray')
      plt.title('Original Image')
      plt.subplot(1, 3, 2)
      plt.imshow(mask == 1, cmap='gray')
      plt.title('Measurement Mask')
      plt.subplot(1, 3, 3)
      plt.imshow(np.real(x_hat_zero), cmap='gray')
      plt.title('Zero-filling Reconstruction')
      plt.savefig('proxy.png')
      #plt.show()
      '''
    return x_recon

  MeasurementRate = param['MeasurementRate']

  sizeM = data['x_train'].shape[1]
  sizeN = data['x_train'].shape[2]
  K = sizeM * sizeN # Signal size.

  m = round(MeasurementRate * K) # Number of measurements.

  print('\nAccerelation factor R: ', np.round(1/MeasurementRate, 2))

  mask = np.zeros((sizeM, sizeN), dtype=np.int16)

  # Gaussian random sampling.
  n = 0 # Measurement counter.
  vardens_factor = 3
  while n < m:
    y, x = np.array((sizeM, sizeN)) // 2
    x += int(np.random.randn(1) * sizeN/vardens_factor)
    y += int(np.random.randn(1) * sizeM/vardens_factor)
    
    if any((x < 0, y < 0, x >= sizeN, y >= sizeM)):
      continue
    
    if mask[x, y] == 0:
      mask[x, y] = 1
      n += 1

  # Focusing on center samples.
  area = MeasurementRate/3
  radius = np.sqrt(area/math.pi)
  
  Y, X = np.meshgrid(np.linspace(-0.5, 0.5, sizeM), np.linspace(-0.5, 0.5, sizeN))
  center_samples = X**2 + Y**2 < (radius)**2

  mask += center_samples

  # Need to remove additional samples from outside circle to match the sampling rate.
  mask_outside = np.array(mask)
  mask_outside[center_samples] = 2
  rows, cols = np.where(mask_outside == 1) # Outside measurements.
  nu_remSamples = np.sum(mask != 0) - m
  indices = np.random.randint(low = 0, high = np.sum(mask_outside == 1), size = nu_remSamples)
  for o in range(0, nu_remSamples):
    mask[rows[indices[o]], cols[indices[o]]] = 0

  MeasurementRate_new = np.sum(mask != 0) / (sizeM * sizeN) # New MR after focusing on center pixels.

  if np.round(MeasurementRate, 2) != np.round(MeasurementRate_new, 2):
    # Need to remove additional samples from outside circle to match the sampling rate.
    mask_outside = np.array(mask)
    mask_outside[center_samples] = 2
    indices = np.random.randint(low = 0, high = np.sum(mask_outside == 1), size = 100)
    for o in range(0, 100):
      mask[rows[indices[o]], cols[indices[o]]] = 0

    MeasurementRate_new = np.sum(mask != 0) / (sizeM * sizeN) # New MR after focusing on center pixels.
       
    if np.round(MeasurementRate, 2) != np.round(MeasurementRate_new, 2):
      sys.exit('Measurement rate is not the same after creating the mask.')

  mask[mask > 0] = 1

  x_train_tmp = zero_fill_recon(data['x_train'], mask)
  x_val_tmp = zero_fill_recon(data['x_val'], mask)
  x_test_tmp = zero_fill_recon(data['x_test'], mask)

  PSNR(data['x_test'], x_test_tmp)
  
  # Ground-truth segmentation masks, gradients.
  y_train_du_tmp, y_train_dv_tmp = computeSupports(data['x_train'])
  y_val_du_tmp, y_val_dv_tmp = computeSupports(data['x_val'])
  y_test_du_tmp, y_test_dv_tmp = computeSupports(data['x_test'])

  # Input proxies.
  x_train_du_tmp, x_train_dv_tmp = computeGradients(x_train_tmp)
  x_val_du_tmp, x_val_dv_tmp = computeGradients(x_val_tmp)
  x_test_du_tmp, x_test_dv_tmp = computeGradients(x_test_tmp)

  '''
  image = data['x_train'][i][1000, :, :]
  image2 = x_train_du_tmp[1000, :, :]
  image3 = x_train_dv_tmp[1000, :, :]

  image4 = data['x_train'][i][4850, :, :]
  image5 = x_train_du_tmp[4850, :, :]
  image6 = x_train_dv_tmp[4850, :, :]
  
  import matplotlib.pyplot as plt
  plt.figure(figsize=(24, 8))
  plt.subplot(2, 3, 1)
  plt.imshow(image, cmap='gray')
  plt.title('Original Image')
  plt.subplot(2, 3, 2)
  plt.imshow(image2, cmap='gray')
  plt.title('du')
  plt.subplot(2, 3, 3)
  plt.imshow(image3, cmap='gray')
  plt.title('dv')

  plt.subplot(2, 3, 4)
  plt.imshow(image4, cmap='gray')
  plt.title('Original Image')
  plt.subplot(2, 3, 5)
  plt.imshow(image5, cmap='gray')
  plt.title('du')
  plt.subplot(2, 3, 6)
  plt.imshow(image6, cmap='gray')
  plt.title('dv')
  plt.show()
  plt.savefig('proxy_0.04.png')
  '''
  
  # Normalizations:
  x_train_du_tmp = np.reshape(x_train_du_tmp, [len(x_train_du_tmp), sizeM * sizeN])
  x_train_dv_tmp = np.reshape(x_train_dv_tmp, [len(x_train_dv_tmp), sizeM * sizeN])
  x_val_du_tmp = np.reshape(x_val_du_tmp, [len(x_val_du_tmp), sizeM * sizeN])
  x_val_dv_tmp = np.reshape(x_val_dv_tmp, [len(x_val_dv_tmp), sizeM * sizeN])
  x_test_du_tmp = np.reshape(x_test_du_tmp, [len(x_test_du_tmp), sizeM * sizeN])
  x_test_dv_tmp = np.reshape(x_test_dv_tmp, [len(x_test_dv_tmp), sizeM * sizeN])

  # Normalization of the proxies.
  scaler = StandardScaler().fit(x_train_du_tmp)
  x_train_du_tmp = scaler.transform(x_train_du_tmp)
  x_train_du_tmp = np.reshape(x_train_du_tmp, [len(x_train_du_tmp), sizeM, sizeN])
  x_val_du_tmp = scaler.transform(x_val_du_tmp)
  x_val_du_tmp = np.reshape(x_val_du_tmp, [len(x_val_du_tmp), sizeM, sizeN])
  x_test_du_tmp = scaler.transform(x_test_du_tmp)
  x_test_du_tmp = np.reshape(x_test_du_tmp, [len(x_test_du_tmp), sizeM, sizeN])

  scaler = StandardScaler().fit(x_train_dv_tmp)
  x_train_dv_tmp = scaler.transform(x_train_dv_tmp)
  x_train_dv_tmp = np.reshape(x_train_dv_tmp, [len(x_train_dv_tmp), sizeM, sizeN])
  x_val_dv_tmp = scaler.transform(x_val_dv_tmp)
  x_val_dv_tmp = np.reshape(x_val_dv_tmp, [len(x_val_dv_tmp), sizeM, sizeN])
  x_test_dv_tmp = scaler.transform(x_test_dv_tmp)
  x_test_dv_tmp = np.reshape(x_test_dv_tmp, [len(x_test_dv_tmp), sizeM, sizeN])

  # Concatenating du and dv to obtain 2 channel data:
  data['y_train_d'] = np.concatenate((np.expand_dims(y_train_du_tmp, axis = 3),
                                      np.expand_dims(y_train_dv_tmp, axis=3)), axis = -1)

  data['y_val_d'] = np.concatenate((np.expand_dims(y_val_du_tmp, axis = 3),
                                    np.expand_dims(y_val_dv_tmp, axis=3)), axis = -1)

  data['y_test_d'] = np.concatenate((np.expand_dims(y_test_du_tmp, axis = 3),
                                    np.expand_dims(y_test_dv_tmp, axis=3)), axis = -1)

  data['x_train_d'] = np.concatenate((np.expand_dims(x_train_du_tmp, axis = 3),
                                      np.expand_dims(x_train_dv_tmp, axis=3)), axis = -1)

  data['x_val_d'] = np.concatenate((np.expand_dims(x_val_du_tmp, axis = 3),
                                    np.expand_dims(x_val_dv_tmp, axis=3)), axis = -1)

  data['x_test_d'] = np.concatenate((np.expand_dims(x_test_du_tmp, axis = 3),
                                    np.expand_dims(x_test_dv_tmp, axis=3)), axis = -1)

  data['M_masks'] = mask # Measurement mask.

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

  y_pred_thresh = np.zeros(y_pred.shape, dtype = 'float32')
  y_tmp = np.interp(y_pred, (y_pred.min(), y_pred.max()), (0, +255))
 
  y_pred_thresh[y_tmp >= thres] = 1.0
  y_pred_thresh[y_tmp < thres] = 0.0
  y_gtd[y_gtd == 255] = 1.0

  TP = y_pred_thresh[y_gtd == 1].sum()
  FN = y_gtd[y_pred_thresh == 0].sum()
  FP = y_pred_thresh.sum() - y_gtd[y_pred_thresh == 1].sum()
  TN = (y_pred_thresh == 0).sum() - y_gtd[y_pred_thresh == 0].sum()

  CE = ((FP + FN) / (sizeM * sizeN * y_gtd.shape[0] * y_gtd.shape[-1]))
  TPR = TP / (TP + FN)# Sensitivity
  FPR = 1 - (TN / (FP + TN)) # 1 - Specificity
  precision = TP / y_pred_thresh.sum()
  specificity = TN / (FP + TN)
  recall = TP / (TP + FN)

  F1score = 2*((precision * recall) / (precision + recall))
  F2score = (5*(precision * recall)) / ((4 * precision) + recall)
  

  return precision, specificity, recall, F1score, F2score, CE, TPR, FPR
###############

###############
def calculatePerformance(model, x_test, y_test, x_val, y_val):

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