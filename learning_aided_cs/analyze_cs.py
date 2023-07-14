import numpy as np
import matplotlib.pyplot as plt
import math

def PSNR(x, x_predicted):
  psnrs = []
  PIXEL_MAX = 1.0

  for i in range(0, len(x_predicted)):
    recon = x_predicted[i, :, :]
    image = x[i, :, :]
    mse = np.mean((image - recon) ** 2)    
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    psnrs.append(psnr)
  
  print('Number of images: ', len(psnrs))
  print('Average PSNR: ', np.mean(psnrs))
  return np.mean(psnrs)

def NMSE(x, x_predicted):
  nmses = []

  for i in range(0, len(x_predicted)):
    recon = x_predicted[i, :, :]
    image = x[i, :, :]
    mse = np.sum((image - recon) ** 2)    
    nmse = mse / np.sum(image ** 2)
    nmses.append(nmse)
  
  print('Number of images: ', len(nmses))
  print('Average NMSE: ', np.mean(nmses))
  return np.mean(nmses)

path = 'cs_output_nu'
mr = '0.05/'

x_test = np.load('x_test.npy')

ps_zero = []
ns_zero = []
ps = []
ns = []
for i in range(0, 5):
  filepath = path + str(i + 1) + '/' + mr
  zero_preds = np.zeros(x_test.shape)
  tv_preds = np.zeros(x_test.shape)

  for j in range(0, len(x_test)):
    pred = np.load(filepath + str(j) + '.npy', allow_pickle=True)

    zero_preds[j, :, :] = pred.item()['zero_recon']
    tv_preds[j, :, :] = pred.item()['tv_recon']

  print('Zero-fill reconstruction:')
  ps_zero.append(PSNR(x_test, zero_preds))
  ns_zero.append(NMSE(x_test, zero_preds))
  print('TV reconstruction:')
  ps.append(PSNR(x_test, tv_preds))
  ns.append(NMSE(x_test, tv_preds))

print('\n\nAll zero-fill reconstruction:')
print(ps_zero)
print(ns_zero)
print('\n\nAll TV reconstruction:')
print(ps)
print(ns)

print('\n\nAll zero-fill reconstruction:')
print(np.mean(ps_zero))
print(np.mean(ns_zero))
print(np.std(ps))
print(np.std(ns))
print('\n\nAll TV reconstruction:')
print(np.mean(ps))
print(np.mean(ns))
print(np.std(ps))
print(np.std(ns))