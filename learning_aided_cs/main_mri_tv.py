import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import tv

ap = argparse.ArgumentParser()
#ap.add_argument('--sample', default = np.arange(2047), help = "Sample from 0 to 2047.") # If you want to test all samples at once.
ap.add_argument('--sample', default = np.arange(2047), help = "Sample from 0 to 2047.")
ap.add_argument('--nu', default = 1, help = "Which run?")
ap.add_argument('--mr', default = 0.05, help="Measurement rate.")
ap.add_argument('--learn_aid', default = 'True', help = "Learning-aided CS.")
ap.add_argument('--model_aid', default = 'CSEN1', help = "If learning-aided CS.")
ap.add_argument('--q', default = 3, help = "If learning-aided CS, and operational layers used.")
ap.add_argument('--support_path', default = 'output/', help = "Estimated support probabilities by the networks.")

args = vars(ap.parse_args())
if isinstance(args['sample'], str):
    ind = np.array([int(args['sample'])]) # Samples from the test set.
else:
    ind = args['sample'] # Samples from the test set.
nu = int(args['nu']) # Which run?
MeasurementRate = float(args['mr']) # Sampling rate.
q = int(args['q'])
nu = int(args['nu'])

param = {}
param['l_aid'] = args['learn_aid']
param['model_aid'] = args['model_aid']

support_path = args['support_path'] + 'predictions_' + param['model_aid']

if 'OSEN' in param['model_aid']:
    support_path = support_path + '_q_' + str(q)  + '_cr_' + str(MeasurementRate) + '_nuR_' + str(nu) + '.npy'
else:
    support_path = support_path + '_cr_' + str(MeasurementRate) + '_nuR_' + str(nu) + '.npy'

param['lam'] = 0.1
param['rho'] = 1.0
param['alpha'] = 1.6

param['lam'] = 0.01
param['rho'] = 1.0
param['alpha'] = 0.7

np.random.seed(int(nu) * 10)

if param['l_aid'] == 'True':
    prob = np.load(support_path)
    w_g = np.abs(prob[:, :, :, 0]) + np.abs(prob[:, :, :, 1])
    w_g = w_g[ind, :, :]
    w_g = w_g/np.max(w_g)
    w_g = 1/(w_g + 0.2)
    w_g = np.reshape(w_g, (w_g.shape[0], w_g.shape[1] * w_g.shape[2]))
    
    if 'OSEN' in param['model_aid']:
        outDir = 'cs_output_' + param['model_aid'] + '_q_' + str(q) + '_aided_nu' + str(nu) + '/' + str(MeasurementRate) + '/'
    else:
        outDir = 'cs_output_' + param['model_aid'] + '_aided_nu' + str(nu) + '/' + str(MeasurementRate) + '/'
else:
    outDir = 'cs_output_nu' + str(nu) + '/' + str(MeasurementRate) + '/'
if not os.path.exists(outDir): os.makedirs(outDir)

x_selected = np.load('x_test.npy')[ind, :, :]

# Measurement mask.
mask = np.load('masks_' + str(MeasurementRate) + '_' + str(nu) + '.npy')
mask = np.squeeze(mask)

im_dims = mask.shape

print('Accerelation factor R: ', np.round(1/MeasurementRate, 2))

for i in range(len(ind)):

    image = np.squeeze(x_selected[i, :, :])
    I = np.fft.fftshift(np.fft.fft2(image*100))[mask > 0] # FFT domain measurements, vectorized.

    y = I

    # Zero-filling reconstruction.
    z_ = np.zeros(im_dims, y.dtype)
    z_[mask > 0] = y
    x_hat_zero = np.fft.ifft2(np.fft.fftshift(z_))

    if param['l_aid'] == 'True':
        param['w_g'] = np.squeeze(w_g[i, :])
    
    hist, x_hat, k = tv.tv_recon_2d(y, mask, param)
    print(k)
    x_hat = np.reshape(x_hat, mask.shape)

    # PSNR.
    recon_zero = np.real(x_hat_zero) / 100
    recon = np.real(x_hat) / 100

    print('Zero-fill reconstruction: ')
    tv.PSNR(np.expand_dims(image, axis = 0), np.expand_dims(recon_zero, axis = 0))
    tv.NMSE(np.expand_dims(image, axis = 0), np.expand_dims(recon_zero, axis = 0))
    print('TV reconstruction: ')
    tv.PSNR(np.expand_dims(image, axis = 0), np.expand_dims(recon, axis = 0))
    tv.NMSE(np.expand_dims(image, axis = 0), np.expand_dims(recon, axis = 0))

    np.save(outDir + str(ind[i]) + '.npy',
            {'zero_recon': recon_zero,
             'tv_recon': recon})

    # Plot and imshows.
    plt.semilogy(hist['r_norm'], label = 'r_norm')
    plt.semilogy(hist['eps_prim'], '--', label = 'eps_prim')
    plt.semilogy(hist['s_norm'], label = 's_norm')
    plt.semilogy(hist['eps_dual'], '--', label = 'eps_dual')
    plt.semilogy(hist['objval'], label = 'objval')
    plt.legend()
    plt.savefig(outDir + str(ind[i]) + '_losses.png')

    fig, axs = plt.subplots(2, 2, gridspec_kw={'wspace':0.1, 'hspace':0.2}, figsize=(8,8))
    axs[0,0].imshow(image, cmap = 'gray')
    axs[0,0].axis('off')
    axs[0,0].set_title('Original Image')
    axs[0,1].imshow(recon_zero, cmap = 'gray')
    axs[0,1].axis('off')
    axs[0,1].set_title('Zero-Filling Recon')
    axs[1,0].imshow(recon, cmap = 'gray')
    axs[1,0].axis('off')
    axs[1,0].set_title('TV Recon')
    axs[1,1].imshow(np.abs(image - recon))
    axs[1,1].axis('off')
    axs[1,1].set_title('TV Recon Error')
    fig.savefig(outDir + str(ind[i]) + '_TV_recon_2d.png', bbox_inches='tight')

    fig, axs = plt.subplots(1, 3, gridspec_kw={'wspace':0.1, 'hspace':0.2}, figsize=(16, 48))
    axs[0].imshow(np.abs(np.fft.fftshift(np.fft.fft2(image))) ** 0.1, cmap = 'gray')
    axs[0].axis('off')
    axs[0].set_title("Original k-Space")
    axs[1].imshow(mask > 0, interpolation='none', cmap = 'gray')
    axs[1].set_title("Random Sample Locations")
    axs[1].axis('off')
    axs[2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(recon_zero))) ** 0.1, cmap = 'gray')
    axs[2].axis('off')
    axs[2].set_title("Undersampled k-Space")
    plt.savefig(outDir + str(ind[i]) + '_random_sampling_2d.png', bbox_inches='tight')