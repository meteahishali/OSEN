import numpy as np
import math
import scipy.sparse.linalg as la

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

def computeGradients(xx):
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

'''
function sol=prox_l1_w(x,gamma,c)
    % c is cost
    sol= max(0, x - c.*gamma) - max(0, -x - c.*gamma);
end
'''

def shrinkage(a, kappa):
    return np.clip(a - (kappa), a_min=0, a_max=None) - np.clip(-a - (kappa), a_min=0, a_max=None)

def shrinkage_weighted(a, kappa, w_g):
    return np.clip(a - (w_g * kappa), a_min=0, a_max=None) - np.clip(-a - (w_g * kappa), a_min=0, a_max=None)

def tv_recon_2d(b, mask, param):

    """ Solve total variation minimization via ADMM *without forming the difference matrices*

    Solves the following problem via ADMM:

       min  (1/2)||Ax - b||_2^2 + lambda * sum_i |x_{i+1} - x_i|
       
       where b is the measured signal (undersampled FT samples)
       x is the reconstructed signal
       A is both FT and undersampling (BU - see Afonso 2009 section III-C-4)
       
    """

    lam = param['lam']
    rho = param['rho']
    alpha = param['alpha']
    
    MAX_ITER = 2000
    ABSTOL = 1e-4
    RELTOL = 1e-2

    n = b.size
    im_dims = ny, nx = mask.shape
    N = ny*nx
    
    # if mask.dtype != bool:
    #     mask = mask > 0

    # sampling and reconstruction operators
    # FFT and undersample
    def U(v):
        v = np.reshape(v, im_dims)
        Uv = np.fft.fftshift(np.fft.fft2(v))[mask > 0]
        return np.reshape(Uv, (n,))

    # zero-fill and IFFT
    def U_H(v):
        v_ = np.zeros(im_dims, v.dtype)
        v_[mask > 0] = v
        U_Hv = np.fft.ifft2(np.fft.fftshift(v_))
        return np.reshape(U_Hv, (N,))

    # difference matrix operator - 2D TV
    def D(v):
        v = np.reshape(v, im_dims)
        Dv = 2*v - np.roll(v, -1, 0) - np.roll(v, -1, 1)
        return np.reshape(Dv, (N,))
    
    def D_H(v):
        v = np.reshape(v, im_dims)
        D_Hv = 2*v - np.roll(v, 1, 0) - np.roll(v, 1, 1)
        return np.reshape(D_Hv, (N,))

    def F_matvec(v):
        return U_H(U(v)) + rho * D_H(D(v))

    F = la.LinearOperator((N, N), matvec=F_matvec, rmatvec=F_matvec)
    
    def tv_recon_objective(b, lam, x, z):
        return 0.5 * np.linalg.norm(U(x) - b)**2 + lam * np.linalg.norm(z)

    x = np.zeros((N,))
    z = x.copy()
    u = x.copy()

    history = {'objval' : [None]*MAX_ITER,
               'r_norm': [None]*MAX_ITER,
               's_norm': [None]*MAX_ITER,
               'eps_prim': [None]*MAX_ITER,
               'eps_dual': [None]*MAX_ITER}

    for k in range(MAX_ITER):

        # x-update (minimization)
        # iterative version
        x, _ = la.cg(F, U_H(b) + rho * D_H(z - u), maxiter=k//100, x0=x)

        # z-update (minimization) with relaxation
        # uses soft thresholding - the proximity operator of the l-1 norm
        z_ = z
        Dx_hat = alpha * D(x) + (1 - alpha) * z_
        
        if param['l_aid'] == 'False': # Learning-aided CS or not.
            z = shrinkage(Dx_hat + u, lam / rho)
        else:
            z = shrinkage_weighted(Dx_hat + u, lam / rho, param['w_g'])
        

        # y-update (dual update)
        u = u + Dx_hat - z

        # keep track of progress
        objval = tv_recon_objective(b, lam, x, z)

        r_norm = np.linalg.norm(D(x) - z)
        s_norm = np.linalg.norm(rho * D_H(z_ - z))

        eps_prim = np.sqrt(N) * ABSTOL + RELTOL * max(np.linalg.norm(D(x)),
                                                      np.linalg.norm(-z))
        eps_dual = np.sqrt(N) * ABSTOL + RELTOL * np.linalg.norm(rho * D_H(u))

        history['objval'][k] = objval
        history['r_norm'][k] = r_norm
        history['s_norm'][k] = s_norm
        history['eps_prim'][k] = eps_prim
        history['eps_dual'][k] = eps_dual

        if r_norm < eps_prim and s_norm < eps_dual:
            break

    return history, x, k