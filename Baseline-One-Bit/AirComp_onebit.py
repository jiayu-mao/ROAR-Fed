# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
import torch

def transmission (libopt,A,p,theta,H_RB,h_UR,h_d,w):
    signs = [np.sign(tensor.cpu().numpy()) for tensor in w]
    signs_array = np.stack(signs)
    random_signs = np.random.choice([-1, 1], size=signs_array.shape)
    signs = np.where(signs == 0, random_signs, signs_array)

    H_RB_all = np.zeros((libopt.K, libopt.J, libopt.L),dtype=complex)
    h_UR_all = np.zeros((libopt.K, libopt.L),dtype=complex)
    h_d_all = np.zeros((libopt.K, libopt.J),dtype=complex)
    h_k_all = np.zeros((libopt.K, libopt.J), dtype=complex)
    x_est = []
    for k,x_k in zip(range(libopt.K),signs):
        d = len(x_k)
        assigned_subband = np.where(A[k, :] == 1)[0][0]
        H_RB_all[k] = H_RB[k, assigned_subband, :, :]
        h_UR_all[k] = h_UR[assigned_subband, :, k]
        h_d_all[k] = h_d[assigned_subband, :, k]
        h_k_all[k] = np.dot(H_RB_all[k], np.diag(theta) @ h_UR_all[k]) + h_d_all[k]
        h_k_repeated = np.tile(h_k_all[k], (d, 1))
        p_k = p[k]
        n_k = (np.random.randn(d,libopt.J)+1j*np.random.randn(d,libopt.J))/(2)**0.5*libopt.sigma
        y_k = np.sqrt(p_k) * h_k_repeated * x_k[:, np.newaxis] + n_k # d * J

        h_k_scaled =  h_k_repeated * np.sqrt(p_k) # d * J
        likelihood_plus = -np.linalg.norm(y_k - h_k_scaled, axis=1) ** 2 / (2 * libopt.sigma ** 2)
        likelihood_minus = -np.linalg.norm(y_k + h_k_scaled, axis=1) ** 2 / (2 * libopt.sigma ** 2)

        x_est_k = np.where(likelihood_plus > likelihood_minus, +1, -1)
        x_est.append(x_est_k)

    return x_est

def transmission_no_channel (w):
    signs = [torch.sign(tensor) for tensor in w]
    signs_tensor = torch.stack(signs)
    random_signs = torch.randint(low=0, high=2, size=signs_tensor.shape, device=signs_tensor.device) * 2 - 1
    signs_tensor = torch.where(signs_tensor == 0, random_signs, signs_tensor)

    return signs_tensor