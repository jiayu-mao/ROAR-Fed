import numpy as np
from numpy import linalg as LA


def phase_sca(L, theta_init, ls, s_t, G, mu_sca, iter_sca):
    max_index = ls.index(max(ls))
    s_t_i = s_t[0,max_index]
    g_t_i = G[:, :, max_index]
    g_t_i = g_t_i.transpose()
    a = g_t_i[:,0]
    v = g_t_i * s_t_i
    v = v[:,0]
    U = g_t_i * g_t_i.T.conjugate()
    phi = np.angle(theta_init)
    lst_norm = []
    theta = theta_init
    for i in range(iter_sca):
        tmp = s_t_i - np.dot(a.conjugate(), theta)
        tmp_norm = LA.norm(tmp)
        lst_norm.append(tmp_norm)
        theta_conj = np.conj(theta)
        tmp_real = - 1j * theta_conj * (np.dot(U, theta) - v)
        grad = 2 * np.real(tmp_real)
        phi = phi - 1/mu_sca * grad
        theta = np.exp(1j*phi)
    return theta

