# -*- coding: utf-8 -*-
import numpy as np
import logging
import os

np.set_printoptions(precision=6, threshold=1e3)
import argparse
import torch

from scipy.stats import norm
from optimization import subband_assign,power_assign,phase_design
import models
from trainer import server

from utils import write_data
import pickle
import random


def initial():
    libopt = argparse.ArgumentParser()
    libopt.add_argument('--K', type=int, default=10, help='total # of devices')
    libopt.add_argument('--J', type=int, default=1, help='# of BS antennas')
    libopt.add_argument('--L', type=int, default=16, help='RIS Size')
    libopt.add_argument('--M',type=int, default=100, help='total # of sub-bands')

    # optimization parameters
    libopt.add_argument('--round_opt', type=int, default=1, help='a_max, # of total optimization loops')
    libopt.add_argument('--round_sca', type=int, default=10, help='d_max, # of SCA loops in sub_band assignment')
    libopt.add_argument('--round_imax',type=int, default=20, help='i_max, # of loops in algorithm 1 power assign')
    libopt.add_argument('--round_jmax',type=int, default=20, help='j_max, # of loops in algorithm 1 power assign')
    libopt.add_argument('--round_phase',type=int, default=10, help='# of loops in RIS phase design')

    # learning parameters
    libopt.add_argument('--gpu', type=int, default=0, help=r'Use Which Gpu')
    libopt.add_argument('--model', default='cnn_mnist', help='model architecture',choices=['logistic','cnn_mnist','cnn_cifar10'])
    libopt.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    libopt.add_argument('--unequal', type=int, default=0, help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    libopt.add_argument('--niid_diri', type=int, default=1,
                        help='set 1 for non-iid distribution-based label imbalance according to Dirichlet distribution')
    libopt.add_argument('--diri_beta', type=float, default=0.5, help='parameter for Dirichlet distribution')
    libopt.add_argument('--epochs', type=int, default=1, help="rounds of training,T")
    libopt.add_argument('--b', '--batch_size', type=int,
                        help='batch size for each client to train', default=64)
    libopt.add_argument('--show', type=int, default=1, choices=[1, 5, 10, 20, 50, 100],
                        help='round to compute accuracy')
    libopt.add_argument('--dataset', default='fmnist',
                        help='dataset', choices=['cifar10', 'mnist', 'fmnist'])
    libopt.add_argument('--optimizer', default='sgd', help='optimizer for client')
    libopt.add_argument('--lr', '--learning_rate', type=float,
                        default=0.01, help='learning rate for local training')
    libopt.add_argument('--glr', '--global_lr', type=float, default=1.0,
                        help='global learning rate for server aggregation')
    libopt.add_argument('--momentum', type=float, default=0.0, help='momentum value for sgd optimizer')
    libopt.add_argument('--local_epoch', type=int, default=1, help='local update epoch')
    libopt.add_argument('--local_opti', default='fedavg',help='local optimization method',
                         choices=['fedavg', 'fedprox'])
    libopt.add_argument('--algorithm', default='onebit',
                        help='algorithm type')
    libopt.add_argument('--downlink', type=int, default=1, choices=[0, 1, 2],
                        help='noiseless downlink 0, noisy downlink perfect csi 1, noisy downlink imperfect csi 2,')
    libopt.add_argument('--SNR_dl', type=float, default=30,
                        help='downlink signal-to-noise ratio, dB')
    libopt.add_argument('--SNR', type=float, default=20.0)
    libopt.add_argument('--cs_sigma_hat', type=float, default=0.1, help='sigma of h_hat')
    libopt.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    args = libopt.parse_args()
    return args

def objective_func (libopt,A,theta,p,H_RB,h_UR,h_d):
    nonzero_elements = np.count_nonzero(p)
    h = np.zeros((libopt.J, libopt.K, libopt.M), dtype=complex)
    for k in range(libopt.K):
        for m in range(libopt.M):
            h[:, k, m] = np.dot(H_RB[k, m, :, :], np.diag(theta) @ h_UR[m, :, k]) + h_d[m, :, k]
    h_norm_squared = np.linalg.norm(h, axis=0) ** 2
    p_broadcasted = np.broadcast_to(p[:, None], A.shape)
    product_1 = A * h_norm_squared
    product_2 = product_1 * p_broadcasted
    x = 2 * np.sum(product_2, axis=1) / libopt.sigma ** 2
    Q_sqrt_values = norm.sf(np.sqrt(x))
    sum_value = np.sum(1 - 2 * Q_sqrt_values)
    obj_value = np.sqrt(nonzero_elements) / sum_value

    return obj_value

def set_logging(logfile = 'log.txt'):
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=logfile,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    libopt = initial()
    set_seed(libopt.seed)
    print(libopt)
    libopt.save_dir = 'noniid' + '{}'.format(libopt.iid) + '_data'+ '{}'.format(libopt.dataset) + '_' +'{}'.format(libopt.model)
    libopt.results_dir = './Results'
    libopt.device = torch.device(
        'cuda:{}'.format(libopt.gpu) if torch.cuda.is_available() and libopt.gpu != -1 else 'cpu')
    print(libopt.device)
    save_path = os.path.join(libopt.results_dir, libopt.save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    exp_tag = (
        f"RIS{libopt.L}"
        f"_lr{libopt.lr}"
        f"_dl{libopt.downlink}"
        f"_diri{libopt.diri_beta}"
        f"_seed{libopt.seed}"
        f"_rds{libopt.epochs}"
        f"_b{libopt.b}"
    )
    file_name_txt = f"{exp_tag}.log"
    set_logging(os.path.join(save_path, file_name_txt))
    file_name_csv = f"{exp_tag}.csv"
    metrics = write_data.Metrics(os.path.join(save_path, file_name_csv),
                                os.path.join(save_path, file_name_csv))
    logging.info(libopt)
    logging.info('Results are saving to file: %s', save_path)

    eps = 1e-5
    libopt.alpha_direct = 4
    fc = 915 * 10 ** 6
    BS_Gain = 10 ** (5.0 / 10)
    RIS_Gain = 10 ** (5.0 / 10)
    User_Gain = 10 ** (0.0 / 10)
    d_RIS = 1.0 / 10
    libopt.BS = np.array([-50, 0, 10])
    libopt.RIS = np.array([0, 0, 10])
    libopt.range = 20
    libopt.P0 = 1
    ref = (1e-10) ** 0.5
    libopt.ref = ref
    sigma_n = 10 ** (libopt.SNR / 10)
    libopt.sigma = libopt.P0 / sigma_n
    libopt.dx2 = np.array([-9.02372992, -5.69621267, -7.94473248, -9.10233634, -11.52690401])
    libopt.dx1 = np.array([-7.08211774, -11.24825577, -2.16453998, -0.72674479, -12.33116962])
    libopt.dx = np.concatenate((libopt.dx1, libopt.dx2))
    libopt.dy = np.array([5.83450076, 0.5778984, 1.36089122, 8.51193277, -8.57927884, -8.25741401, -9.59563205, 6.65239691, 5.56313502,
         7.40024296])
    libopt.d_UR = np.array([13.57194867, 15.06178026, 10.32168873, 13.15223013, 18.04610123, 15.7991325, 14.98409134, 14.4004569,
         14.62193551, 16.95974976])
    libopt.d_RB = 50.0
    libopt.d_direct = np.array([44.45206426, 40.02538753, 48.88847778, 50.99320228, 39.90670128, 42.97952531, 46.42091906, 43.73671125,
         42.46842785, 40.43442482])
    libopt.PL_direct = np.array([3.75312866e-13, 5.70977513e-13, 2.56528207e-13, 2.16726513e-13, 5.77800437e-13, 4.29452098e-13, 3.15577658e-13,
         4.00476183e-13, 4.50501717e-13, 5.48221556e-13])
    libopt.PL_RIS = np.array([2.05006137e-12, 1.66455697e-12, 3.54445742e-12, 2.18299369e-12, 1.15953670e-12, 1.51281143e-12,
         1.68186238e-12, 1.82095300e-12, 1.76620679e-12, 1.31284181e-12])

#================ can run below optimization problem first and seperately from machine learning ====================
    # result_record =[]
    # channels = {}
    # opt_params = {}
    #
    # for i in range(libopt.epochs):
    #     print('this is the {}-th channel realization'.format(i))
    #     theta_init = np.ones(libopt.L) + 1j * np.zeros(libopt.L)
    #     p_init = np.ones(libopt.K) * libopt.P0 / libopt.K
    #     A0 = np.zeros((libopt.K, libopt.M))
    #     min_dim = min(libopt.K, libopt.M)
    #     A0[np.arange(min_dim), np.arange(min_dim)] = 1
    #     theta = theta_init
    #     p = p_init
    #
    #     h_d = np.zeros((libopt.M, libopt.J, libopt.K), dtype=complex)
    #     for m in range(libopt.M):
    #         h_d_1 = (np.random.randn(libopt.J, libopt.K) + 1j * np.random.randn(libopt.J, libopt.K)) / 2 ** 0.5
    #         h_d_2 = h_d_1 @ np.diag(libopt.PL_direct ** 0.5) / ref
    #         h_d[m,:,:] = h_d_2
    #
    #     H_RB = np.zeros((libopt.K, libopt.M, libopt.J, libopt.L), dtype=complex)
    #     for k in range(libopt.K):
    #         H_RB_2 = np.zeros((libopt.M, libopt.J, libopt.L), dtype=complex)
    #         for m in range(libopt.M):
    #             H_RB_1 = (np.random.randn(libopt.J, libopt.L) + 1j * np.random.randn(libopt.J, libopt.L)) / 2 ** 0.5
    #             H_RB_2[m, :, :] = H_RB_1
    #         H_RB[k, :, :, :] = H_RB_2
    #
    #     h_UR = np.zeros((libopt.M, libopt.L, libopt.K), dtype=complex)
    #     for m in range(libopt.M):
    #         h_UR_1 = (np.random.randn(libopt.L, libopt.K) + 1j * np.random.randn(libopt.L, libopt.K)) / 2 ** 0.5
    #         h_UR_2 = h_UR_1 @ np.diag(libopt.PL_RIS ** 0.5) / ref
    #         h_UR[m, :, :] = h_UR_2
    #
    #     H_RB_noise = np.zeros((libopt.K, libopt.M, libopt.J, libopt.L), dtype=complex)
    #     for k in range(libopt.K):
    #         H_RB_noise2 = np.zeros((libopt.M, libopt.J, libopt.L), dtype=complex)
    #         for m in range(libopt.M):
    #             H_RB_noise1 = (libopt.cs_sigma_hat * (np.random.randn(libopt.J, libopt.L) + 1j * np.random.randn(libopt.J, libopt.L))) / 2 ** 0.5
    #             H_RB_noise2[m, :, :] = H_RB_noise1
    #         H_RB_noise[k, :, :, :] = H_RB_noise2
    #     h_UR_noise = np.zeros((libopt.M, libopt.L, libopt.K), dtype=complex)
    #     for m in range(libopt.M):
    #         h_UR_noise1 = (np.random.randn(libopt.L, libopt.K) + 1j * np.random.randn(libopt.L, libopt.K)) / 2 ** 0.5
    #         h_UR_noise2 = h_UR_noise1 @ np.diag((libopt.cs_sigma_hat * libopt.PL_RIS) ** 0.5) / ref
    #         h_UR_noise[m, :, :] = h_UR_noise2
    #     h_d_noise = np.zeros((libopt.M, libopt.J, libopt.K), dtype=complex)
    #     for m in range(libopt.M):
    #         h_d_noise1 = (np.random.randn(libopt.J, libopt.K) + 1j * np.random.randn(libopt.J, libopt.K)) / 2 ** 0.5
    #         h_d_noise2 = h_d_noise1 @ np.diag((libopt.cs_sigma_hat * libopt.PL_direct) ** 0.5) / ref
    #         h_d_noise[m, :, :] = h_d_noise2
    #
    #     h_d_est = h_d + h_d_noise
    #     h_UR_est = h_UR + h_UR_noise
    #     H_RB_est = H_RB + H_RB_noise
    #
    #     if libopt.downlink == 1 or libopt.downlink==0:
    #         h_d_o = h_d
    #         h_UR_o = h_UR
    #         H_RB_o = H_RB
    #     else:
    #         h_d_o = h_d_est
    #         h_UR_o = h_UR_est
    #         H_RB_o = H_RB_est
    #     channels[i] = (h_d, H_RB, h_UR)
    #
    #     for t in range(libopt.round_opt):
    #         d_max = libopt.round_sca
    #         tol_sca = eps
    #         optimal_A,mu_record,obj_record = subband_assign(libopt,H_RB_o,h_UR_o,h_d_o,theta,p,A0,d_max,tol_sca)
    #         threshold = 0.5
    #         A_binary = np.where(optimal_A>threshold, 1, 0)
    #         for k in range(libopt.K):
    #             if np.sum(A_binary[k, :]) == 0:
    #                 best_m = np.argmax(optimal_A[k, :])
    #                 A_binary[k, best_m] = 1
    #         l=0
    #         u=50
    #         imax = libopt.round_imax
    #         jmax = libopt.round_jmax
    #         optimal_p,f2,f1 = power_assign (libopt,A_binary,theta,p,libopt.P0,H_RB_o,h_UR_o,h_d_o,l,u,eps,imax,jmax)
    #
    #         rho = 1
    #         iter_max = libopt.round_phase
    #         optimal_theta=phase_design(libopt, A_binary, optimal_p, theta, H_RB_o, h_UR_o, h_d_o, rho,iter_max)
    #
    #         A0 = A_binary
    #         p = optimal_p
    #         theta = optimal_theta
    #         obj_value = objective_func(libopt, A0, theta, p,H_RB_o,h_UR_o,h_d_o)
    #         result_record.append(obj_value)
    #         if t>0:
    #             if abs(result_record[t-1] - obj_value) < eps:
    #                 break
    #         print(f'The objective value in overall iteration {t} is: {obj_value}')
    #     opt_params[i] = (A0,p,theta)
    # filename_channels = f"channels_L{libopt.L}_rds{libopt.epochs}.p"
    # with open(filename_channels, 'wb') as handle:
    #     pickle.dump(channels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # filename_opt = f"opt_params_L{libopt.L}_rds{libopt.epochs}.p"
    # with open(filename_opt, 'wb') as handle:
    #     pickle.dump(opt_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ================ can run above optimization problem first and seperately from machine learning ====================
    ## Fashion-MNIST
    # channels = pickle.load(open("channels_L100_rds121.p","rb"))
    # opt_params = pickle.load(open("opt_params_L100_rds121.p","rb"))
    ## MNIST
    # channels = pickle.load(open("channels_L45_compare.p", "rb"))
    # opt_params = pickle.load(open("opt_params_L45_compare.p","rb"))
    ## Cifar-10
    channels = pickle.load(open("channels_L128_rds500.p", "rb"))
    opt_params = pickle.load(open("opt_params_L128_rds500.p", "rb"))

    if libopt.model == 'logistic':
        server_model = models.logistic.Logistic()
    elif libopt.model == 'cnn_mnist':
        server_model = models.cnn_mnist.CNNMnist()
    elif libopt.model == 'cnn_cifar10':
        server_model = models.cnn_cifar10.CifarCnn()

    if libopt.gpu==0:
        server_model = server_model.to(libopt.device)
    dict_client = libopt.iid
    master = server.Server(server_model, dict_client, libopt, metrics, opt_params, channels)
    master.train()