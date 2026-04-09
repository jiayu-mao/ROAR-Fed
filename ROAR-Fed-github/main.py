import argparse
import logging
import os

import torch
from torchvision import datasets, transforms
import numpy as np

import models
from trainer import server
from utils import write_data

import random

from datetime import datetime

class GroupNorm32(torch.nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, **kargs):
        super().__init__(num_groups, num_channels, **kargs)


parser = argparse.ArgumentParser(description="Pytorch Training")

parser.add_argument('--results_dir', default='./Results',
                    help='results dir')
parser.add_argument('--save_dir', default='noniid10_SNR10_beta10',
                    help='folder to save the result')
parser.add_argument('--dataset', default='mnist',
                    help='dataset', choices=['cifar10', 'mnist','fmnist'])
parser.add_argument('--niid', default='1', type=str,
                    help='Non-IId parameter')
parser.add_argument('--iid', type=int, default=0,
                    help='Default set to IID. Set to 0 for non-IID.')
parser.add_argument('--unequal', type=int, default=0,
                    help='whether to use unequal data splits for  \
                    non-i.i.d setting (use 0 for equal splits)')
parser.add_argument('--niid_diri', type=int, default=0,
                    help='set 1 for non-iid distribution-based label imbalance according to Dirichlet distribution')
parser.add_argument('--diri_beta', type=float, default=0.5, help='parameter for Dirichlet distribution')
parser.add_argument('--model', default='logistic',
                    help='model architecture',
                    choices=['cnn_cifar10', 'logistic', 'cnn_mnist'])
# ------ training parameters
parser.add_argument('--b', '--batch_size', type=int,
                    help='batch size for each client to train',
                    default=64)
parser.add_argument('--lr', '--learning_rate', type=float,
                    default=0.1, help='learning rate for local training')
parser.add_argument('--glr', '--global_lr', type=float, default=1.0,
                    help='global learning rate for server aggregation')
parser.add_argument('--momentum', type=float, default=0.0, help= 'momentum value for sgd optimizer')
parser.add_argument('--rounds', type=int, default=150,
                    help='number of rounds to run')
parser.add_argument('--optimizer', default='sgd', help='optimizer for client')
parser.add_argument('--clients_perc', type=float, default=1.0,
                    help='clients in each communication round')
parser.add_argument('--total_clients', type=int, default=10,
                    help='clients in each communication round')
parser.add_argument('--gpu', dest='gpu', action='store_true',
                    help='gpu or not')
parser.add_argument('--no-gpu', dest='gpu', action='store_false',
                    help='gpu not')
parser.set_defaults(gpu=True)
parser.add_argument('--local_epoch', type=int, default=1,
                    help='local update epoch')
parser.add_argument('--biased', type=int, default=0,
                    help='baised sampling parameter')
parser.add_argument('--local_opti', default='fedavg',
                    help='local optimization method',
                    choices=['fedavg', 'fedprox'])
parser.add_argument('--mu', type=float, default=0.1,
                    help='hyper-parameter for fedprox')
parser.add_argument('--show', type=int, default=5, choices=[1, 5, 10, 20, 50, 100], help='round to compute accuracy')
parser.add_argument('--algorithm', default='AOAFL',
                    help='algorithm type')
parser.add_argument('--SNR', type=float, default=10,
                    help='uplink signal-to-noise ratio, dB')
parser.add_argument('--SNR_dl', type=float, default=20,
                    help='downlink signal-to-noise ratio, dB')
parser.add_argument('--P', type=float, default=1,
                    help='Power Constraint')
parser.add_argument('--s_beta', type=float, default=50,
                    help='server beta')
parser.add_argument('--local_alg', default='AOAFL',
                    help='algorithm type (do not need to change)')
parser.add_argument('--client_scale', default='yes',
                    help='client scale (yes or no)',
                    choices=['yes', 'no'])
parser.add_argument('--cs_sigma', type=float,
                    default=1.0, help='sigma of h')
parser.add_argument('--cs_sigma_hat', type=float,
                    default=1.0, help='sigma of h_hat')
parser.add_argument('--csi', type=int, default=1,
                    help='perfect CSI (0 is imperfect CSI)')
parser.add_argument('--downlink', type=int, default=0, choices=[0,1,2],
                    help='noiseless downlink 0, noisy downlink perfect csi 1, noisy downlink imperfect csi 2')

# RIS communication system
parser.add_argument('--M', type=int, default=10, help='total # of devices') # same as total_clients
parser.add_argument('--N', type=int, default=1, help='# of BS antennas')
parser.add_argument('--L', type=int, default=16, help='RIS Size')
parser.add_argument('--RIS_num', type=int, default=1, help='number of RISs')
parser.add_argument('--set', type=int, default=1, help=r'=1 if concentrated devices+ euqal dataset;\
                    =2 if two clusters + unequal dataset')
parser.add_argument('--phase_design', default='sca',
                    help='RIS phase design method')
parser.add_argument('--phase_status', default='discrete',
                    help='RIS phase is continuous or discrete')

parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--weightdecay', type=float,default=0.0, help='weight decay parameter in SGD optimizer')
parser.add_argument('--clip_norm', type=float, default=60)
parser.add_argument('--dl_beta', type=float, default=50)


def main(server_model=None):
    global args
    args = parser.parse_args()
    set_seed(args.seed)
    args.save_dir = 'noniid' + '{}'.format(args.niid)  + '_beta' + '{}'.format(args.s_beta) \
                    + '_data'+ '{}'.format(args.dataset) + '_' +'{}'.format(args.model)
    if not torch.cuda.is_available():
        args.gpu = False
    args.device = torch.device('cuda:{}'.format(0) if args.gpu else 'cpu')
    save_path = os.path.join(args.results_dir, args.save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    time_tag = datetime.now().strftime("%m%d_%H%M")
    exp_tag = (
        f"RIS{args.L}"
        f"_lr{args.lr}"
        f"_dl{args.downlink}"
        f"_diri{args.diri_beta}"
        f"_seed{args.seed}"
        f"_rds{args.rounds}"
        f"_b{args.b}"
        f"_wd{args.weightdecay}"
        f"_{time_tag}"
    )
    file_name_txt = f"{exp_tag}.log"
    set_logging(os.path.join(save_path, file_name_txt))
    file_name_csv = f"{exp_tag}.csv"
    metrics = write_data.Metrics(os.path.join(save_path, file_name_csv),
                                os.path.join(save_path, file_name_csv))
    logging.info(args)
    logging.info('Results are saving to file: %s', save_path)

    # load datasets and clients
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        data_dir = '../data/mnist/'
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=trans_mnist)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=trans_mnist)

    elif args.dataset == 'cifar10':
        trans_cifar = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        train_dataset = datasets.CIFAR10('data/cifar/', train=True, download=True, transform=trans_cifar)
        test_dataset = datasets.CIFAR10('data/cifar/', train=False, download=True,
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]))
    elif args.dataset == 'fmnist':
        trans_fmnist = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5,), std=(0.5,))
                                        ])
        train_dataset = datasets.FashionMNIST('./data/FASHION_MNIST/', download=True, train=True,
                                              transform=trans_fmnist)
        test_dataset = datasets.FashionMNIST('./data/FASHION_MNIST/', download=True, train=False,
                                             transform=trans_fmnist)

    dict_client=args.iid

    # --------
    if args.model == 'logistic':
        server_model = models.logistic.Logistic()
    elif args.model == 'cnn_cifar10':
        server_model = models.cnn_cifar10.CifarCnn()
    elif args.model == 'cnn_mnist':
        server_model = models.cnn_mnist.CNNMnist()
    if args.gpu:
        server_model = server_model.to(args.device)

    logging.info('load model %s successful', args.model)

    # RIS communication system
    args.alpha_direct = 4
    args.BS = np.array([-50, 0, 10])
    if args.RIS_num == 1:
        args.RIS = np.array([0, 0, 10])
    else:
        args.RIS = np.zeros((args.RIS_num,3))
    args.range = 20
    args.sigma = 0
    fc = 915 * 10 ** 6
    BS_Gain = 10 ** (5.0 / 10)
    RIS_Gain = 10 ** (5.0 / 10)
    User_Gain = 10 ** (0.0 / 10)
    d_RIS = 1.0 / 10
    ref = (1e-10) ** 0.5
    args.ref = ref
    sigma_n = np.power(10, -args.SNR / 10)
    args.sigma = sigma_n / ref ** 2
    if args.set == 1:
        args.K = np.ones(args.M, dtype=int) * int(30000.0 / args.M)
        print(sum(args.K))
        args.dx2 = np.random.rand(int(args.M - np.round(args.M / 2))) * args.range - args.range
    else:
        args.K = np.random.randint(1000, high=2001, size=(int(args.M)))
        lessuser_size = int(args.M / 2)
        args.K2 = np.random.randint(100, high=201, size=(lessuser_size))
        args.lessuser = np.random.choice(args.M, size=lessuser_size, replace=False)
        args.K[args.lessuser] = args.K2
        print(sum(args.K))
        args.dx2 = np.random.rand(int(args.M - np.round(args.M / 2))) * args.range + 100
    args.dx1 = np.random.rand(int(np.round(args.M / 2))) * args.range - args.range
    args.dx = np.concatenate((args.dx1, args.dx2))
    args.dy = np.random.rand(args.M) * 20 - 10
    if args.RIS_num == 1:
        args.d_UR = ((args.dx - args.RIS[0]) ** 2 + (args.dy - args.RIS[1]) ** 2 + args.RIS[2] ** 2
                 ) ** 0.5
        args.d_RB = np.linalg.norm(args.BS - args.RIS)
    else:
        args.RIS[:,0] = args.dx
        args.RIS[:,1] = args.dy
        args.RIS[:,2] = np.ones(args.RIS_num) * 2
        args.d_UR = args.RIS[:,2]
        args.d_RB = ((args.BS[0] - args.RIS[:,0]) ** 2 + (args.BS[1] - args.RIS[:,1]) ** 2 + (args.BS[2] - args.RIS[:,2]) ** 2
                     ) ** 0.5
    args.d_direct = ((args.dx - args.BS[0]) ** 2 + (args.dy - args.BS[1]) ** 2 + args.BS[2] ** 2
                     ) ** 0.5
    args.PL_direct = BS_Gain * User_Gain * (3 * 10 ** 8 / fc / 4 / np.pi / args.d_direct) ** args.alpha_direct
    args.PL_RIS = BS_Gain * User_Gain * RIS_Gain * args.L ** 2 * d_RIS ** 2 / 4 / np.pi \
                  * (3 * 10 ** 8 / fc / 4 / np.pi / args.d_UR) ** 2 * (
                          3 * 10 ** 8 / fc / 4 / np.pi / args.d_RB) ** 2

    master = server.Server(server_model, train_dataset, test_dataset, dict_client, args, metrics)
    master.train()
    argsDict = args.__dict__
    file_add_txt = save_path + '_args_'+ file_name_csv
    with open(file_add_txt,'w') as f:
        f.writelines('------------------- start ----------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ':' + str(value) + '\n')
        f.writelines('------------------- end ----------------')

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
    main()
