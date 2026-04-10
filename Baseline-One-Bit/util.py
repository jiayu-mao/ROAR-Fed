#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_noniid
from sampling import noniid_dirich

def get_dataset(args):
    if args.dataset == 'cifar10':
        data_dir = '../data/cifar/'
        trans_cifar = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]))
        user_groups = noniid_dirich(
            train_dataset,
            args.K,
            N=50000,
            beta=args.diri_beta,
            seed=2025,
        )

    elif args.dataset == 'mnist':
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        user_groups = mnist_noniid(train_dataset, args.K)

    elif args.dataset == 'fmnist':
        data_dir = '../data/FASHION_MNIST/'
        trans_fmnist = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.5,), std=(0.5,))
                                           ])
        train_dataset = datasets.FashionMNIST(data_dir, download=True, train=True,
                                              transform=trans_fmnist)
        test_dataset = datasets.FashionMNIST(data_dir, download=True, train=False,
                                             transform=trans_fmnist)
        user_groups = noniid_dirich(
            train_dataset,
            args.K,
            N=60000,
            beta=args.diri_beta,
            seed=2026,
        )

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] = w_avg[key] + w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


