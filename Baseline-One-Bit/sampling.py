#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    num_shards = num_users
    num_imgs = int(60000 / num_shards)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs = np.arange(num_shards * num_imgs)

    labels = dataset.targets.numpy()
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    for i in range(num_users):
        rand = np.random.choice(idx_shard, 1, replace=False)[0]
        idx_shard.remove(rand)
        dict_users[i] = idxs[rand * num_imgs:(rand + 1) * num_imgs]

    return dict_users

def noniid_dirich(dataset, num_users, N, beta, seed=2025):
    rng = np.random.default_rng(seed)

    min_size = 0
    min_require_size = 10
    num_class = 10

    idxs = np.arange(N)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(num_class):
            idx_k = np.where(idxs_labels[1] == k)[0]
            rng.shuffle(idx_k)
            proportions = rng.dirichlet(np.repeat(beta, num_users))
            proportions = np.array([
                p * (len(idx_j) < int(N / num_users))
                for p, idx_j in zip(proportions, idx_batch)
            ])
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, split_points))]
            min_size = min(len(idx_j) for idx_j in idx_batch)

    for j in range(num_users):
        rng.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map

if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
