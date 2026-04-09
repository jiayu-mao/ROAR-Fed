#!/bin/bash

# Run CIFAR-10 experiments with different seeds and downlink settings

SEEDS=(1 2 3 4 5)
DOWNLINKS=(0 2)

for seed in "${SEEDS[@]}"; do
  for downlink in "${DOWNLINKS[@]}"; do

    echo "Running CIFAR10 seed=$seed downlink=$downlink"

    python main.py \
    --niid=1 \
    --rounds=500 \
    --SNR=20 \
    --cs_sigma_hat=0.1 \
    --csi=0 \
    --phase_design=sca \
    --algorithm=AOAFL \
    --iid=0 \
    --L=128 \
    --local_alg=AOAFL \
    --s_beta=100 \
    --client_scale=yes \
    --show=1 \
    --M=10 \
    --total_clients=10 \
    --RIS_num=1 \
    --b=512 \
    --downlink=$downlink \
    --SNR_dl=30 \
    --dataset=cifar10 \
    --model=cnn_cifar10 \
    --lr=0.1 \
    --local_epoch=1 \
    --niid_diri=1 \
    --diri_beta=1 \
    --momentum=0.9 \
    --phase_status=continue \
    --seed=$seed

  done
done