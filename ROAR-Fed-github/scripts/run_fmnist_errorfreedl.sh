#!/bin/bash

# Run experiments for different Dirichlet beta values and seeds (FMNIST, downlink=0)

SEEDS=(1 2 3 4 5)
BETAS=(0.1 0.5 1.0)

for seed in "${SEEDS[@]}"; do
  for beta in "${BETAS[@]}"; do

    echo "Running seed=$seed diri_beta=$beta downlink=0"

    python main.py \
    --niid=1 \
    --rounds=120 \
    --SNR=20 \
    --cs_sigma_hat=0.1 \
    --csi=0 \
    --phase_design=sca \
    --algorithm=AOAFL \
    --iid=0 \
    --L=100 \
    --local_alg=AOAFL \
    --s_beta=100 \
    --client_scale=yes \
    --show=1 \
    --M=10 \
    --total_clients=10 \
    --RIS_num=1 \
    --b=64 \
    --downlink=0 \
    --SNR_dl=30 \
    --dataset=fmnist \
    --model=cnn_mnist \
    --lr=0.05 \
    --local_epoch=1 \
    --niid_diri=1 \
    --diri_beta=$beta \
    --momentum=0.9 \
    --phase_status=continue \
    --seed=$seed

  done
done