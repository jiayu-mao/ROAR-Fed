#!/bin/bash

# Run baseline FMNIST experiments with multiple seeds and Dirichlet values

SEEDS=(1 2 3 4 5)
BETAS=(0.5 1)

for beta in "${BETAS[@]}"; do
  for seed in "${SEEDS[@]}"; do

    echo "Running FMNIST baseline with seed=$seed, diri_beta=$beta"

    python varying_onebit.py \
    --round_opt=5 \
    --epochs=121 \
    --b=128 \
    --lr=0.02 \
    --momentum=0.9 \
    --downlink=2 \
    --iid=0 \
    --model=cnn_mnist \
    --dataset=fmnist \
    --L=100 \
    --round_sca=5 \
    --round_imax=9 \
    --round_jmax=9 \
    --diri_beta=$beta \
    --seed=$seed

  done
done