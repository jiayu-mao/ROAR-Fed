#!/bin/bash

# Run baseline CIFAR-10 experiments with multiple seeds

SEEDS=(1 2 3 4 5)

for seed in "${SEEDS[@]}"; do

  echo "Running CIFAR-10 baseline with seed=$seed"

  python varying_onebit.py \
  --round_opt=2 \
  --epochs=500 \
  --b=512 \
  --lr=0.01 \
  --momentum=0.9 \
  --downlink=2 \
  --iid=0 \
  --dataset=cifar10 \
  --model=cnn_cifar10 \
  --L=128 \
  --round_sca=5 \
  --round_imax=9 \
  --round_jmax=9 \
  --diri_beta=1 \
  --seed=$seed

done