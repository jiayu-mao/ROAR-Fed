#!/bin/bash

# Run baseline MNIST experiments with multiple seeds

SEEDS=(1 2 3 4 5)

for seed in "${SEEDS[@]}"; do

  echo "Running baseline MNIST with seed=$seed"

  python varying_onebit.py \
  --round_opt=5 \
  --epochs=120 \
  --b=30 \
  --lr=0.1 \
  --momentum=0 \
  --downlink=2 \
  --iid=0 \
  --model=logistic \
  --dataset=mnist \
  --L=45 \
  --round_sca=5 \
  --round_imax=9 \
  --round_jmax=9 \
  --diri_beta=0.5 \
  --seed=$seed

done