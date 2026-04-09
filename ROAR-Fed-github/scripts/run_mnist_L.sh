#!/bin/bash

# Run experiments for different L values and seeds

SEEDS=(1 2 3 4 5)
L_VALUES=(35 65)

for seed in "${SEEDS[@]}"; do
  for L in "${L_VALUES[@]}"; do

    echo "Running seed=$seed L=$L"

    python main.py \
    --niid=1 \
    --rounds=120 \
    --SNR=20 \
    --cs_sigma_hat=0.1 \
    --csi=0 \
    --phase_design=sca \
    --algorithm=AOAFL \
    --iid=0 \
    --L=$L \
    --local_alg=AOAFL \
    --s_beta=60 \
    --client_scale=yes \
    --show=1 \
    --M=10 \
    --total_clients=10 \
    --RIS_num=1 \
    --b=30 \
    --downlink=2 \
    --SNR_dl=30 \
    --dataset=mnist \
    --model=logistic \
    --lr=0.1 \
    --local_epoch=1 \
    --niid_diri=0 \
    --momentum=0.9 \
    --phase_status=continue \
    --seed=$seed \
    --weightdecay=0.1

  done
done