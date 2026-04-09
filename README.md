
# ROAR-Fed

This repository provides the code associated with our paper:

**Adaptive Personalized Over-the-Air Federated Learning with Reflecting Intelligent Surfaces**
arXiv: [2412.03514](https://arxiv.org/abs/2412.03514)

## Overview

This repository is organized to support the reproducibility of the main experiments reported in our paper.

At present, the repository contains the following folder:

* **`ROAR-Fed-github/`**: implementation of our proposed **ROAR-Fed** method.

A second folder will be added later:

* **`Baseline-One-bit/`**: our reproduction of the following baseline method, used in our paper for comparison:
  **H. Li, R. Wang, W. Zhang, and J. Wu, "One Bit Aggregation for Federated Edge Learning With Reconfigurable Intelligent Surface: Analysis and Optimization," IEEE Transactions on Wireless Communications, 22(2), 872–888, 2023.**
  DOI: [10.1109/TWC.2022.3198881](https://doi.org/10.1109/TWC.2022.3198881)

Another baseline used in our paper already has an official public implementation:

* **RIS-FL**: [https://github.com/liuhang1994/RIS-FL](https://github.com/liuhang1994/RIS-FL)

In our experiments, all baseline methods were evaluated under the same experimental settings as our proposed method whenever applicable.

## Note on the public release

The public code in `ROAR-Fed-github/` contains the implementation of the main **ROAR-Fed** framework used to reproduce the reported results.

Some components related to our personalized extension are **not included** in this public release, since they are associated with patent-related content and therefore cannot be disclosed at this time.

## Repository structure

```text
ROAR-Fed/
└── ROAR-Fed-github/
    ├── models/
    ├── scripts/
    ├── trainer/
    ├── utils/
    ├── RIS.py
    ├── environment.yml
    ├── main.py
    ├── sampling.py
    ├── update.py
    └── util.py
```

## Environment setup

The environment used for our experiments is provided in:

```text
ROAR-Fed-github/environment.yml
```

To create the conda environment, run:

```bash
conda env create -f ROAR-Fed-github/environment.yml
conda activate roar-fed
```

## Datasets

The experiments use standard public datasets, including:

* MNIST
* Fashion-MNIST
* CIFAR-10

These datasets are automatically downloaded by `torchvision` if they are not already available.

To improve reproducibility, the dataset split is fixed in the code using a predefined seed.

## Running the code

First move into the code directory:

```bash
cd ROAR-Fed-github
```

We provide shell scripts in the `scripts/` folder to reproduce the main experiment settings used in the paper.

### Main CIFAR-10 experiments

```bash
bash scripts/run_cifar10.sh
```

### MNIST experiments with different RIS element settings

```bash
bash scripts/run_L_sweep.sh
```

### Fashion-MNIST experiments with different Dirichlet beta values (`downlink = 2`)

```bash
bash scripts/run_fmnist_beta.sh
```

### Fashion-MNIST experiments with different Dirichlet beta values (`downlink = 0`)

```bash
bash scripts/run_fmnist_beta_downlink0.sh
```

## Main experimental settings

A representative configuration for the main CIFAR-10 experiments is as follows:

```text
--rounds=500
--SNR=20
--cs_sigma_hat=0.1
--csi=0
--phase_design=sca
--algorithm=AOAFL
--iid=0
--L=128
--local_alg=AOAFL
--s_beta=100
--client_scale=yes
--show=1
--M=10
--total_clients=10
--RIS_num=1
--b=512
--downlink=0
--SNR_dl=30
--dataset=cifar10
--model=cnn_cifar10
--lr=0.1
--local_epoch=1
--niid_diri=1
--diri_beta=1
--momentum=0.9
--phase_status=continue
--seed=1,2,3,4,5
```

For other figures and tables in the paper, the exact settings are provided in the corresponding shell scripts under `ROAR-Fed-github/scripts/`.

## Reproducibility statement

This public release is a minimal and cleaned implementation intended for reproducibility of the experiments reported in the paper.

To improve reproducibility:

* experiment scripts corresponding to the main settings are provided;
* random seeds are explicitly controlled in the runs;
* dataset splitting is fixed in the code;
* the environment file used in our implementation is included.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{mao2024adaptive,
  title={Adaptive Personalized Over-the-Air Federated Learning with Reflecting Intelligent Surfaces},
  author={Mao, Jiayu and Yener, Aylin},
  journal={arXiv preprint arXiv:2412.03514},
  year={2024}
}
```

## Contact

For questions regarding the code release, please open an issue in this repository.

