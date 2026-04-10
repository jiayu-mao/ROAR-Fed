
# Baseline: One-Bit RIS-FL

This folder contains our **re-implementation and adaptation** of the following work:

**H. Li, R. Wang, W. Zhang, and J. Wu**,  
*"One Bit Aggregation for Federated Edge Learning With Reconfigurable Intelligent Surface: Analysis and Optimization,"*  
IEEE Transactions on Wireless Communications, 22(2), 872–888, 2023.  
DOI: https://doi.org/10.1109/TWC.2022.3198881

---

## Overview

This code is **not a direct reproduction under the original paper settings**.

Instead, it is adapted and re-implemented to match the **same experimental pipeline and settings used in our ROAR-Fed paper**, in order to ensure a fair comparison.

---

## Environment Setup

The environment file is provided in this folder:

```bash
environment.yml
````

To create the environment:

```bash
conda env create -f environment.yml
conda activate onebit
```

---

## Precomputed Parameters (Important)

This implementation relies on **precomputed channel realizations and optimization parameters**.

Please download them from:

👉 https://drive.google.com/drive/folders/1-BYXtLITZ5mrI6T2RkDcOj230ITZ2D7v?usp=sharing

After downloading, place the files in the project root directory (same level as `varying_onebit.py`).

---

## Running the Code

All experiment configurations are provided in:

```bash
scripts/
```

For example:

```bash
bash scripts/run_cifar10_baseline.sh
```

---

## Important: Enable Loading of Precomputed Parameters

Before running, you must **enable loading of the precomputed files**.

In `varying_onebit.py`, go to:

```
Lines 274–282
```

Uncomment the corresponding lines depending on the dataset.

For example, for **CIFAR-10**:

```python
channels = pickle.load(open("channels_L128_rds500.p", "rb"))
opt_params = pickle.load(open("opt_params_L128_rds500.p", "rb"))
```

---

## How the Parameters Were Generated

We also provide the code used to generate these parameters. The optimization stage is computationally expensive and time-consuming, therefore we provide precomputed parameters to facilitate efficient reproducibility.

In `varying_onebit.py`, go to:

```
Lines 169–271
```

Uncomment this block to:

* generate channel realizations
* solve the optimization problem
* save the resulting parameters

This corresponds to the **optimization stage described in the original paper**.

---

## Notes

* The pipeline separates:

  * optimization (channel + RIS parameter generation)
  * federated learning training

* The provided `.sh` scripts assume that **precomputed parameters are already available**.

* All experiments use the **same settings as our ROAR-Fed paper** for fair comparison.

---

## Reproducibility

To reproduce results:

1. Download precomputed parameters from Google Drive
2. Place them in the correct directory
3. Uncomment the loading code (Lines 274–282)
4. Run the corresponding script in `scripts/`

---

## Contact

For questions, please contact the first author.

