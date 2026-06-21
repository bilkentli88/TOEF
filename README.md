# Beyond Fidelity: Trust-Oriented Evaluation of Synthetic ECG Generators for Medical AI

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research-orange)
![Reproducibility](https://img.shields.io/badge/focus-reproducibility-informational)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This repository supports the paper:

**Beyond Fidelity: Trust-Oriented Evaluation of Synthetic ECG Generators for Medical AI**

Synthetic electrocardiogram (ECG) generation is increasingly used for medical AI development, data augmentation, and model evaluation. However, commonly used scalar metrics such as distributional fidelity or downstream utility can miss clinically relevant failure modes. A generator may appear realistic or useful under one metric while producing physiologically invalid, unstable, or class-imbalanced synthetic signals.

This repository provides code and analysis artifacts for a **Trust-Oriented Evaluation Framework** that represents generator behavior through a multi-dimensional **Trust Profile** rather than a single aggregate score.

The Trust Profile contains six dimensions:

1. **Distributional fidelity**
2. **Morphological diversity**
3. **Generation stability**
4. **Diagnostic utility**
5. **Class-wise reliability**
6. **Physiological validity**

The framework is evaluated using:

- a controlled synthetic time-series sanity check,
- the MIT-BIH Arrhythmia Database with five AAMI-style classes `{N,S,V,F,Q}`,
- the MIT-BIH Supraventricular Arrhythmia Database (SVDB) as an external companion cohort using `{N,S,V}`,
- three deep generator families: **TimeGAN**, **LSTM-VAE**, and **DDPM**,
- two diagnostic proxy generators: **Baseline** and **Robust**.

## Main purpose

The code is intended to reproduce the paper's central empirical claim:

> Synthetic ECG generators should not be ranked using fidelity or utility alone. Their suitability depends on the deployment context and on trade-offs among fidelity, diversity, stability, utility, class-wise reliability, and physiological validity.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── .gitignore
├── scripts/
│   ├── mitbih_multi_class_colab.py
│   ├── svdb_multi_class_colab_AIM.py
│   ├── ecg_ddpm_colab.py
│   ├── evaluate_ddpm_mitbih_trust.py
│   ├── evaluate_ddpm_svdb_trust.py
│   └── generate_figure2_mitbih_trust_profile_ddpm.py
├── results/
│   ├── mitbih_trust_profiles_summary.csv
│   ├── svdb_trust_profiles_summary.csv
│   ├── ddpm_trust_profiles_summary.csv
│   └── ddpm_svdb_trust_profiles_summary.csv
├── figures/
│   ├── Figure1_Framework_AIM.pdf
│   └── Figure2_MIT_BIH_Trust_Profile_DDPM_final.pdf
└── docs/
    └── index.md
```

The filenames above reflect the intended final layout. Some scripts or results may need to be copied from the experimental workspace before public release.

## Data

This repository does **not** redistribute ECG datasets. The experiments use public PhysioNet resources:

- MIT-BIH Arrhythmia Database
- MIT-BIH Supraventricular Arrhythmia Database (SVDB)

Users should download the datasets from PhysioNet and respect the corresponding licenses and citation requirements.

Expected local layout:

```text
data_mitbih/
  100.dat
  100.hea
  100.atr
  ...

data_svdb/
  800.dat
  800.hea
  800.atr
  ...
```

## Experimental components

### 1. Controlled synthetic sanity check

The controlled experiment verifies that two generators can obtain identical fidelity values while differing in generation stability and downstream utility.

### 2. MIT-BIH five-class ECG experiment

The MIT-BIH experiment evaluates synthetic ECG generation for the classes:

```text
N, S, V, F, Q
```

The experiment uses record-wise splits to avoid patient-level leakage and repeats the full process over three seeds:

```text
19, 88, 123
```

### 3. SVDB companion-cohort experiment

SVDB is used as a compact external ECG cohort. Because fusion and unknown beats are too rare for stable record-wise evaluation, the SVDB experiment is restricted to:

```text
N, S, V
```

### 4. DDPM experiments

The DDPM baseline uses a one-dimensional denoising diffusion model for class-conditional ECG beat generation. The DDPM experiments use broad denoised-output clipping during sampling for numerical stabilization, while physiological validity is evaluated with stricter fixed constraints.

Representative DDPM settings:

```text
epochs = 120
batch_size = 64
base_ch = 64
T = 300
ddim_steps = 50
guidance = 1.0
samples_per_class = 2000
x0_clip = 6
```

Physiological validity is evaluated using:

```text
max |x_t| <= 5
max |x_{t+1} - x_t| <= 4
```

## Reproducing the analysis

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run MIT-BIH experiments

```bash
python scripts/mitbih_multi_class_colab.py
```

### Run SVDB experiments

```bash
python scripts/svdb_multi_class_colab_AIM.py
```

### Run DDPM generation

Example for MIT-BIH seed 19:

```bash
python scripts/ecg_ddpm_colab.py \
  --data_dir outputs/ddpm_prepared_mitbih/seed_19 \
  --x_file X_train.npy \
  --y_file y_train.npy \
  --out_dir outputs/ddpm_mitbih_seed19 \
  --seed 19 \
  --num_classes 5 \
  --seq_len 256 \
  --epochs 120 \
  --batch_size 64 \
  --base_ch 64 \
  --T 300 \
  --ddim_steps 50 \
  --guidance 1.0 \
  --samples_per_class 2000 \
  --x0_clip 6 \
  --save_checkpoint
```

Example for SVDB seed 19:

```bash
python scripts/ecg_ddpm_colab.py \
  --data_dir outputs/ddpm_prepared_svdb/seed_19 \
  --x_file X_train.npy \
  --y_file y_train.npy \
  --out_dir outputs/ddpm_svdb_seed19 \
  --seed 19 \
  --num_classes 3 \
  --seq_len 256 \
  --epochs 120 \
  --batch_size 64 \
  --base_ch 64 \
  --T 300 \
  --ddim_steps 50 \
  --guidance 1.0 \
  --samples_per_class 2000 \
  --x0_clip 6 \
  --save_checkpoint
```

### Evaluate DDPM Trust Profiles

```bash
python scripts/evaluate_ddpm_mitbih_trust.py
python scripts/evaluate_ddpm_svdb_trust.py
```

### Generate Figure 2

```bash
python scripts/generate_figure2_mitbih_trust_profile_ddpm.py
```

## Main reported results

### MIT-BIH DDPM Trust Profile

| Metric | DDPM mean ± std |
|---|---:|
| Distributional fidelity ↓ | 1.482 ± 0.190 |
| Morphological diversity ↑ | 22.243 ± 0.150 |
| Generation stability error ↓ | 1.92e-2 ± 4.84e-3 |
| Diagnostic utility, Accuracy ↑ | 0.470 ± 0.046 |
| Diagnostic utility, Macro-F1 ↑ | 0.365 ± 0.033 |
| Class-wise reliability ↑ | 0.806 ± 0.100 |
| Physiological validity ↑ | 0.865 ± 0.004 |

### SVDB DDPM Trust Profile

| Metric | DDPM mean ± std |
|---|---:|
| Distributional fidelity ↓ | 0.872 ± 0.054 |
| Morphological diversity ↑ | 22.133 ± 0.011 |
| Generation stability error ↓ | 2.07e-2 ± 4.24e-3 |
| Diagnostic utility, Accuracy ↑ | 0.640 ± 0.058 |
| Diagnostic utility, Macro-F1 ↑ | 0.618 ± 0.041 |
| Class-wise reliability ↑ | 0.874 ± 0.047 |
| Physiological validity ↑ | 0.847 ± 0.015 |

## Interpretation

The experiments show that generator choice is deployment-dependent:

- **LSTM-VAE** is preferred under safety-critical and moderate medical augmentation constraints because of stronger perturbation stability.
- **DDPM** is preferred for exploratory filtered augmentation because of stronger diversity, macro-F1, class-wise reliability, and physiological validity.
- Under the most conservative clinical constraints, the framework can return **no admissible generator**.

This supports the paper's main conclusion: trustworthy synthetic ECG evaluation requires multi-dimensional, context-sensitive assessment rather than fidelity-centric ranking.

## Citation

If this repository supports your work, please cite the accompanying paper:

```bibtex
@article{trust_ecg_generation,
  title   = {Beyond Fidelity: Trust-Oriented Evaluation of Synthetic ECG Generators for Medical AI},
  author  = {Author information omitted for peer review},
  journal = {Under review},
  year    = {2026}
}
```

## License

This repository is released under the MIT License. Dataset licenses remain governed by the original data providers.
