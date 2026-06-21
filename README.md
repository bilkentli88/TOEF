# Beyond Fidelity: Trust-Oriented Evaluation of Synthetic ECG Generators for Medical AI

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Reproducibility](https://img.shields.io/badge/focus-reproducibility-informational)

## Overview

This repository contains the code accompanying the paper:

**Beyond Fidelity: Trust-Oriented Evaluation of Synthetic ECG Generators for Medical AI**

The paper introduces a **Trust-Oriented Evaluation Framework** for synthetic ECG and medical time-series generation. Instead of reducing generator quality to a single fidelity or utility score, the framework represents generator behavior through a multi-dimensional **Trust Profile**.

The Trust Profile contains six dimensions:

1. **Distributional fidelity**
2. **Morphological diversity**
3. **Generation stability**
4. **Diagnostic utility**
5. **Class-wise reliability**
6. **Physiological validity**

The empirical study evaluates adversarial, variational, and diffusion-based synthetic ECG generators on two ECG cohorts:

- **MIT-BIH Arrhythmia Database** using the five-class grouping `{N,S,V,F,Q}`
- **MIT-BIH Supraventricular Arrhythmia Database (SVDB)** using the sufficiently represented classes `{N,S,V}`

The evaluated deep generative models are:

- **TimeGAN** — adversarial time-series generator
- **LSTM-VAE** — variational recurrent generator
- **DDPM** — denoising diffusion probabilistic model

Two additional proxy generators, **Baseline** and **Robust**, are included as diagnostic references for interpreting stability and validity behavior. They are not treated as deployable deep generators in the risk-aware selection analysis.

## Main claim

The repository supports the paper's central claim:

> Synthetic ECG generators should not be ranked using fidelity or utility alone. Generator suitability depends on deployment context and on trade-offs among fidelity, diversity, stability, diagnostic utility, class-wise reliability, and physiological validity.

The experiments show that no generator is uniformly preferable:

- **DDPM** achieves strong diversity, macro-F1, class-wise reliability, and physiological validity.
- **LSTM-VAE** remains preferable under stability-constrained safety settings.
- **TimeGAN** can appear competitive under some fidelity or accuracy views but performs poorly under physiological-validity constraints.
- Under the most conservative clinical constraints, the framework can return **no admissible generator**.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── run_mitbih_timegan_lstmvae.py
│   ├── run_svdb_timegan_lstmvae.py
│   ├── train_generate_ddpm_ecg.py
│   ├── evaluate_ddpm_mitbih_trust.py
│   ├── evaluate_ddpm_svdb_trust.py
│   └── plot_mitbih_trust_profile_radar.py
├── figures/
│   ├── Figure1_Framework_AIM.pdf
│   └── Figure2_MIT_BIH_Trust_Profile_DDPM_final.pdf
└── results/
    ├── mitbih_trust_profiles_summary.csv
    ├── svdb_trust_profiles_summary.csv
    ├── ddpm_trust_profiles_summary.csv
    └── ddpm_svdb_trust_profiles_summary.csv
```

Only the `src/` folder is required to run the scripts. The `figures/` and `results/` folders are recommended for organizing generated outputs.

## Data availability

This repository does **not** redistribute ECG datasets.

The experiments use public PhysioNet datasets:

- MIT-BIH Arrhythmia Database
- MIT-BIH Supraventricular Arrhythmia Database (SVDB)

Users should download the datasets from PhysioNet and follow the corresponding license and citation requirements.

A typical local layout is:

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

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The scripts were developed for Python 3.10+.

## Scripts

### `run_mitbih_timegan_lstmvae.py`

Runs the MIT-BIH Trust Profile experiment for:

- Baseline proxy
- Robust proxy
- TimeGAN
- LSTM-VAE

This script produces the raw and summary Trust Profile outputs for the non-diffusion generators in the five-class MIT-BIH setting.

### `run_svdb_timegan_lstmvae.py`

Runs the SVDB companion-cohort experiment for:

- TimeGAN
- LSTM-VAE

SVDB is restricted to `{N,S,V}` because the `F` and `Q` classes are too rare for stable record-wise evaluation.

### `train_generate_ddpm_ecg.py`

Trains and samples a class-conditional one-dimensional DDPM for ECG beat generation.

Representative DDPM settings used in the paper:

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

The broad denoised-output clipping value is used for numerical stabilization during sampling. Physiological validity is still evaluated using stricter fixed validity constraints.

### `evaluate_ddpm_mitbih_trust.py`

Computes the DDPM Trust Profile metrics for the MIT-BIH experiment.

The script expects generated DDPM samples for seeds:

```text
19, 88, 123
```

### `evaluate_ddpm_svdb_trust.py`

Computes the DDPM Trust Profile metrics for the SVDB experiment.

The script expects generated DDPM samples for the same seeds:

```text
19, 88, 123
```

### `plot_mitbih_trust_profile_radar.py`

Generates the reference-anchored MIT-BIH Trust Profile radar plot used as Figure 2 in the paper.

Outputs:

```text
Figure2_MIT_BIH_Trust_Profile_DDPM_final.pdf
Figure2_MIT_BIH_Trust_Profile_DDPM_final.png
figure2_normalized_scores.csv
```

## Example DDPM command

Example MIT-BIH run for seed 19:

```bash
python src/train_generate_ddpm_ecg.py \
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

Example SVDB run for seed 19:

```bash
python src/train_generate_ddpm_ecg.py \
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

## Physiological-validity constraints

Generated ECG beats are evaluated in the normalized representation. A generated beat is counted as physiologically valid if it satisfies both constraints:

```text
max |x_t| <= 5
max |x_{t+1} - x_t| <= 4
```

These fixed constraints are intended to detect obvious amplitude and slew-rate violations. They do not replace expert clinical validation.

## Main reported DDPM results

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

## Risk-aware interpretation

The framework separates hard feasibility constraints from preference-based model selection.

For the MIT-BIH experiment:

| Deployment context | Selected outcome |
|---|---|
| Safety-critical synthetic ECG augmentation | LSTM-VAE |
| Moderate medical augmentation | LSTM-VAE |
| Exploratory augmentation with filtering | DDPM |
| Conservative clinical pipeline | No admissible generator |

This illustrates the intended use of the Trust Profile: different deployment contexts can produce different admissible choices.

## Reproducibility notes

- The experiments use three random seeds: `19`, `88`, and `123`.
- MIT-BIH uses a record-wise split to avoid patient-level leakage.
- SVDB is used as a compact external companion-cohort check.
- DDPM results are evaluated separately and then integrated into the Trust Profile analysis.
- The scripts are designed for research reproducibility rather than clinical deployment.

## Citation

If this repository supports your work, please cite the accompanying paper:

```bibtex
@article{trust_ecg_generation,
  title   = {Beyond Fidelity: Trust-Oriented Evaluation of Synthetic ECG Generators for Medical AI},
  author  = {Aykut T. Altay},
  journal = {Under review},
  year    = {2026}
}
```

## License

This repository is released under the MIT License. Dataset licenses remain governed by the original data providers.

## Disclaimer

This code is provided for research reproducibility. It is not intended for clinical decision-making, diagnostic use, or deployment in medical workflows.
