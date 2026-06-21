#!/usr/bin/env python3
"""
plot_mitbih_trust_profile_radar.py
-----------------------------------
Generate the MIT-BIH Trust Profile radar plot used as Figure 2 in the paper:

    Beyond Fidelity: Trust-Oriented Evaluation of Synthetic ECG Generators
    for Medical AI

The figure summarizes reference-anchored Trust Profile scores for five
configurations:

    Baseline, Robust, TimeGAN, LSTM-VAE, DDPM

The raw mean values are those reported in the MIT-BIH results table. The
normalization is intentionally explicit so that the figure can be regenerated
without depending on notebook state.

Outputs
-------
    Figure2_MIT_BIH_Trust_Profile_DDPM_final.pdf
    Figure2_MIT_BIH_Trust_Profile_DDPM_final.png
    figure2_normalized_scores.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PDF_OUT = "Figure2_MIT_BIH_Trust_Profile_DDPM_final.pdf"
PNG_OUT = "Figure2_MIT_BIH_Trust_Profile_DDPM_final.png"
CSV_OUT = "figure2_normalized_scores.csv"

MODEL_ORDER = ["Baseline", "Robust", "TimeGAN", "LSTM-VAE", "DDPM"]

RADAR_AXES = [
    "Distributional\nFidelity",
    "Morphological\nDiversity",
    "Generation\nStability",
    "Diagnostic\nUtility",
    "Class-wise\nReliability",
    "Physiological\nValidity",
]

RAW = pd.DataFrame([
    {
        "Model": "Baseline", "Fidelity": 2.434, "Diversity": 18.745,
        "Robustness": 3.1e-4, "Accuracy": 0.746, "MacroF1": 0.435,
        "Reliability": 0.591, "Validity": 0.604,
    },
    {
        "Model": "Robust", "Fidelity": 1.682, "Diversity": 16.941,
        "Robustness": 1.6e-5, "Accuracy": 0.734, "MacroF1": 0.464,
        "Reliability": 0.626, "Validity": 0.689,
    },
    {
        "Model": "TimeGAN", "Fidelity": 1.355, "Diversity": 15.833,
        "Robustness": 1.2e-2, "Accuracy": 0.528, "MacroF1": 0.206,
        "Reliability": 0.706, "Validity": 0.460,
    },
    {
        "Model": "LSTM-VAE", "Fidelity": 1.284, "Diversity": 12.747,
        "Robustness": 2.5e-3, "Accuracy": 0.528, "MacroF1": 0.192,
        "Reliability": 0.703, "Validity": 0.815,
    },
    {
        "Model": "DDPM", "Fidelity": 1.482, "Diversity": 22.243,
        "Robustness": 1.9e-2, "Accuracy": 0.470, "MacroF1": 0.365,
        "Reliability": 0.806, "Validity": 0.865,
    },
])

# Reference anchors used for the radar plot. Larger normalized values are better.
FIDELITY_BAD = 3.5
FIDELITY_GOOD = 0.0
ROBUSTNESS_BAD = 0.05
ROBUSTNESS_GOOD = 0.0
ACCURACY_BAD = 0.20
ACCURACY_GOOD = 1.0
MACROF1_BAD = 0.0
MACROF1_GOOD = 1.0
DIVERSITY_BAD = 0.0
DIVERSITY_GOOD = float(RAW["Diversity"].max())


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def lower_is_better(v, good, bad):
    return clip01((bad - v) / (bad - good))


def higher_is_better(v, bad, good):
    return clip01((v - bad) / (good - bad))


def compute_scores(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw[["Model"]].copy()
    out["Distributional Fidelity"] = lower_is_better(
        raw["Fidelity"], FIDELITY_GOOD, FIDELITY_BAD
    )
    out["Morphological Diversity"] = higher_is_better(
        raw["Diversity"], DIVERSITY_BAD, DIVERSITY_GOOD
    )
    out["Generation Stability"] = lower_is_better(
        raw["Robustness"], ROBUSTNESS_GOOD, ROBUSTNESS_BAD
    )

    accuracy_score = higher_is_better(raw["Accuracy"], ACCURACY_BAD, ACCURACY_GOOD)
    macrof1_score = higher_is_better(raw["MacroF1"], MACROF1_BAD, MACROF1_GOOD)
    out["Diagnostic Utility"] = 0.5 * accuracy_score + 0.5 * macrof1_score

    out["Class-wise Reliability"] = raw["Reliability"].clip(0, 1)
    out["Physiological Validity"] = raw["Validity"].clip(0, 1)
    return out


def plot_radar(scores: pd.DataFrame) -> None:
    scores = scores.set_index("Model").loc[MODEL_ORDER].reset_index()

    n_axes = len(RADAR_AXES)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(9, 8.2), subplot_kw=dict(polar=True))
    ax.set_title("MIT-BIH Trust Profile", fontsize=22, pad=30)

    # First axis at the right side, counterclockwise order.
    ax.set_theta_offset(0)
    ax.set_theta_direction(1)

    # Hide automatic polar tick labels and place labels manually.
    ax.set_xticks(angles)
    ax.set_xticklabels([])

    base_radius = 1.14
    radius_offsets = {
        0: 0.035,  # Distributional Fidelity
        1: 0.000,
        2: 0.000,
        3: 0.025,  # Diagnostic Utility
        4: 0.000,
        5: 0.000,
    }
    angle_offsets = {
        0: 0.000,
        1: 0.000,
        2: 0.000,
        3: 0.000,
        4: 0.000,
        5: 0.000,
    }
    alignments = {
        0: ("left", "center"),
        1: ("center", "center"),
        2: ("center", "center"),
        3: ("right", "center"),
        4: ("center", "center"),
        5: ("center", "center"),
    }

    for i, label in enumerate(RADAR_AXES):
        ha, va = alignments[i]
        r = base_radius + radius_offsets[i]
        theta = angles[i] + angle_offsets[i]
        ax.text(theta, r, label, ha=ha, va=va, fontsize=15)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.grid(True, alpha=0.55)

    for model in MODEL_ORDER:
        row = scores[scores["Model"] == model].iloc[0]
        vals = [float(row[col.replace("\n", " ")]) for col in RADAR_AXES]
        vals_closed = vals + vals[:1]
        ax.plot(closed_angles, vals_closed, linewidth=2.2, label=model)
        ax.fill(closed_angles, vals_closed, alpha=0.10)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.08, 1.05),
        fontsize=12,
        frameon=True,
    )

    plt.tight_layout()
    fig.savefig(PDF_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    scores = compute_scores(RAW)
    scores.to_csv(CSV_OUT, index=False)
    plot_radar(scores)
    print(f"Saved: {PDF_OUT}")
    print(f"Saved: {PNG_OUT}")
    print(f"Saved: {CSV_OUT}")


if __name__ == "__main__":
    main()
