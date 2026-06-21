"""
evaluate_ddpm_mitbih_trust.py
------------------------------
Compute Trust Profile metrics for DDPM synthetic MIT-BIH outputs using the same
metric conventions as the existing MIT-BIH Trust Profile script.

Expected Colab folder structure:
  /content/drive/MyDrive/EAAI_New/
      ecg_ddpm_colab.py
      outputs/
          ddpm_prepared_mitbih/
              seed_19/X_test.npy, y_test.npy
              seed_88/X_test.npy, y_test.npy
              seed_123/X_test.npy, y_test.npy
          ddpm_mitbih_seed19/ddpm_X_syn_seed19.npy, ddpm_y_syn_seed19.npy, ddpm_seed19.pt
          ddpm_mitbih_seed88/ddpm_X_syn_seed88.npy, ddpm_y_syn_seed88.npy, ddpm_seed88.pt
          ddpm_mitbih_seed123/ddpm_X_syn_seed123.npy, ddpm_y_syn_seed123.npy, ddpm_seed123.pt

Run:
  %cd /content/drive/MyDrive/EAAI_New
  !python evaluate_ddpm_mitbih_trust.py

Outputs:
  outputs/ddpm_mitbih_trust_eval/ddpm_trust_profiles_raw.csv
  outputs/ddpm_mitbih_trust_eval/ddpm_trust_profiles_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

# The DDPM class is expected in the same project root.
from ecg_ddpm_colab import ECGDiffusion


CLASSES = [0, 1, 2, 3, 4]


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def physiological_validity(X: np.ndarray, amax: float = 5.0, dmax: float = 4.0) -> float:
    amp_ok = np.max(np.abs(X), axis=1) <= amax
    grad_ok = np.max(np.abs(np.diff(X, axis=1)), axis=1) <= dmax
    return float(np.mean(amp_ok & grad_ok))


def diversity_score(X: np.ndarray, seed: int, sample: int = 500, pairs: int = 2000) -> float:
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    sub = X[idx].reshape(len(idx), -1)
    i1 = rng.integers(0, len(sub), size=pairs)
    i2 = rng.integers(0, len(sub), size=pairs)
    return float(np.mean(np.linalg.norm(sub[i1] - sub[i2], axis=1)))


def rf_utility_fidelity_reliability(
    X_gen: np.ndarray,
    y_gen: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    rf_estimators: int = 50,
) -> Dict[str, float]:
    clf = RandomForestClassifier(n_estimators=rf_estimators, n_jobs=-1, random_state=seed)
    clf.fit(X_gen.reshape(len(X_gen), -1), y_gen)

    probs = clf.predict_proba(X_test.reshape(len(X_test), -1))
    preds = clf.predict(X_test.reshape(len(X_test), -1))

    # Ensure probability matrix has all five columns even if a class is absent.
    full_probs = np.zeros((len(y_test), 5), dtype=float)
    for i, c in enumerate(clf.classes_):
        if 0 <= int(c) < 5:
            full_probs[:, int(c)] = probs[:, i]

    # Avoid exact zeros in log-loss when a class was not predicted/probability collapsed.
    eps = 1e-12
    full_probs = np.clip(full_probs, eps, 1.0)
    full_probs = full_probs / full_probs.sum(axis=1, keepdims=True)

    recalls = recall_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)

    return {
        "Fidelity": float(log_loss(y_test, full_probs, labels=CLASSES)),
        "Utility": float(accuracy_score(y_test, preds)),
        "MacroF1": float(f1_score(y_test, preds, average="macro", labels=CLASSES, zero_division=0)),
        "Fairness": float(1.0 - np.std(recalls)),  # class-wise reliability
    }


def load_ddpm_from_checkpoint(ckpt_path: Path, device: str) -> ECGDiffusion:
    ckpt = torch.load(ckpt_path, map_location=device)
    extra = ckpt.get("extra", {}) or {}

    seq_len = int(ckpt.get("seq_len", extra.get("seq_len", 256)))
    num_classes = int(ckpt.get("num_classes", extra.get("num_classes", 5)))
    T = int(ckpt.get("T", extra.get("T", 300)))
    p_uncond = float(ckpt.get("p_uncond", extra.get("p_uncond", 0.1)))
    base_ch = int(extra.get("base_ch", 64))
    seed = int(extra.get("seed", 0))

    gen = ECGDiffusion(
        seq_len=seq_len,
        num_classes=num_classes,
        T=T,
        base_ch=base_ch,
        p_uncond=p_uncond,
        device=device,
        seed=seed,
    )
    gen.model.load_state_dict(ckpt["model_state"])
    gen.ema_state = ckpt.get("ema_state", None)
    return gen


@torch.no_grad()
def ddim_from_initial(
    gen: ECGDiffusion,
    x_init: torch.Tensor,
    y: torch.Tensor,
    ddim_steps: int = 50,
    guidance: float = 1.0,
    x0_clip: float = 6.0,
    use_ema: bool = True,
) -> torch.Tensor:
    backup = None
    if use_ema and gen.ema_state is not None:
        backup = {k: v.detach().clone() for k, v in gen.model.state_dict().items()}
        gen.model.load_state_dict(gen.ema_state)

    gen.model.eval()
    device = gen.device
    x = x_init.to(device)
    y = y.to(device)
    B = x.shape[0]

    step_idx = torch.linspace(gen.T - 1, 0, ddim_steps, device=device).long()
    for i, t_cur_tensor in enumerate(step_idx):
        t_cur = int(t_cur_tensor.item())
        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
        ac_t = gen.acp[t_cur]

        if i + 1 < len(step_idx):
            t_next = int(step_idx[i + 1].item())
            ac_next = gen.acp[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
        else:
            ac_next = torch.tensor(1.0, device=device)

        eps_c = gen.model(x, t_batch, y)
        if guidance is not None and guidance != 1.0:
            y_null = torch.full_like(y, gen.model.null_idx)
            eps_u = gen.model(x, t_batch, y_null)
            eps = eps_u + guidance * (eps_c - eps_u)
        else:
            eps = eps_c

        x0 = (x - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()
        if x0_clip is not None and x0_clip > 0:
            x0 = x0.clamp(-x0_clip, x0_clip)
        x = ac_next.sqrt() * x0 + (1 - ac_next).sqrt() * eps

    if backup is not None:
        gen.model.load_state_dict(backup)

    return x


def ddpm_robustness(
    ckpt_path: Path,
    seed: int,
    device: str,
    n: int = 200,
    latent_delta: float = 0.05,
    ddim_steps: int = 50,
    guidance: float = 1.0,
    x0_clip: float = 6.0,
    batch: int = 50,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = load_ddpm_from_checkpoint(ckpt_path, device=device)
    device_t = gen.device

    mses = []
    done = 0
    while done < n:
        b = min(batch, n - done)
        # balanced labels over classes
        labels = torch.tensor([(done + i) % gen.num_classes for i in range(b)], device=device_t, dtype=torch.long)
        z = torch.randn(b, gen.seq_len, device=device_t)
        delta = torch.randn_like(z) * latent_delta

        x1 = ddim_from_initial(gen, z, labels, ddim_steps=ddim_steps, guidance=guidance, x0_clip=x0_clip)
        x2 = ddim_from_initial(gen, z + delta, labels, ddim_steps=ddim_steps, guidance=guidance, x0_clip=x0_clip)
        mses.append(torch.mean((x1 - x2) ** 2).item())
        done += b

    return float(np.mean(mses))


def evaluate_seed(args, seed: int) -> Dict[str, float]:
    prepared_dir = Path(args.prepared_root) / f"seed_{seed}"
    gen_dir = Path(args.generated_template.format(seed=seed))

    X_test = np.load(prepared_dir / "X_test.npy").astype(np.float32)
    y_test = np.load(prepared_dir / "y_test.npy").astype(np.int64)
    X_gen = np.load(gen_dir / f"ddpm_X_syn_seed{seed}.npy").astype(np.float32)
    y_gen = np.load(gen_dir / f"ddpm_y_syn_seed{seed}.npy").astype(np.int64)
    ckpt_path = gen_dir / f"ddpm_seed{seed}.pt"

    print(f"\nSeed {seed}")
    print("  X_gen:", X_gen.shape, "y_gen:", y_gen.shape)
    print("  X_test:", X_test.shape, "y_test:", y_test.shape)

    row = {
        "Model": "DDPM",
        "Seed": seed,
        "Diversity": diversity_score(X_gen, seed=seed, sample=args.diversity_sample, pairs=args.diversity_pairs),
        "Safety": physiological_validity(X_gen, amax=args.amax, dmax=args.dmax),
    }

    row.update(rf_utility_fidelity_reliability(
        X_gen, y_gen, X_test, y_test, seed=seed, rf_estimators=args.rf_estimators
    ))

    if ckpt_path.exists():
        row["Robustness"] = ddpm_robustness(
            ckpt_path=ckpt_path,
            seed=seed,
            device=args.device,
            n=args.robust_n,
            latent_delta=args.latent_delta,
            ddim_steps=args.ddim_steps,
            guidance=args.guidance,
            x0_clip=args.x0_clip,
            batch=args.robust_batch,
        )
    else:
        print(f"  WARNING: checkpoint not found: {ckpt_path}. Robustness set to NaN.")
        row["Robustness"] = float("nan")

    print("  metrics:", {k: row[k] for k in ["Fidelity", "Diversity", "Robustness", "Utility", "MacroF1", "Fairness", "Safety"]})
    return row


def summarize(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    metrics = ["Fidelity", "Diversity", "Robustness", "Utility", "MacroF1", "Fairness", "Safety"]
    summary = {"Model": "DDPM", "NSeeds": len(rows)}
    for m in metrics:
        vals = np.array([float(r[m]) for r in rows], dtype=float)
        summary[f"{m}_mean"] = float(np.nanmean(vals))
        summary[f"{m}_std"] = float(np.nanstd(vals))  # population std, matching previous script convention
    return [summary]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default="/content/drive/MyDrive/EAAI_New")
    parser.add_argument("--prepared_root", type=str, default=None)
    parser.add_argument("--generated_template", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="19,88,123")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--diversity_sample", type=int, default=500)
    parser.add_argument("--diversity_pairs", type=int, default=2000)
    parser.add_argument("--rf_estimators", type=int, default=50)
    parser.add_argument("--amax", type=float, default=5.0)
    parser.add_argument("--dmax", type=float, default=4.0)

    parser.add_argument("--robust_n", type=int, default=200)
    parser.add_argument("--robust_batch", type=int, default=50)
    parser.add_argument("--latent_delta", type=float, default=0.05)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--x0_clip", type=float, default=6.0)

    args = parser.parse_args()

    project_root = Path(args.project_root)
    if args.prepared_root is None:
        args.prepared_root = str(project_root / "outputs" / "ddpm_prepared_mitbih")
    if args.generated_template is None:
        args.generated_template = str(project_root / "outputs" / "ddpm_mitbih_seed{seed}")
    if args.out_dir is None:
        args.out_dir = str(project_root / "outputs" / "ddpm_mitbih_trust_eval")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print("device:", args.device)
    print("seeds:", seeds)
    print("prepared_root:", args.prepared_root)
    print("generated_template:", args.generated_template)

    rows = [evaluate_seed(args, seed) for seed in seeds]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_fields = ["Model", "Seed", "Fidelity", "Diversity", "Robustness", "Utility", "MacroF1", "Fairness", "Safety"]
    write_csv(out_dir / "ddpm_trust_profiles_raw.csv", rows, raw_fields)

    summary_rows = summarize(rows)
    summary_fields = list(summary_rows[0].keys())
    write_csv(out_dir / "ddpm_trust_profiles_summary.csv", summary_rows, summary_fields)

    print("\nSaved:")
    print(" ", out_dir / "ddpm_trust_profiles_raw.csv")
    print(" ", out_dir / "ddpm_trust_profiles_summary.csv")
    print("\nSummary:")
    print(summary_rows[0])


if __name__ == "__main__":
    main()
