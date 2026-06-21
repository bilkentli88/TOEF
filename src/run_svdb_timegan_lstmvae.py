# =============================================================================
# FIXED MASTER SCRIPT: Trust-Oriented Evaluation (SVDB external companion cohort)
# NOW INCLUDES: Baseline, Robust Proxy, TimeGAN, and LSTM-VAE
# =============================================================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, asdict
from typing import Dict, List
from pathlib import Path
import csv
import json
import platform
import sys
import time
import shutil
import warnings
from math import gcd

# --- 1. SETUP & INSTALLATION ---
try:
    import wfdb
except ImportError:
    print("Installing wfdb...")
    os.system('pip install wfdb')
    import wfdb

from sklearn.ensemble import RandomForestClassifier

try:
    from scipy.signal import resample_poly
except ImportError:
    print("Installing scipy...")
    os.system("pip install scipy")
    from scipy.signal import resample_poly
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    recall_score
)

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


# =============================================================================
# 2. CONFIGURATION
# =============================================================================
@dataclass
class Config:
    # EXPERIMENT SETTINGS
    seeds: List[int] = (19, 88, 123)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # DATASET
    data_dir: str = "data_svdb"
    db_name: str = "svdb"
    beat_len: int = 256  # Fixed beat length
    pre_r: int = 90  # Samples before R-peak
    post_r: int = 166  # Samples after R-peak
    channel: int = 0  # SVDB uses ECG1/ECG2; channel 0 = ECG1
    source_fs: int = 128
    target_fs: int = 360  # Resample to MIT-BIH-compatible rate
    train_frac: float = 0.7
    per_class_cap: int = 200  # Max samples per class per record

    # GENERATOR SETTINGS
    baseline_noise: float = 0.35
    robust_noise: float = 0.08

    # TimeGAN Settings
    tg_hidden: int = 24
    tg_layers: int = 3
    tg_batch: int = 64
    tg_lr: float = 0.001
    tg_epochs: int = 150

    # LSTM-VAE Settings (New)
    vae_hidden: int = 64
    vae_latent: int = 20
    vae_layers: int = 1
    vae_batch: int = 64
    gen_batch: int = 256  # Batch synthetic generation to avoid CUDA OOM
    vae_lr: float = 0.001
    vae_epochs: int = 100 # VAE converges faster than GAN

    # EVALUATION METRICS
    latent_delta: float = 0.05
    diversity_sample: int = 500
    diversity_pairs: int = 2000
    rf_estimators: int = 50

    # OUTPUT / REPRODUCIBILITY
    output_root: str = "outputs"
    run_name: str = "aim_svdb_nsv_rerun"
    save_after_each_seed: bool = True

    # Safety Constraints
    safety_min: float = -5.0
    safety_max: float = 5.0
    safety_slope: float = 4.0


cfg = Config()


def set_seeds(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make PyTorch/CUDA behavior as reproducible as possible.
    # Some GPU kernels may still be nondeterministic depending on the runtime,
    # but these settings substantially reduce avoidable variation.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


print(f"Running on {cfg.device} with seeds: {cfg.seeds}")

# =============================================================================
# 3. DATA LOADING
# =============================================================================
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q", "?": "Q"
}
CLASSES = ["N", "S", "V"]  # SVDB external check uses sufficiently represented classes only


def load_svdb_data(seed):
    """Downloads SVDB, resamples to 360 Hz, and extracts N/S/V beats."""
    set_seeds(seed)

    # Download
    os.makedirs(cfg.data_dir, exist_ok=True)
    try:
        if not os.path.exists(os.path.join(cfg.data_dir, '800.hea')):
            print("Downloading SVDB Database...")
            wfdb.dl_database(cfg.db_name, dl_dir=cfg.data_dir)
    except:
        pass

    # Get records
    recs = [f.split('.')[0] for f in os.listdir(cfg.data_dir) if f.endswith('.hea')]
    recs = [r for r in recs if r.isdigit()]
    random.shuffle(recs)

    # Split
    split_idx = int(len(recs) * cfg.train_frac)
    train_recs = recs[:split_idx]
    test_recs = recs[split_idx:]

    def resample_signal_and_annotations(sig, ann_samples, fs):
        """Resamples a signal to cfg.target_fs and scales annotation locations."""
        fs = int(round(fs))
        if fs == cfg.target_fs:
            return sig, ann_samples.astype(int)
        g = gcd(cfg.target_fs, fs)
        up = cfg.target_fs // g
        down = fs // g
        sig_resampled = resample_poly(sig, up, down)
        ann_resampled = np.round(ann_samples.astype(float) * cfg.target_fs / fs).astype(int)
        return sig_resampled, ann_resampled

    def extract(record_list):
        X, y = [], []
        split_counts = {c: 0 for c in CLASSES}

        for rec in record_list:
            try:
                path = os.path.join(cfg.data_dir, rec)
                ann = wfdb.rdann(path, 'atr')
                record = wfdb.rdrecord(path)
                sig = record.p_signal[:, cfg.channel]
                sig, ann_samples = resample_signal_and_annotations(sig, np.array(ann.sample), record.fs)
            except Exception as e:
                print(f"    Skipping record {rec}: {e}")
                continue

            cnt = {c: 0 for c in CLASSES}
            for sample, sym in zip(ann_samples, ann.symbol):
                if sym not in AAMI_MAP:
                    continue
                label = AAMI_MAP[sym]
                if label not in CLASSES:
                    # SVDB has too few F and Q beats for a reliable record-wise experiment.
                    continue
                if cnt[label] >= cfg.per_class_cap:
                    continue

                start, end = sample - cfg.pre_r, sample + cfg.post_r
                if start >= 0 and end <= len(sig):
                    beat = sig[start:end]
                    beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-6)
                    if len(beat) == cfg.beat_len:
                        X.append(beat)
                        y.append(CLASSES.index(label))
                        cnt[label] += 1
                        split_counts[label] += 1

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=int)
        print("    Extracted counts:", {CLASSES[i]: int(np.sum(y == i)) for i in range(len(CLASSES))})
        return X, y

    print("  Extracting Train...")
    X_train, y_train = extract(train_recs)
    print("  Extracting Test...")
    X_test, y_test = extract(test_recs)

    return np.expand_dims(X_train, -1), y_train, np.expand_dims(X_test, -1), y_test, train_recs, test_recs


# =============================================================================
# 4A. FIXED TIMEGAN IMPLEMENTATION
# =============================================================================
class TimeGAN_Net(nn.Module):
    def __init__(self, d_in, d_out, hidden, layers, output_sig=False):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden, layers, batch_first=True)
        self.lin = nn.Linear(hidden, d_out)
        self.output_sig = output_sig
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        o, _ = self.rnn(x)
        o = self.lin(o)
        if self.output_sig:
            o = self.sigmoid(o)
        return o


class TimeGAN:
    def __init__(self):
        self.E = TimeGAN_Net(1, cfg.tg_hidden, cfg.tg_hidden, cfg.tg_layers).to(cfg.device)
        self.R = TimeGAN_Net(cfg.tg_hidden, 1, cfg.tg_hidden, cfg.tg_layers).to(cfg.device)
        self.G = TimeGAN_Net(1, cfg.tg_hidden, cfg.tg_hidden, cfg.tg_layers).to(cfg.device)
        self.S = TimeGAN_Net(cfg.tg_hidden, cfg.tg_hidden, cfg.tg_hidden, cfg.tg_layers).to(cfg.device)
        self.D = TimeGAN_Net(cfg.tg_hidden, 1, cfg.tg_hidden, cfg.tg_layers, output_sig=True).to(cfg.device)

        self.opt_e = optim.Adam(list(self.E.parameters()) + list(self.R.parameters()), lr=cfg.tg_lr)
        self.opt_g = optim.Adam(list(self.G.parameters()) + list(self.S.parameters()), lr=cfg.tg_lr)
        self.opt_d = optim.Adam(self.D.parameters(), lr=cfg.tg_lr)

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def train(self, X_train):
        dataset = TensorDataset(torch.FloatTensor(X_train))
        loader = DataLoader(dataset, batch_size=cfg.tg_batch, shuffle=True)

        ae_epoch = int(cfg.tg_epochs * 0.2)
        sup_epoch = int(cfg.tg_epochs * 0.2)
        joint_epoch = cfg.tg_epochs

        # 1. Embedding
        for _ in range(ae_epoch):
            for batch in loader:
                x = batch[0].to(cfg.device)
                l = self.mse(x, self.R(self.E(x)))
                self.opt_e.zero_grad(); l.backward(); self.opt_e.step()

        # 2. Supervisor
        for _ in range(sup_epoch):
            for batch in loader:
                x = batch[0].to(cfg.device)
                h = self.E(x).detach()
                l = self.mse(h[:, 1:], self.S(h)[:, :-1])
                self.opt_g.zero_grad(); l.backward(); self.opt_g.step()

        # 3. Joint
        for _ in range(joint_epoch):
            for batch in loader:
                x = batch[0].to(cfg.device)
                b_size = x.size(0)

                # Generator
                z = torch.randn(b_size, cfg.beat_len, 1).to(cfg.device)
                e_hat = self.G(z)
                h_hat = self.S(e_hat)
                x_hat = self.R(h_hat)
                y_fake = self.D(h_hat)
                h = self.E(x).detach()
                loss_g_adv = self.bce(y_fake, torch.ones_like(y_fake))
                loss_s = self.mse(h[:, 1:], self.S(h)[:, :-1])
                loss_mom = torch.mean(torch.abs(torch.mean(x_hat, 0) - torch.mean(x, 0))) + \
                           torch.mean(torch.abs(torch.std(x_hat, 0) - torch.std(x, 0)))
                loss_g = loss_g_adv + 10 * loss_s + 100 * loss_mom
                self.opt_g.zero_grad(); loss_g.backward(); self.opt_g.step()

                # Discriminator
                y_real = self.D(h)
                h_hat_d = self.S(self.G(z)).detach()
                y_fake_d = self.D(h_hat_d)
                loss_d = self.bce(y_real, torch.ones_like(y_real)) + \
                         self.bce(y_fake_d, torch.zeros_like(y_fake_d))
                self.opt_d.zero_grad(); loss_d.backward(); self.opt_d.step()

    def generate(self, n, batch_size=None):
        """Generates samples in mini-batches to avoid GPU memory spikes."""
        self.G.eval(); self.S.eval(); self.R.eval()
        batch_size = batch_size or cfg.gen_batch
        outs = []
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                b = min(batch_size, n - start)
                z = torch.randn(b, cfg.beat_len, 1, device=cfg.device)
                x = self.R(self.S(self.G(z))).detach().cpu().numpy()
                outs.append(x)
                del z, x
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return np.concatenate(outs, axis=0) if outs else np.empty((0, cfg.beat_len, 1), dtype=np.float32)

# =============================================================================
# 4B. LSTM-VAE IMPLEMENTATION (NEW)
# =============================================================================
class LSTM_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(LSTM_VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.final_layer = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1] # Take last layer
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        z = self.reparameterize(mu, logvar)

        # Decode
        # Expand z to sequence length
        seq_len = x.size(1)
        # We repeat the latent vector to be the input for every timestep
        # Alternatively, we could use it as hidden state initialization.
        # Here: Project z back to hidden and repeat.
        d_in = self.decoder_input(z).unsqueeze(1).repeat(1, seq_len, 1)

        out, _ = self.decoder_lstm(d_in)
        recon_x = self.final_layer(out)

        return recon_x, mu, logvar

class VAE_Trainer:
    def __init__(self):
        self.model = LSTM_VAE(1, cfg.vae_hidden, cfg.vae_latent, cfg.vae_layers).to(cfg.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.vae_lr)

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        # KL Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD

    def train(self, X_train):
        self.model.train()
        dataset = TensorDataset(torch.FloatTensor(X_train))
        loader = DataLoader(dataset, batch_size=cfg.vae_batch, shuffle=True)

        for _ in range(cfg.vae_epochs):
            for batch in loader:
                x = batch[0].to(cfg.device)
                recon_x, mu, logvar = self.model(x)
                loss = self.loss_function(recon_x, x, mu, logvar)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def generate(self, n, batch_size=None):
        """Generates samples in mini-batches to avoid CUDA OOM during LSTM decoding."""
        self.model.eval()
        batch_size = batch_size or cfg.gen_batch
        outs = []
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                b = min(batch_size, n - start)
                z = torch.randn(b, cfg.vae_latent, device=cfg.device)
                d_in = self.model.decoder_input(z).unsqueeze(1).repeat(1, cfg.beat_len, 1)
                out, _ = self.model.decoder_lstm(d_in)
                recon_x = self.model.final_layer(out).detach().cpu().numpy()
                outs.append(recon_x)
                del z, d_in, out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return np.concatenate(outs, axis=0) if outs else np.empty((0, cfg.beat_len, 1), dtype=np.float32)

# =============================================================================
# 5. METRIC FUNCTIONS
# =============================================================================
def compute_metrics(X_gen, y_gen, X_test, y_test, seed=None):
    metrics = {}

    # A. Diversity
    idx = np.random.choice(len(X_gen), min(len(X_gen), cfg.diversity_sample), replace=False)
    sub = X_gen[idx].reshape(len(idx), -1)
    i1 = np.random.randint(0, len(sub), cfg.diversity_pairs)
    i2 = np.random.randint(0, len(sub), cfg.diversity_pairs)
    metrics['Diversity'] = np.mean(np.linalg.norm(sub[i1] - sub[i2], axis=1))

    # B. Safety
    valid = np.mean([1 if np.all(np.abs(x) <= 5) and np.all(np.abs(np.diff(x, axis=0)) <= 4) else 0 for x in X_gen])
    metrics['Safety'] = valid

    # C. Utility/Fairness/Fidelity
    clf = RandomForestClassifier(n_estimators=cfg.rf_estimators, n_jobs=-1, random_state=seed)
    clf.fit(X_gen.reshape(len(X_gen), -1), y_gen)
    probs = clf.predict_proba(X_test.reshape(len(X_test), -1))
    preds = np.argmax(probs, axis=1)

    n_classes = len(CLASSES)
    full_probs = np.zeros((len(y_test), n_classes))
    for i, c in enumerate(clf.classes_):
        if c < n_classes:
            full_probs[:, c] = probs[:, i]

    metrics['Fidelity'] = log_loss(y_test, full_probs, labels=list(range(n_classes)))
    metrics['Accuracy'] = accuracy_score(y_test, preds)
    metrics['F1'] = f1_score(y_test, preds, average='macro')
    metrics['Fairness'] = 1.0 - np.std(recall_score(y_test, preds, average=None, labels=list(range(len(CLASSES))), zero_division=0))

    return metrics


def compute_robustness(X_seed, noise, model_type='proxy', model=None):
    if model_type == 'proxy':
        z = np.random.normal(0, 1, X_seed.shape)
        d = np.random.normal(0, cfg.latent_delta, X_seed.shape)
        return np.mean(((X_seed + noise * z) - (X_seed + noise * (z + d))) ** 2)

    elif model_type == 'timegan':
        z = torch.randn(200, cfg.beat_len, 1).to(cfg.device)
        d = torch.randn_like(z) * cfg.latent_delta
        model.G.eval(); model.S.eval(); model.R.eval()
        with torch.no_grad():
            x1 = model.R(model.S(model.G(z)))
            x2 = model.R(model.S(model.G(z + d)))
            return torch.mean((x1 - x2) ** 2).item()

    elif model_type == 'vae':
        z = torch.randn(200, cfg.vae_latent).to(cfg.device)
        d = torch.randn_like(z) * cfg.latent_delta
        model.model.eval()
        with torch.no_grad():
            # Decode z
            d_in1 = model.model.decoder_input(z).unsqueeze(1).repeat(1, cfg.beat_len, 1)
            out1, _ = model.model.decoder_lstm(d_in1)
            x1 = model.model.final_layer(out1)

            # Decode z + d
            d_in2 = model.model.decoder_input(z + d).unsqueeze(1).repeat(1, cfg.beat_len, 1)
            out2, _ = model.model.decoder_lstm(d_in2)
            x2 = model.model.final_layer(out2)

            return torch.mean((x1 - x2) ** 2).item()



# =============================================================================
# 5B. OUTPUT / REPRODUCIBILITY HELPERS
# =============================================================================
def make_output_dir():
    """Creates a timestamped output directory for a reproducible run."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_root) / f"{cfg.run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_run_metadata(out_dir):
    """Saves configuration and environment information."""
    config_path = out_dir / "run_config.json"
    env_path = out_dir / "environment.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": np.__version__,
    }
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)


def write_csv(path, rows, fieldnames):
    """Writes a list of dictionaries to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def results_to_rows(results_log):
    """Converts results_log into per-seed raw rows."""
    rows = []
    for model_name in ["Baseline", "Robust", "TimeGAN", "LSTM-VAE"]:
        for seed, metrics in zip(cfg.seeds, results_log[model_name]):
            rows.append({
                "Model": model_name,
                "Seed": seed,
                "Fidelity": metrics["Fidelity"],
                "Diversity": metrics["Diversity"],
                "Robustness": metrics["Robustness"],
                "Utility": metrics["Accuracy"],
                "Fairness": metrics["Fairness"],
                "Safety": metrics["Safety"],
                "MacroF1": metrics["F1"],
            })
    return rows


def save_current_results(out_dir, results_log, record_split_rows=None, final=False):
    """Saves raw per-seed metrics and summary tables. Safe to call after each seed."""
    raw_rows = results_to_rows(results_log)
    raw_fields = ["Model", "Seed", "Fidelity", "Diversity", "Robustness", "Utility", "Fairness", "Safety", "MacroF1"]
    write_csv(out_dir / "svdb_nsv_trust_profiles_raw.csv", raw_rows, raw_fields)

    # Summary uses population std (ddof=0), matching np.std in the original LaTeX table.
    summary_rows = []
    for model_name in ["Baseline", "Robust", "TimeGAN", "LSTM-VAE"]:
        model_rows = [r for r in raw_rows if r["Model"] == model_name]
        if not model_rows:
            continue
        summary = {"Model": model_name, "NSeeds": len(model_rows)}
        for metric in ["Fidelity", "Diversity", "Robustness", "Utility", "MacroF1", "Fairness", "Safety"]:
            vals = np.array([float(r[metric]) for r in model_rows], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(vals))
            summary[f"{metric}_std"] = float(np.std(vals))
        summary_rows.append(summary)

    if summary_rows:
        summary_fields = list(summary_rows[0].keys())
        write_csv(out_dir / "svdb_nsv_trust_profiles_summary.csv", summary_rows, summary_fields)

    if record_split_rows is not None:
        split_fields = ["Seed", "Split", "Records"]
        write_csv(out_dir / "svdb_nsv_record_splits.csv", record_split_rows, split_fields)

    if final:
        # Create a zip archive for easy Colab download.
        zip_base = str(out_dir)
        shutil.make_archive(zip_base, "zip", root_dir=out_dir)
        print(f"\nSaved zip archive: {zip_base}.zip")


def build_latex_table(results_log):
    """Builds the LaTeX table from results_log using the original formatting convention."""
    metrics_order = ["Fidelity", "Diversity", "Robustness", "Accuracy", "F1", "Fairness", "Safety"]
    arrows = {
        "Fidelity": "↓",
        "Diversity": "↑",
        "Robustness": "↓",
        "Accuracy": "↑",
        "F1": "↑",
        "Fairness": "↑",
        "Safety": "↑"
    }

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering \small")
    lines.append(r"\caption{SVDB N/S/V external cohort results (Mean $\pm$ Std over 3 seeds)}")
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"\toprule \textbf{Metric} & \textbf{Baseline} & \textbf{Robust} & \textbf{TimeGAN} & \textbf{LSTM-VAE} \\ \midrule")

    for m in metrics_order:
        row = f"{m} ({arrows[m]})"
        for mod in ["Baseline", "Robust", "TimeGAN", "LSTM-VAE"]:
            v = [r[m] for r in results_log[mod]]
            if m == "Robustness":
                row += f" & {np.mean(v):.1e} $\\pm$ {np.std(v):.1e}"
            elif m == "Fidelity":
                row += f" & {np.mean(v):.3f} $\\pm$ {np.std(v):.3f}"
            else:
                row += f" & {np.mean(v):.4f} $\\pm$ {np.std(v):.4f}"
        lines.append(row + " " + r"\\")

    lines.append(r"\bottomrule \end{tabular} \end{table}")
    return "\n".join(lines)


# =============================================================================
# 5C. SVDB CLASS-COUNT HELPER
# =============================================================================
def scan_svdb_class_counts(out_dir=None):
    """Scans SVDB AAMI counts before training and optionally saves them."""
    os.makedirs(cfg.data_dir, exist_ok=True)
    if not os.path.exists(os.path.join(cfg.data_dir, '800.hea')):
        print("Downloading SVDB Database for class-count scan...")
        wfdb.dl_database(cfg.db_name, dl_dir=cfg.data_dir)

    recs = sorted([f.split('.')[0] for f in os.listdir(cfg.data_dir) if f.endswith('.hea') and f.split('.')[0].isdigit()])
    rows = []
    overall = {c: 0 for c in ["N", "S", "V", "F", "Q"]}

    for rec in recs:
        try:
            path = os.path.join(cfg.data_dir, rec)
            ann = wfdb.rdann(path, 'atr')
        except Exception:
            continue
        cnt = {c: 0 for c in ["N", "S", "V", "F", "Q"]}
        for sym in ann.symbol:
            if sym in AAMI_MAP:
                label = AAMI_MAP[sym]
                if label in cnt:
                    cnt[label] += 1
                    overall[label] += 1
        row = {"Record": rec}
        row.update(cnt)
        rows.append(row)

    print("SVDB AAMI counts:", overall)
    if out_dir is not None:
        write_csv(out_dir / "svdb_aami_class_counts_by_record.csv", rows, ["Record", "N", "S", "V", "F", "Q"])
        with open(out_dir / "svdb_aami_class_counts_overall.json", "w", encoding="utf-8") as f:
            json.dump(overall, f, indent=2)
    return overall

# =============================================================================
# 6. MAIN LOOP
# =============================================================================
results_log = {"Baseline": [], "Robust": [], "TimeGAN": [], "LSTM-VAE": []}
record_split_rows = []
out_dir = make_output_dir()
save_run_metadata(out_dir)
print(f"Output directory: {out_dir}")
scan_svdb_class_counts(out_dir)

print("=" * 60)
print("STARTING SVDB N/S/V EXPERIMENT (Baseline, Robust, TimeGAN, LSTM-VAE)")
print("=" * 60)

for i, seed in enumerate(cfg.seeds):
    print(f"\n>>> SEED {seed} ({i + 1}/{len(cfg.seeds)})")

    # 1. Load Data
    X_tr, y_tr, X_te, y_te, train_recs, test_recs = load_svdb_data(seed)
    record_split_rows.append({"Seed": seed, "Split": "train", "Records": " ".join(train_recs)})
    record_split_rows.append({"Seed": seed, "Split": "test", "Records": " ".join(test_recs)})

    # 2. Train TimeGAN
    print("  Training TimeGAN...")
    tg = TimeGAN()
    tg.train(X_tr)

    # 3. Train LSTM-VAE (New)
    print("  Training LSTM-VAE...")
    vae = VAE_Trainer()
    vae.train(X_tr)

    # Generate Synthetic Data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    X_tg_list, y_tg_list = [], []
    X_vae_list, y_vae_list = [], []

    for c in range(len(CLASSES)):
        n = len(np.where(y_tr == c)[0])
        # TimeGAN
        X_tg_list.append(tg.generate(n))
        y_tg_list.append(np.full(n, c))
        # VAE
        X_vae_list.append(vae.generate(n))
        y_vae_list.append(np.full(n, c))

    X_tg = np.concatenate(X_tg_list)
    y_tg = np.concatenate(y_tg_list)
    X_vae = np.concatenate(X_vae_list)
    y_vae = np.concatenate(y_vae_list)

    # 4. Proxies
    def gen_p(n_val):
        xl, yl = [], []
        for c in range(len(CLASSES)):
            idx = np.where(y_tr == c)[0]
            base = X_tr[np.random.choice(idx, len(idx), replace=True)]
            xl.append(base + n_val * np.random.normal(0, 1, base.shape))
            yl.append(np.full(len(idx), c))
        return np.concatenate(xl), np.concatenate(yl)

    X_b, y_b = gen_p(cfg.baseline_noise)
    X_r, y_r = gen_p(cfg.robust_noise)

    # 5. Evaluate
    print("  Evaluating...")
    m_b = compute_metrics(X_b, y_b, X_te, y_te, seed=seed)
    m_r = compute_metrics(X_r, y_r, X_te, y_te, seed=seed)
    m_t = compute_metrics(X_tg, y_tg, X_te, y_te, seed=seed)
    m_v = compute_metrics(X_vae, y_vae, X_te, y_te, seed=seed)

    m_b['Robustness'] = compute_robustness(X_tr, cfg.baseline_noise, 'proxy')
    m_r['Robustness'] = compute_robustness(X_tr, cfg.robust_noise, 'proxy')
    m_t['Robustness'] = compute_robustness(None, None, 'timegan', model=tg)
    m_v['Robustness'] = compute_robustness(None, None, 'vae', model=vae)

    results_log['Baseline'].append(m_b)
    results_log['Robust'].append(m_r)
    results_log['TimeGAN'].append(m_t)
    results_log['LSTM-VAE'].append(m_v)

    if cfg.save_after_each_seed:
        save_current_results(out_dir, results_log, record_split_rows=record_split_rows, final=False)
        print(f"  Saved intermediate results to: {out_dir}")

# =============================================================================
# 7. FINAL TABLE
# =============================================================================
latex_table = build_latex_table(results_log)
print("\n" + "=" * 100)
print(latex_table)

with open(out_dir / "svdb_nsv_table_latex.txt", "w", encoding="utf-8") as f:
    f.write(latex_table)

save_current_results(out_dir, results_log, record_split_rows=record_split_rows, final=True)

print("\nFINAL OUTPUT FILES")
print("-", out_dir / "svdb_nsv_trust_profiles_raw.csv")
print("-", out_dir / "svdb_nsv_trust_profiles_summary.csv")
print("-", out_dir / "svdb_nsv_record_splits.csv")
print("-", out_dir / "run_config.json")
print("-", out_dir / "environment.json")
print("-", out_dir / "svdb_nsv_table_latex.txt")
print("-", str(out_dir) + ".zip")
