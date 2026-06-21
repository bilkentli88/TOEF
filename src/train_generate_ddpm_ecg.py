"""
ecg_ddpm_colab_modified.py
---------------------------
Colab-friendly minimal 1D class-conditional DDPM for ECG beat generation.

Purpose
-------
Representative diffusion-paradigm generator for the Trust-Oriented Evaluation
Framework. This is not intended as a state-of-the-art ECG generator.

Expected data
-------------
X_train: float32 NumPy array, shape (N, 256), already normalized exactly as in
         the TimeGAN/LSTM-VAE experiments.
y_train: int NumPy array, shape (N,), class labels 0..num_classes-1.

Key fixes over the initial draft
--------------------------------
1. Full seed control before model initialization.
2. Class-balanced training through WeightedRandomSampler for imbalanced ECG data.
3. Colab command-line interface for loading .npy files, training, sampling, and
   saving generated samples/checkpoints.
4. Optional automatic mixed precision on CUDA.

Typical Colab usage
-------------------
!python ecg_ddpm_colab_modified.py \
    --data_dir /content/drive/MyDrive/aim_ecg_data/seed_19 \
    --x_file X_train.npy \
    --y_file y_train.npy \
    --out_dir /content/drive/MyDrive/aim_ddpm_outputs/seed_19 \
    --seed 19 \
    --num_classes 5 \
    --seq_len 256 \
    --epochs 120 \
    --batch_size 128 \
    --base_ch 64 \
    --T 300 \
    --ddim_steps 50 \
    --guidance 2.0 \
    --samples_per_class 2000

If class-specific sample counts are needed:
!python ecg_ddpm_colab_modified.py ... --counts_json '{"0":2000,"1":2000,"2":2000,"3":2000,"4":2000}'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# Timestep sinusoidal embedding
# ----------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        if half <= 1:
            raise ValueError("Embedding dimension must be at least 4.")
        scale = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -scale)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ----------------------------------------------------------------------
# Residual block with additive time + class conditioning
# ----------------------------------------------------------------------
class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        if in_ch % groups != 0 or out_ch % groups != 0:
            raise ValueError(
                f"Channels must be divisible by groups={groups}. "
                f"Got in_ch={in_ch}, out_ch={out_ch}."
            )
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# ----------------------------------------------------------------------
# Small 1D U-Net for 256-sample ECG beats
# ----------------------------------------------------------------------
class UNet1D(nn.Module):
    def __init__(
        self,
        seq_len: int = 256,
        base_ch: int = 64,
        ch_mults: Tuple[int, ...] = (1, 2, 4),
        num_classes: int = 5,
        emb_dim: int = 256,
        groups: int = 8,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.null_idx = num_classes

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)

        chs = [base_ch * m for m in ch_mults]
        self.in_conv = nn.Conv1d(1, base_ch, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = base_ch
        for c in chs:
            self.down_blocks.append(ResBlock1D(prev, c, emb_dim, groups=groups))
            self.downsamples.append(nn.Conv1d(c, c, kernel_size=4, stride=2, padding=1))
            prev = c

        self.mid = ResBlock1D(prev, prev, emb_dim, groups=groups)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for c in reversed(chs):
            self.upsamples.append(nn.ConvTranspose1d(prev, c, kernel_size=4, stride=2, padding=1))
            self.up_blocks.append(ResBlock1D(c * 2, c, emb_dim, groups=groups))
            prev = c

        self.out_norm = nn.GroupNorm(groups, prev)
        self.out_conv = nn.Conv1d(prev, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, L) -> (B, 1, L)
        x = x[:, None, :]
        emb = self.time_mlp(t) + self.class_emb(y)

        h = self.in_conv(x)
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, emb)
            skips.append(h)
            h = down(h)

        h = self.mid(h, emb)

        for up, block, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, emb)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h[:, 0, :]


# ----------------------------------------------------------------------
# Noise schedule
# ----------------------------------------------------------------------
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return betas.clamp(1e-4, 0.999)


# ----------------------------------------------------------------------
# Class-conditional 1D DDPM
# ----------------------------------------------------------------------
class ECGDiffusion:
    def __init__(
        self,
        seq_len: int = 256,
        num_classes: int = 5,
        T: int = 300,
        base_ch: int = 64,
        ch_mults: Tuple[int, ...] = (1, 2, 4),
        emb_dim: int = 256,
        p_uncond: float = 0.1,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        deterministic: bool = True,
    ):
        if seed is not None:
            set_seed(seed, deterministic=deterministic)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.seq_len = seq_len
        self.num_classes = num_classes
        self.T = T
        self.p_uncond = p_uncond
        self.device = torch.device(device)

        self.model = UNet1D(
            seq_len=seq_len,
            base_ch=base_ch,
            ch_mults=ch_mults,
            num_classes=num_classes,
            emb_dim=emb_dim,
        ).to(self.device)

        betas = cosine_beta_schedule(T).to(self.device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.acp = torch.cumprod(self.alphas, dim=0)
        self.acp_prev = F.pad(self.acp[:-1], (1, 0), value=1.0)
        self.ema_state = None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        balanced: bool = True,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.long)
        if X_t.ndim != 2 or X_t.shape[1] != self.seq_len:
            raise ValueError(f"Expected X shape (N, {self.seq_len}), got {tuple(X_t.shape)}")
        if y_t.ndim != 1 or y_t.shape[0] != X_t.shape[0]:
            raise ValueError(f"Expected y shape (N,), got {tuple(y_t.shape)}")
        if y_t.min().item() < 0 or y_t.max().item() >= self.num_classes:
            raise ValueError(f"Labels must be in 0..{self.num_classes - 1}.")

        ds = torch.utils.data.TensorDataset(X_t, y_t)

        if balanced:
            # Class-balanced sampling is important for MIT-BIH-style imbalance.
            class_counts = torch.bincount(y_t, minlength=self.num_classes).float()
            class_counts[class_counts == 0] = 1.0
            sample_weights = 1.0 / class_counts[y_t]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights.double(),
                num_samples=len(sample_weights),
                replacement=True,
            )
            return torch.utils.data.DataLoader(
                ds,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 120,
        batch_size: int = 128,
        lr: float = 2e-4,
        seed: int = 19,
        ema_decay: float = 0.999,
        balanced: bool = True,
        amp: bool = True,
        verbose: bool = True,
    ) -> "ECGDiffusion":
        set_seed(seed)
        dl = self._make_loader(X, y, batch_size=batch_size, balanced=balanced)

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=(amp and self.device.type == "cuda"))

        ema = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        for ep in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_seen = 0

            for xb, yb in dl:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                B = xb.size(0)
                t = torch.randint(0, self.T, (B,), device=self.device)

                # Classifier-free guidance training: randomly drop conditioning.
                y_in = yb.clone()
                drop = torch.rand(B, device=self.device) < self.p_uncond
                y_in[drop] = self.model.null_idx

                noise = torch.randn_like(xb)
                ac = self.acp[t][:, None]
                x_t = ac.sqrt() * xb + (1 - ac).sqrt() * noise

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(amp and self.device.type == "cuda")):
                    pred = self.model(x_t, t, y_in)
                    loss = F.mse_loss(pred, noise)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                with torch.no_grad():
                    sd = self.model.state_dict()
                    for k, v in sd.items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)
                        else:
                            ema[k] = v.detach().clone()

                total_loss += loss.item() * B
                total_seen += B

            if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
                print(f"[seed {seed}] epoch {ep:03d}/{epochs} loss={total_loss/max(total_seen,1):.5f}")

        self.ema_state = ema
        return self

    @torch.no_grad()
    def _sample_batch(
        self,
        y: torch.Tensor,
        ddim_steps: int = 50,
        guidance: float = 2.0,
        use_ema: bool = True,
        x0_clip: Optional[float] = None,
    ) -> np.ndarray:
        backup = None
        if use_ema and self.ema_state is not None:
            backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.ema_state)

        self.model.eval()
        B = y.size(0)
        x = torch.randn(B, self.seq_len, device=self.device)
        step_idx = torch.linspace(self.T - 1, 0, ddim_steps, device=self.device).long()

        for i, t_cur_tensor in enumerate(step_idx):
            t_cur = int(t_cur_tensor.item())
            t_batch = torch.full((B,), t_cur, device=self.device, dtype=torch.long)
            ac_t = self.acp[t_cur]

            if i + 1 < len(step_idx):
                t_next = int(step_idx[i + 1].item())
                ac_next = self.acp[t_next] if t_next > 0 else torch.tensor(1.0, device=self.device)
            else:
                ac_next = torch.tensor(1.0, device=self.device)

            eps_c = self.model(x, t_batch, y)
            if guidance is not None and guidance != 1.0:
                y_null = torch.full_like(y, self.model.null_idx)
                eps_u = self.model(x, t_batch, y_null)
                eps = eps_u + guidance * (eps_c - eps_u)
            else:
                eps = eps_c

            x0 = (x - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()
            if x0_clip is not None and x0_clip > 0:
                x0 = x0.clamp(-x0_clip, x0_clip)
            x = ac_next.sqrt() * x0 + (1 - ac_next).sqrt() * eps

        if backup is not None:
            self.model.load_state_dict(backup)

        return x.detach().cpu().numpy().astype(np.float32)

    def sample_per_class(
        self,
        counts: Dict[int, int],
        ddim_steps: int = 50,
        guidance: float = 2.0,
        batch: int = 512,
        use_ema: bool = True,
        x0_clip: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for cls in sorted(counts.keys()):
            n = int(counts[cls])
            if n <= 0:
                continue
            done = 0
            while done < n:
                b = min(batch, n - done)
                y = torch.full((b,), int(cls), device=self.device, dtype=torch.long)
                Xs.append(self._sample_batch(y, ddim_steps=ddim_steps, guidance=guidance, use_ema=use_ema, x0_clip=x0_clip))
                ys.append(np.full(b, int(cls), dtype=np.int64))
                done += b
                print(f"sampled class {cls}: {done}/{n}")

        if not Xs:
            raise ValueError("No samples requested. Check counts.")
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    def save_checkpoint(self, path: str | Path, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "ema_state": self.ema_state,
            "seq_len": self.seq_len,
            "num_classes": self.num_classes,
            "T": self.T,
            "p_uncond": self.p_uncond,
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.ema_state = ckpt.get("ema_state", None)


# ----------------------------------------------------------------------
# Utility functions for Colab CLI
# ----------------------------------------------------------------------
def parse_counts(args, y_train: np.ndarray) -> Dict[int, int]:
    if args.counts_json:
        raw = json.loads(args.counts_json)
        return {int(k): int(v) for k, v in raw.items()}

    if args.match_train_counts:
        vals, cnts = np.unique(y_train.astype(int), return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnts)}

    if args.samples_per_class is None:
        raise ValueError("Provide --samples_per_class, --counts_json, or --match_train_counts.")

    return {c: int(args.samples_per_class) for c in range(args.num_classes)}


def load_arrays(data_dir: str | Path, x_file: str, y_file: str) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    X = np.load(data_dir / x_file).astype(np.float32)
    y = np.load(data_dir / y_file).astype(np.int64)
    return X, y


def summarize_labels(y: np.ndarray, num_classes: int) -> None:
    counts = np.bincount(y.astype(int), minlength=num_classes)
    print("class counts:", {i: int(counts[i]) for i in range(num_classes)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/sample a class-conditional 1D DDPM for ECG beats.")

    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--x_file", type=str, default="X_train.npy")
    parser.add_argument("--y_file", type=str, default="y_train.npy")
    parser.add_argument("--out_dir", type=str, default="./ddpm_outputs")

    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--p_uncond", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--sample_batch", type=int, default=512)
    parser.add_argument("--x0_clip", type=float, default=0.0, help="Clip predicted x0 during sampling; <=0 disables clipping. Use <=0 for fair physiological-validity evaluation.")
    parser.add_argument("--samples_per_class", type=int, default=None)
    parser.add_argument("--counts_json", type=str, default=None)
    parser.add_argument("--match_train_counts", action="store_true")

    parser.add_argument("--no_balanced_sampler", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    X_train, y_train = load_arrays(args.data_dir, args.x_file, args.y_file)
    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    summarize_labels(y_train, args.num_classes)

    counts = parse_counts(args, y_train)
    print("sampling counts:", counts)

    gen = ECGDiffusion(
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        T=args.T,
        base_ch=args.base_ch,
        p_uncond=args.p_uncond,
        device=device,
        seed=args.seed,
    )
    print("trainable parameters:", gen.count_parameters())

    gen.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        ema_decay=args.ema_decay,
        balanced=(not args.no_balanced_sampler),
        amp=(not args.no_amp),
        verbose=True,
    )

    if args.save_checkpoint:
        ckpt_path = out_dir / f"ddpm_seed{args.seed}.pt"
        gen.save_checkpoint(ckpt_path, extra=vars(args))
        print("saved checkpoint:", ckpt_path)

    x0_clip_arg = None if args.x0_clip <= 0 else float(args.x0_clip)

    X_syn, y_syn = gen.sample_per_class(
        counts=counts,
        ddim_steps=args.ddim_steps,
        guidance=args.guidance,
        batch=args.sample_batch,
        use_ema=True,
        x0_clip=x0_clip_arg,
    )

    x_out = out_dir / f"ddpm_X_syn_seed{args.seed}.npy"
    y_out = out_dir / f"ddpm_y_syn_seed{args.seed}.npy"
    np.save(x_out, X_syn.astype(np.float32))
    np.save(y_out, y_syn.astype(np.int64))

    print("saved:", x_out, X_syn.shape)
    print("saved:", y_out, y_syn.shape)
    print("done")


if __name__ == "__main__":
    main()
