"""
Spectral Autoencoder + (Supervised) Contrastive Training
-------------------------------------------------------
- Pixel-level spectra input: (B, C) where C = n_bands (e.g., 242)
- Reconstruction loss: MSE between decoded output and target (typically preprocessed spectra)
- Contrastive loss:
    * "simclr": unsupervised NT-Xent using two augmented views of same sample
    * "supcon": supervised contrastive using labels (supports multi-label too)

Expected DataLoader batch formats:
1) Unsupervised (no labels):
    batch = (x,) or x
2) Supervised:
    batch = (x, y)
Where:
    x: torch.FloatTensor [B, C]
    y: torch.LongTensor [B] for single-label
       OR torch.FloatTensor [B, K] multi-hot for multi-label

Notes:
- For multi-label supcon: positives are samples sharing at least one label bit.
- If you only want contrastive on latent (no recon), set recon_weight=0.0
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim


# ---------------------
# Model blocks
# ---------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class SpectralAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 242,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Encoder
        enc_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_blocks):
            enc_layers.append(ResidualBlock(hidden_dim, dropout))
        enc_layers += [
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        ]
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirrors)
        dec_layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_blocks):
            dec_layers.append(ResidualBlock(hidden_dim, dropout))
        dec_layers += [
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        ]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    @torch.no_grad()
    def encode(self, x):
        return self.encoder(x)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu"):
        self.load_state_dict(torch.load(path, map_location=device))


# ---------------------
# Augmentor (two-view)
# ---------------------
class SpectralAugmentor:
    """
    Scale-aware augmentations for hyperspectral 1D spectra.
    Returns one augmented view (x_aug) and the "clean" target (x_target).
    """
    def __init__(
        self,
        noise_scale: float = 0.02,
        scale_jitter: float = 0.10,
        max_shift: int = 1,
        mixup_alpha: Optional[float] = 0.2,
        p_mixup: float = 0.3,
    ):
        self.noise_scale = noise_scale
        self.scale_jitter = scale_jitter
        self.max_shift = max_shift
        self.mixup_alpha = mixup_alpha
        self.p_mixup = p_mixup

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C = x.shape
        x_aug = x.clone()

        # 1) Gaussian noise scaled by per-band std (batch-based)
        if self.noise_scale and self.noise_scale > 0:
            band_std = torch.std(x, dim=0, keepdim=True) + 1e-8
            noise = torch.randn_like(x_aug) * (band_std * self.noise_scale)
            x_aug = x_aug + noise

        # 2) Intensity scaling
        if self.scale_jitter and self.scale_jitter > 0:
            jitter = 1.0 + (torch.rand(B, 1, device=x.device) - 0.5) * (2 * self.scale_jitter)
            x_aug = x_aug * jitter

        # 3) Spectral shift
        if self.max_shift and self.max_shift > 0:
            shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,), device=x.device).item()
            if shift != 0:
                x_aug = torch.roll(x_aug, shifts=shift, dims=1)

        # 4) Mixup
        if self.mixup_alpha is not None and self.mixup_alpha > 0 and torch.rand(1).item() < self.p_mixup:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
            idx = torch.randperm(B, device=x.device)
            x_aug = lam * x_aug + (1 - lam) * x_aug[idx]

        return x_aug, x


def make_two_views(augment: SpectralAugmentor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates two independent augmented views and a shared target.
    Returns: x1, x2, x_target
    """
    x1, x_target = augment(x)
    x2, _ = augment(x)
    return x1, x2, x_target


# ---------------------
# Contrastive losses
# ---------------------
def l2_normalize(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return z / (z.norm(dim=1, keepdim=True) + eps)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    SimCLR-style NT-Xent loss for a batch.
    Positives: (z1[i], z2[i])
    Negatives: all other pairs.
    """
    z1 = l2_normalize(z1)
    z2 = l2_normalize(z2)
    B = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.T) / temperature  # [2B, 2B]

    # mask self-similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positives: i <-> i+B
    pos_idx = torch.arange(B, device=z.device)
    positives = torch.cat([pos_idx + B, pos_idx], dim=0)  # [2B]
    loss = F.cross_entropy(sim, positives)
    return loss


def _labels_to_pos_mask(y: torch.Tensor) -> torch.Tensor:
    """
    Build positive mask for supervised contrastive:
    - single-label: y shape [B] (int)
    - multi-label: y shape [B, K] (float/bool multi-hot)
    Returns pos_mask [B, B] with pos_mask[i, j] = True if j is positive for i (excluding i itself).
    """
    if y.ndim == 1:
        # single-label
        y = y.view(-1, 1)
        pos = (y == y.T)
    else:
        # multi-label: positive if share any label bit
        # y assumed multi-hot (0/1), float ok
        yb = (y > 0).float()
        pos = (yb @ yb.T) > 0  # share at least one label

    # exclude self
    pos.fill_diagonal_(False)
    return pos


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F

def supervised_contrastive_loss(
    z1, z2, y,
    temperature=0.2,
    eps=1e-12,
    multilabel_mode="overlap",   # "overlap" or "jaccard"
    tau_pos=0.5                 # only used for jaccard
):
    """
    SupCon for 2 views per sample.

    Key fixes vs your version:
    1) Always treat the other view of the same sample as a positive.
    2) Multi-label positives can be defined by overlap or Jaccard threshold.
    """

    def l2_normalize(z):
        return z / (z.norm(dim=1, keepdim=True) + eps)

    B = z1.size(0)
    z1 = l2_normalize(z1)
    z2 = l2_normalize(z2)

    z = torch.cat([z1, z2], dim=0)              # (2B, D)
    logits = (z @ z.T) / temperature            # (2B, 2B)

    # mask self comparisons (same row/col)
    self_mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, -1e9)

    # --- build pos_bb at sample level (B,B) ---
    if y.ndim == 1:
        y_ = y.view(-1, 1)
        pos_bb = (y_ == y_.T)
    else:
        yb = (y > 0).float()

        if multilabel_mode == "overlap":
            # positive if share at least one label
            pos_bb = (yb @ yb.T) > 0

        elif multilabel_mode == "jaccard":
            # Jaccard similarity = |A∩B| / |A∪B|
            inter = (yb @ yb.T)                         # (B,B)
            row_sum = yb.sum(dim=1, keepdim=True)       # (B,1)
            union = row_sum + row_sum.T - inter
            jacc = inter / (union + 1e-9)
            pos_bb = jacc >= tau_pos

        else:
            raise ValueError(f"Unknown multilabel_mode: {multilabel_mode}")

    # IMPORTANT: do NOT remove diagonal yet, because we need cross-view identity positives
    # We'll build the full 2B mask and then remove exact self-comparisons only.

    # --- expand to views (2B,2B) ---
    pos = torch.zeros((2 * B, 2 * B), device=z.device, dtype=torch.bool)
    pos[:B, :B] = pos_bb
    pos[:B, B:] = pos_bb
    pos[B:, :B] = pos_bb
    pos[B:, B:] = pos_bb

    # remove self-comparisons within the same view (already masked in logits, but keep consistent)
    pos = pos & (~self_mask)

    # ✅ Force cross-view identity positives: i (view1) ↔ i+B (view2)
    idx = torch.arange(B, device=z.device)
    pos[idx, idx + B] = True
    pos[idx + B, idx] = True

    # compute log-prob
    log_prob = F.log_softmax(logits, dim=1)

    pos_counts = pos.sum(dim=1)   # positives per anchor
    valid = pos_counts > 0
    if valid.sum() == 0:
        return torch.zeros((), device=z.device, dtype=logits.dtype)

    mean_log_prob_pos = (pos.float() * log_prob).sum(dim=1) / pos_counts.clamp_min(1)
    loss = -mean_log_prob_pos[valid].mean()
    return loss




# ---------------------
# Training config
# ---------------------
@dataclass
class AETrainCfg:
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    verbose: bool = True

    # loss weights
    recon_weight: float = 1.0
    contrast_weight: float = 1.0

    # contrastive mode: "none" | "simclr" | "supcon"
    contrastive_mode: str = "supcon"
    temperature: float = 0.2

    # optional projection head (often helps contrastive)
    use_projection_head: bool = True
    proj_dim: int = 64


class ProjectionHead(nn.Module):
    """
    Small MLP head for contrastive learning (encoder output -> projection space).
    Often improves contrastive performance without hurting encoder features.
    """
    def __init__(self, in_dim: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------
# Training loop
# ---------------------
def _unpack_batch(batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Supports:
      - batch = x
      - batch = (x,)
      - batch = (x, y)
    Returns: x, y (y can be None)
    """
    if isinstance(batch, (tuple, list)):
        if len(batch) == 1:
            return batch[0], None
        if len(batch) >= 2:
            return batch[0], batch[1]
    return batch, None


def train_autoencoder_contrastive(
    model: SpectralAE,
    train_loader: DataLoader,
    augment: SpectralAugmentor,
    cfg: AETrainCfg,
    val_loader: Optional[DataLoader] = None,
    device: str = "cuda",
    save_path: str = "best_ae.pt",
) -> SpectralAE:
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    loss_recon_fn = nn.MSELoss()

    proj = None
    if cfg.use_projection_head and cfg.contrastive_mode in ("simclr", "supcon"):
        proj = ProjectionHead(model.encoder[-1].out_features, cfg.proj_dim).to(device)
        opt = optim.Adam(
            list(model.parameters()) + list(proj.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    best_val = float("inf")
    no_improve = 0

    def forward_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          xhat1, z1, xhat2, z2
        """
        x1, x2, x_target = make_two_views(augment, x)
        x1 = x1.to(device, dtype=torch.float32)
        x2 = x2.to(device, dtype=torch.float32)
        x_target = x_target.to(device, dtype=torch.float32)

        xhat1, z1 = model(x1)
        xhat2, z2 = model(x2)
        return xhat1, z1, xhat2, z2, x_target

    for ep in range(1, cfg.epochs + 1):
        model.train()
        if proj is not None:
            proj.train()

        total_loss = 0.0
        total_n = 0

        for batch in train_loader:
            x, y = _unpack_batch(batch)
            x = x.to(device, dtype=torch.float32)
            if y is not None:
                y = y.to(device)

            xhat1, z1, xhat2, z2, x_target = forward_views(x)

            # reconstruction (both views to same target)
            recon_loss = 0.0
            if cfg.recon_weight > 0:
                recon_loss = loss_recon_fn(xhat1, x_target) + loss_recon_fn(xhat2, x_target)

            # contrastive
            contrast_loss = 0.0
            if cfg.contrastive_mode != "none" and cfg.contrast_weight > 0:
                z1c, z2c = z1, z2
                if proj is not None:
                    z1c = proj(z1c)
                    z2c = proj(z2c)

                if cfg.contrastive_mode == "simclr":
                    contrast_loss = nt_xent_loss(z1c, z2c, temperature=cfg.temperature)
                elif cfg.contrastive_mode == "supcon":
                    if y is None:
                        raise ValueError("contrastive_mode='supcon' requires labels y in the train_loader.")
                    contrast_loss = supervised_contrastive_loss(z1c, z2c, y, temperature=cfg.temperature)
                else:
                    raise ValueError(f"Unknown contrastive_mode: {cfg.contrastive_mode}")

            loss = cfg.recon_weight * recon_loss + cfg.contrast_weight * contrast_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        avg_train = total_loss / max(total_n, 1)

        # ---------------- Validation (recon-only is typical, but you can also monitor total)
        avg_val = None
        if val_loader is not None:
            model.eval()
            if proj is not None:
                proj.eval()
            val_loss = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, _ = _unpack_batch(batch)
                    x = x.to(device, dtype=torch.float32)
                    # no aug in val: recon on clean input
                    xhat, _ = model(x)
                    l = loss_recon_fn(xhat, x)  # recon validation on clean
                    val_loss += l.item() * x.size(0)
                    val_n += x.size(0)

            avg_val = val_loss / max(val_n, 1)

            if cfg.verbose:
                print(f"[Epoch {ep}/{cfg.epochs}] Train(total): {avg_train:.6f} | Val(recon): {avg_val:.6f}")

            # checkpoint on best val recon
            if avg_val < best_val:
                best_val = avg_val
                no_improve = 0
                ckpt = {"model": model.state_dict()}
                if proj is not None:
                    ckpt["proj"] = proj.state_dict()
                torch.save(ckpt, save_path)
                if cfg.verbose:
                    print(f"  → Saved best checkpoint (val_recon={best_val:.6f})")
            else:
                no_improve += 1
                if no_improve >= cfg.early_stopping_patience:
                    if cfg.verbose:
                        print(f"Early stopping at epoch {ep}.")
                    break
        else:
            if cfg.verbose:
                print(f"[Epoch {ep}/{cfg.epochs}] Train(total): {avg_train:.6f}")
    ckpt = {"model": model.state_dict()}
    torch.save(ckpt, save_path)
    # restore best
    if os.path.exists(save_path) and val_loader is not None:
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if proj is not None and "proj" in ckpt:
            proj.load_state_dict(ckpt["proj"])
        if cfg.verbose:
            print(f"Restored best checkpoint from {save_path}")

    return model


# ---------------------
# Engine wrapper
# ---------------------
def train_engine_contrastive(
    x_train,
    x_val,
    cfg_dict: dict,
    save_path: str,
    y_train=None,
    y_val=None,
    device: str = "cuda",
) -> SpectralAE:
    """
    If you provide y_train/y_val, the DataLoader will yield (x, y) and you can use supcon.
    If y_train is None, you can still use simclr (unsupervised contrastive) or recon-only.
    """

    # Build model
    ae = SpectralAE(
        input_dim=cfg_dict.get("input_dim", 242),
        latent_dim=cfg_dict["latent_dim"],
        hidden_dim=cfg_dict["hidden_dim"],
        num_blocks=cfg_dict["num_blocks"],
        dropout=cfg_dict["dropout"],
    )

    # Datasets
    Xtr = torch.from_numpy(x_train).float()
    Xva = torch.from_numpy(x_val).float()

    if y_train is not None:
        Ytr = torch.from_numpy(y_train)
        train_ds = TensorDataset(Xtr, Ytr)
    else:
        train_ds = TensorDataset(Xtr)

    if y_val is not None:
        Yva = torch.from_numpy(y_val)
        val_ds = TensorDataset(Xva, Yva)
    else:
        val_ds = TensorDataset(Xva)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_dict.get("batch_size", 256),
        shuffle=True,
        drop_last=False,
        num_workers=cfg_dict.get("num_workers", 0),
        pin_memory=True if device.startswith("cuda") else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_dict.get("batch_size", 256),
        shuffle=False,  # important: don't shuffle val
        drop_last=False,
        num_workers=cfg_dict.get("num_workers", 0),
        pin_memory=True if device.startswith("cuda") else False,
    )

    # Augment
    aug_cfg = cfg_dict.get("data_augment", cfg_dict.get("data-agument", {}))  # tolerate your old key
    augment = SpectralAugmentor(
        noise_scale=aug_cfg.get("noise_scale", 0.02),
        scale_jitter=aug_cfg.get("scale_jitter", 0.10),
        max_shift=aug_cfg.get("max_shift", 1),
        mixup_alpha=aug_cfg.get("mixup_alpha", 0.2),
        p_mixup=aug_cfg.get("p_mixup", 0.3),
    )

    # Train cfg
    train_cfg = AETrainCfg(
        epochs=cfg_dict.get("epochs", 40),
        lr=cfg_dict.get("lr", 1e-3),
        weight_decay=cfg_dict.get("weight_decay", 1e-5),
        early_stopping_patience=cfg_dict.get("early_stopping_patience", 10),
        verbose=cfg_dict.get("verbose", True),
        recon_weight=cfg_dict.get("recon_weight", 1.0),
        contrast_weight=cfg_dict.get("contrast_weight", 1.0),
        contrastive_mode=cfg_dict.get("contrastive_mode", "supcon"),
        temperature=cfg_dict.get("temperature", 0.2),
        use_projection_head=cfg_dict.get("use_projection_head", True),
        proj_dim=cfg_dict.get("proj_dim", 64),
    )

    # Guardrails
    if train_cfg.contrastive_mode == "supcon" and y_train is None:
        raise ValueError("contrastive_mode='supcon' requires y_train (and ideally y_val). "
                         "If you don't have labels for AE training, use 'simclr' or 'none'.")

    ae = train_autoencoder_contrastive(
        model=ae,
        train_loader=train_loader,
        augment=augment,
        cfg=train_cfg,
        val_loader=val_loader,
        device=device,
        save_path=save_path,
    )

    # Save final (best already stored as checkpoint, but keep consistent artifact)
    ae.save(save_path)
    return ae
