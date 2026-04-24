import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import TensorDataset, DataLoader
# ---------------------
# Residual MLP Block
# ---------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)


# ---------------------
# Autoencoder
# ---------------------
class SpectralAE(nn.Module):
    def __init__(
        self,
        input_dim=242,
        latent_dim=32,
        hidden_dim=128,
        num_blocks=3,
        dropout=0.1,
        noise_std=0.02
    ):
        super().__init__()

        self.noise_std = noise_std

        # Encoder
        enc_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_blocks):
            enc_layers.append(ResidualBlock(hidden_dim, dropout))

        enc_layers.append(nn.LayerNorm(hidden_dim))
        enc_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder mirrors encoder (without skip connections)
        dec_layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_blocks):
            dec_layers.append(ResidualBlock(hidden_dim, dropout))

        dec_layers.append(nn.LayerNorm(hidden_dim))
        dec_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        # Add denoising noise during training
        #if self.training and self.noise_std > 0:
            #noise = torch.randn_like(x) * self.noise_std
            #x = x + noise
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device="cpu"):
        state = torch.load(path, map_location=device)
        self.load_state_dict(state)

import torch

class SpectralAugmentor:
    """
    Scale-aware augmentations for hyperspectral 1D spectra.
    All perturbations automatically adapt to the magnitude of input spectra.
    """

    def __init__(
        self,
        noise_scale=0.02,        # noise strength relative to per-band std
        scale_jitter=0.10,       # ±10% intensity scaling
        max_shift=1,             # spectral shift of ±1 band
        mixup_alpha=0.2,
        p_mixup=0.3
    ):
        self.noise_scale = noise_scale
        self.scale_jitter = scale_jitter
        self.max_shift = max_shift
        self.mixup_alpha = mixup_alpha
        self.p_mixup = p_mixup

    def __call__(self, x):
        """
        x: clean batch, shape (B, C)
        Returns (augmented_x, clean_x)
        """
        B, C = x.shape
        x_aug = x.clone()

        # -------------------------
        # 1. Gaussian noise (scaled to per-band variance)
        # -------------------------
        if self.noise_scale > 0:
            # compute per-band std across batch
            band_std = torch.std(x, dim=0, keepdim=True) + 1e-8
            noise = torch.randn_like(x_aug) * (band_std * self.noise_scale)
            x_aug = x_aug + noise

        # -------------------------
        # 2. Intensity scaling (relative)
        # -------------------------
        if self.scale_jitter > 0:
            # scaling factor for each spectrum
            jitter = 1.0 + (torch.rand(B, 1, device=x.device) - 0.5) * (2 * self.scale_jitter)
            x_aug = x_aug * jitter

        # -------------------------
        # 3. Spectral jitter (shift of ±max_shift bands)
        # -------------------------
        if self.max_shift > 0:
            shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,), device=x.device).item()
            if shift != 0:
                x_aug = torch.roll(x_aug, shifts=shift, dims=1)

        # -------------------------
        # 4. Mixup (physics-friendly)
        # -------------------------
        if self.mixup_alpha is not None and torch.rand(1).item() < self.p_mixup:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
            idx = torch.randperm(B)
            x_aug = lam * x_aug + (1 - lam) * x_aug[idx]

        return x_aug, x
# ---------------------
# Training Function
# ---------------------
def train_autoencoder(
    model,
    train_loader,
    augment,
    cfg,
    val_loader=None,          # optional: if None, no validation monitoring

    device="cuda",

    save_path="best_ae.pt",
    
):
    
    """
    Train the autoencoder and automatically save the best model based on validation loss.

    Arguments:
    ----------
    model                 : the SpectralAE instance
    train_loader          : DataLoader for training
    augment               : SpectralAugmentor instance
    val_loader            : DataLoader for validation (optional)
    save_path             : where to save best model
    early_stopping_patience : epochs to wait after last improvement
    """
    epochs = cfg.get('epochs', 40)
    lr = cfg.get('lr', 1e-3)
    weight_decay = cfg.get('weight_decay', 1e-5)
    early_stopping_patience = cfg.get('early_stopping_patience', 10)
    verbose = cfg.get('verbose', True)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0

        # ------------------------
        # Training Loop
        # ------------------------
        for xb in train_loader:
            
            xb = xb[0]
            #xb = batch[0] if isinstance(batch, (tuple, list)) else batch
            #print(xb)
            xb = xb.to(device, dtype=torch.float32)

            # apply augmentation
            xb_aug, xb_target = augment(xb)
            
            # forward pass: reconstruction
            x_hat,_ = model(xb_aug)
            
            loss = loss_fn(x_hat, xb_target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * xb.size(0)
            train_count += xb.size(0)

        avg_train_loss = train_loss / max(train_count, 1)

        # ------------------------
        # Validation Loop
        # ------------------------
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_count = 0

            with torch.no_grad():
                for batch in val_loader:
                    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                    xb = xb.to(device, dtype=torch.float32)

                    # NOTE: no augmentation during validation
                    x_hat,_ = model(xb)
                    loss = loss_fn(x_hat, xb)

                    val_loss += loss.item() * xb.size(0)
                    val_count += xb.size(0)

            avg_val_loss = val_loss / max(val_count, 1)

            if verbose:
                print(f"[Epoch {ep}/{epochs}] Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

            # Check if this is the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), save_path)
                if verbose:
                    print(f"  → Saved best model (val_loss={best_val_loss:.6f})")
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {ep}. Restoring best model.")
                break

        else:
            # No validation: always save last epoch model
            torch.save(model.state_dict(), save_path)
            if verbose:
                print(f"[Epoch {ep}/{epochs}] Train: {avg_train_loss:.6f} (saved model)")

    # Load best model before returning (important!)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        if verbose and val_loader is not None:
            print(f"Restored best model from {save_path}")

    return model
def train_engine(x_train,x_val,cfg,save_path):
    
    ae = SpectralAE(
    input_dim=242,
    latent_dim=cfg['latent_dim'],
    hidden_dim=cfg['hidden_dim'],
    num_blocks=cfg['num_blocks'],     # 3 residual blocks → strong but not overkill
    dropout=cfg['dropout'],
    noise_std=0.02,
    )
    X_tensor = torch.from_numpy(x_train).float()
    
    dataset_train = TensorDataset(X_tensor)
    dataset_val = TensorDataset(torch.from_numpy(x_val).float())
    trainloader = DataLoader(
        dataset_train,
        batch_size=256,      # or 512 if GPU is strong
        shuffle=True,
        drop_last=False,
        num_workers=0        # keep 0 for small 1D data; no need for parallel workers
    )
    valloader = DataLoader(
        dataset_val,
        batch_size=256,      # or 512 if GPU is strong
        shuffle=True,
        drop_last=False,
        num_workers=0        # keep 0 for small 1D data; no need for parallel workers
    )

    noise_scale = cfg['data-agument'].get('noise_scale',0.1)
    scale_jitter = cfg['data-agument'].get('scale_jitter',0.2)
    max_shift = cfg['data-agument'].get('max_shift',0)
    mixup_alpha = cfg['data-agument'].get('mixup_alpha',0)
    p_mixup = cfg['data-agument'].get('p_mixup',0.0)
    augment = SpectralAugmentor(
    noise_scale=noise_scale,
    scale_jitter=scale_jitter,
    max_shift=max_shift,
    mixup_alpha=mixup_alpha,
    p_mixup=p_mixup
    )
    

    train_autoencoder(
        ae,
        trainloader,
        augment,
        cfg,
        val_loader=valloader,
        
        save_path=save_path
        ,
        device="cuda",   # or "cpu"
        
    )
    ae.save(save_path)
    return ae

        