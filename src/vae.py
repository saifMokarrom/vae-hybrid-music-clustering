from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass
class VAEConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, ...] = (512, 256)
    beta: float = 1.0


class MLPVAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        enc_layers = []
        d = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(d, cfg.latent_dim)
        self.fc_logvar = nn.Linear(d, cfg.latent_dim)

        dec_layers = []
        d = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        dec_layers.append(nn.Linear(d, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "mu": mu, "logvar": logvar, "z": z}

    def loss(self, x: torch.Tensor, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_hat = out["x_hat"]
        mu = out["mu"]
        logvar = out["logvar"]

        recon = torch.mean((x_hat - x) ** 2)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon + self.cfg.beta * kl
        return {"loss": total, "recon": recon, "kl": kl}


def train_vae(
    model: MLPVAE,
    x_train: torch.Tensor,
    x_val: torch.Tensor | None = None,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, list]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: Dict[str, list] = {"train_loss": [], "train_recon": [], "train_kl": [], "val_loss": []}

    n = x_train.shape[0]
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        steps = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = x_train[idx].to(device)

            out = model(xb)
            losses = model.loss(xb, out)

            opt.zero_grad()
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += float(losses["loss"].detach().cpu())
            total_recon += float(losses["recon"].detach().cpu())
            total_kl += float(losses["kl"].detach().cpu())
            steps += 1

        history["train_loss"].append(total_loss / max(1, steps))
        history["train_recon"].append(total_recon / max(1, steps))
        history["train_kl"].append(total_kl / max(1, steps))

        if x_val is not None:
            model.eval()
            with torch.no_grad():
                out = model(x_val.to(device))
                vloss = model.loss(x_val.to(device), out)["loss"]
                history["val_loss"].append(float(vloss.detach().cpu()))
        else:
            history["val_loss"].append(float("nan"))

    return history


@torch.no_grad()
def extract_latents(model: MLPVAE, x: torch.Tensor, device: str = "cpu", batch_size: int = 256) -> torch.Tensor:
    model.eval().to(device)
    zs = []
    n = x.shape[0]
    for i in range(0, n, batch_size):
        xb = x[i : i + batch_size].to(device)
        mu, logvar = model.encode(xb)
        z = mu
        zs.append(z.detach().cpu())
    return torch.cat(zs, dim=0)
