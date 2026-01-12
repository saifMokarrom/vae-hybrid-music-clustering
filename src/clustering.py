from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
import umap
import matplotlib.pyplot as plt

from .dataset import BuildConfig, build_features, load_npz, split_train_val
from .evaluation import clustering_metrics
from .vae import MLPVAE, VAEConfig, train_vae, extract_latents


def _ensure_dirs(out_dir: str) -> Dict[str, str]:
    metrics_dir = os.path.join(out_dir, "metrics")
    fig_dir = os.path.join(out_dir, "latent_visualization")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    return {"metrics_dir": metrics_dir, "fig_dir": fig_dir}


def _umap_plot(Z2: np.ndarray, labels: np.ndarray, title: str, out_path: str) -> None:
    plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=18)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _cluster_all(X: np.ndarray, k: int, y_true: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    labels_map: Dict[str, np.ndarray] = {}

    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    labels_map["kmeans"] = km.labels_

    ag = AgglomerativeClustering(n_clusters=k).fit(X)
    labels_map["agglo"] = ag.labels_

    db = DBSCAN(eps=0.5, min_samples=10).fit(X)
    labels_map["dbscan"] = db.labels_

    rows = []
    for name, lab in labels_map.items():
        m = clustering_metrics(X, lab, y_true=y_true)
        rows.append({"method": name, "k": k, **m})

    return pd.DataFrame(rows), labels_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["easy", "medium", "hard"], required=True)
    ap.add_argument("--input_csv", default=None, help="CSV with audio columns + lyrics text.")
    ap.add_argument("--input_npz", default=None, help="Precomputed NPZ with X (and optional y).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.1)
    args = ap.parse_args()

    dirs = _ensure_dirs(args.out_dir)

    if args.input_npz:
        data = load_npz(args.input_npz)
        X = data["X"]
        y = data.get("y", None)
    else:
        if not args.input_csv:
            raise SystemExit("Provide --input_csv or --input_npz")
        npz_path = os.path.join("data", "audio", "features.npz")
        data = build_features(BuildConfig(input_csv=args.input_csv, out_npz=npz_path, max_items=args.max_items))
        X = data["X"]
        y = data.get("y", None)

    y_true = None
    if y is not None:
        y_true = pd.factorize(y)[0]

    if args.mode == "easy":
        dfm, labels_map = _cluster_all(X, k=args.k, y_true=y_true)
        dfm.insert(0, "mode", "easy")
        dfm.to_csv(os.path.join(dirs["metrics_dir"], "clustering_metrics.csv"), index=False)

        reducer = umap.UMAP(random_state=42)
        Z2 = reducer.fit_transform(X)
        _umap_plot(Z2, labels_map["kmeans"], "UMAP features (kmeans)", os.path.join(dirs["fig_dir"], "umap_kmeans.png"))
        _umap_plot(Z2, labels_map["agglo"], "UMAP features (agglo)", os.path.join(dirs["fig_dir"], "umap_agglo.png"))
        _umap_plot(Z2, labels_map["dbscan"], "UMAP features (dbscan)", os.path.join(dirs["fig_dir"], "umap_dbscan.png"))
        print(f"Wrote metrics to {os.path.join(dirs['metrics_dir'], 'clustering_metrics.csv')}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    beta = args.beta
    if beta is None:
        beta = 1.0 if args.mode == "medium" else 4.0

    x_tr, x_va = split_train_val(X, val_split=args.val_split, seed=42)
    x_tr_t = torch.from_numpy(x_tr)
    x_va_t = torch.from_numpy(x_va)

    cfg = VAEConfig(input_dim=X.shape[1], latent_dim=args.latent_dim, beta=beta)
    vae = MLPVAE(cfg)
    hist = train_vae(
        vae,
        x_tr_t,
        x_val=x_va_t,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    Z = extract_latents(vae, torch.from_numpy(X), device=device).numpy()

    dfm, labels_map = _cluster_all(Z, k=args.k, y_true=y_true)
    dfm.insert(0, "mode", args.mode)
    dfm.to_csv(os.path.join(dirs["metrics_dir"], "clustering_metrics.csv"), index=False)

    reducer = umap.UMAP(random_state=42)
    Z2 = reducer.fit_transform(Z)
    _umap_plot(Z2, labels_map["kmeans"], f"UMAP latent ({args.mode}, kmeans)", os.path.join(dirs["fig_dir"], "umap_kmeans.png"))
    _umap_plot(Z2, labels_map["agglo"], f"UMAP latent ({args.mode}, agglo)", os.path.join(dirs["fig_dir"], "umap_agglo.png"))
    _umap_plot(Z2, labels_map["dbscan"], f"UMAP latent ({args.mode}, dbscan)", os.path.join(dirs["fig_dir"], "umap_dbscan.png"))

    pd.DataFrame(hist).to_csv(os.path.join(dirs["metrics_dir"], "train_history.csv"), index=False)
    print(f"Wrote metrics to {os.path.join(dirs['metrics_dir'], 'clustering_metrics.csv')}")


if __name__ == "__main__":
    main()
