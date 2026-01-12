# Hybrid Music Clustering with VAE (Audio + Lyrics)

This repo implements an unsupervised pipeline:
1) build features from audio-descriptor columns + lyric text,
2) train a (beta-)VAE to learn a compact latent space,
3) cluster in feature space (easy) or latent space (medium/hard),
4) export metrics + UMAP plots.

## Folder layout
```
project/
  data/
    audio/
    lyrics/
  notebooks/
    exploratory.ipynb
  src/
    vae.py
    dataset.py
    clustering.py
    evaluation.py
  results/
    latent_visualization/
    clustering_metrics.csv
  README.md
  requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Data formats (pick one)

### Option A: single CSV (recommended)
Place a CSV at `data/lyrics/tracks.csv` with:
- audio feature columns (numeric), e.g. danceability, energy, tempo, ...
- a `lyrics` column (text)
- optional `genre` (used only for reporting / ARI-NMI if provided)

Example:
```csv
danceability,energy,tempo,duration_ms,lyrics,genre
0.5,0.7,120,180000,"some lyric text",rock
```

### Option B: precomputed NPZ
Place `data/audio/features.npz` containing:
- `X` : (N,D) float32 features
- optional `y` : (N,) integer labels
- optional `meta` : dict-like (track ids)

## Run

### Easy: clustering on raw features
```bash
python -m src.clustering --mode easy --input_csv data/lyrics/tracks.csv --out_dir results/easy
```

### Medium: VAE latent clustering (audio+lyrics)
```bash
python -m src.clustering --mode medium --input_csv data/lyrics/tracks.csv --out_dir results/medium
```

### Hard: beta-VAE latent clustering (stronger regularization)
```bash
python -m src.clustering --mode hard --input_csv data/lyrics/tracks.csv --out_dir results/hard --beta 4.0
```

Outputs:
- `results/<mode>/metrics/clustering_metrics.csv`
- `results/<mode>/latent_visualization/umap_kmeans.png` (and others)

## Notes
- Works on CPU (no GPU required).
- If your CSV is large, use `--max_items` to cap processing.
