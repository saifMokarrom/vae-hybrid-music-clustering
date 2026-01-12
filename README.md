# Hybrid Music Clustering with VAE (Audio + Lyrics)

This repo implements an unsupervised pipeline:
1) build features from audio-descriptor columns + lyric text,
2) train a (beta-)VAE to learn a compact latent space,
3) cluster in feature space  or latent space ,
4) export metrics + UMAP plots.


## Setup
```bash
pip install -r requirements.txt
```


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

