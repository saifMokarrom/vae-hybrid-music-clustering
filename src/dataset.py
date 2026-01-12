from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer


AUDIO_DEFAULT_COLS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]


@dataclass
class BuildConfig:
    input_csv: str
    out_npz: str
    audio_cols: Optional[List[str]] = None
    text_col: str = "lyrics"
    label_col: Optional[str] = "genre"
    max_items: Optional[int] = None
    sbert_model: str = "all-MiniLM-L6-v2"
    cache_dir: str = "data/lyrics/.cache"
    seed: int = 42


def _safe_float_frame(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    x = df[cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))
    return x.to_numpy(dtype=np.float32)


def build_features(cfg: BuildConfig) -> Dict[str, np.ndarray]:
    df = pd.read_csv(cfg.input_csv)
    if cfg.max_items is not None and len(df) > cfg.max_items:
        df = df.sample(cfg.max_items, random_state=cfg.seed).reset_index(drop=True)

    audio_cols = cfg.audio_cols or [c for c in AUDIO_DEFAULT_COLS if c in df.columns]
    if not audio_cols:
        raise ValueError("No audio columns found. Provide --audio_cols or add numeric audio columns to the CSV.")

    x_audio = _safe_float_frame(df, audio_cols)

    lyrics = df[cfg.text_col].fillna("").astype(str).tolist()
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache_path = os.path.join(cfg.cache_dir, f"sbert_{cfg.sbert_model.replace('/', '_')}_{len(lyrics)}.joblib")

    if os.path.exists(cache_path):
        x_lyrics = joblib.load(cache_path)
    else:
        model = SentenceTransformer(cfg.sbert_model)
        x_lyrics = model.encode(lyrics, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        x_lyrics = x_lyrics.astype(np.float32)
        joblib.dump(x_lyrics, cache_path)

    scaler = StandardScaler()
    x_audio = scaler.fit_transform(x_audio).astype(np.float32)

    x = np.concatenate([x_audio, x_lyrics], axis=1).astype(np.float32)

    y = None
    if cfg.label_col and cfg.label_col in df.columns:
        y = df[cfg.label_col].fillna("unknown").astype(str).to_numpy()

    meta = {
        "audio_cols": audio_cols,
        "text_col": cfg.text_col,
        "label_col": cfg.label_col if cfg.label_col in df.columns else None,
    }

    os.makedirs(os.path.dirname(cfg.out_npz), exist_ok=True)
    if y is None:
        np.savez(cfg.out_npz, X=x, meta=meta)
    else:
        np.savez(cfg.out_npz, X=x, y=y, meta=meta)

    return {"X": x, "y": y, "meta": meta}


def load_npz(npz_path: str) -> Dict[str, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    out = {"X": z["X"].astype(np.float32)}
    if "y" in z.files:
        out["y"] = z["y"]
    if "meta" in z.files:
        out["meta"] = z["meta"].item() if hasattr(z["meta"], "item") else z["meta"]
    return out


def split_train_val(X: np.ndarray, val_split: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    x_tr, x_va = train_test_split(X, test_size=val_split, random_state=seed, shuffle=True)
    return x_tr.astype(np.float32), x_va.astype(np.float32)
