from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def clustering_metrics(X: np.ndarray, labels: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
            "ari": float("nan"),
            "nmi": float("nan"),
        }

    out = {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "ari": float("nan"),
        "nmi": float("nan"),
    }

    if y_true is not None:
        out["ari"] = float(adjusted_rand_score(y_true, labels))
        out["nmi"] = float(normalized_mutual_info_score(y_true, labels))

    return out
