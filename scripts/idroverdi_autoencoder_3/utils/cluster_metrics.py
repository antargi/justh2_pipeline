"""
Métricas de validación y análisis de clustering.

Incluye:
- Métricas internas (silhouette, Davies–Bouldin, Calinski–Harabasz)
- Estadísticas de tamaño de clúster
- Búsqueda rápida de K para KMeans
- Estabilidad entre particiones (ARI, AMI, NMI, Fowlkes–Mallows)
"""

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def _safe_internal_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: Optional[int],
    random_state: int,
) -> Tuple[float, float, float]:
    """Calcula métricas internas manejando casos degenerados."""
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or X.shape[0] < 2:
        return np.nan, np.nan, np.nan

    sil = silhouette_score(
        X,
        labels,
        sample_size=sample_size,
        random_state=random_state,
    )
    dbi = davies_bouldin_score(X, labels)
    chi = calinski_harabasz_score(X, labels)
    return float(sil), float(dbi), float(chi)


def cluster_size_stats(labels: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Obtiene distribución de tamaños de clúster.

    Returns:
        size_df: conteo y porcentaje por clúster
        summary: métricas de tamaño (min, max, mediana, std)
    """
    labels = np.asarray(labels)
    values, counts = np.unique(labels, return_counts=True)
    total = counts.sum()

    size_df = pd.DataFrame(
        {
            "cluster_id": values,
            "count": counts,
            "pct": counts / total * 100,
        }
    ).sort_values("cluster_id")

    summary = {
        "min_cluster_size": float(counts.min()),
        "max_cluster_size": float(counts.max()),
        "median_cluster_size": float(np.median(counts)),
        "std_cluster_size": float(np.std(counts)),
    }

    return size_df, summary


def compute_internal_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Calcula métricas internas básicas para un clustering.

    Args:
        X: embeddings (n_samples, n_features)
        labels: asignación de clúster por muestra
        sample_size: submuestreo opcional para silhouette
        random_state: semilla para silhouette si se usa sample_size
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    n_samples = X.shape[0]
    n_clusters = len(np.unique(labels))
    sil, dbi, chi = _safe_internal_metrics(X, labels, sample_size, random_state)
    _, size_summary = cluster_size_stats(labels)

    metrics = {
        "n_samples": float(n_samples),
        "n_clusters": float(n_clusters),
        "silhouette": sil,
        "davies_bouldin": dbi,
        "calinski_harabasz": chi,
    }
    metrics.update(size_summary)
    return metrics


def evaluate_kmeans_grid(
    X: np.ndarray,
    k_values: Iterable[int],
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300,
    sample_size: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Evalúa KMeans en un grid de K con métricas internas.

    Args:
        X: embeddings ya estandarizados
        k_values: valores de K a probar
        random_state: semilla reproducible
        n_init: inicializaciones de KMeans
        max_iter: iteraciones máximas
        sample_size: submuestreo opcional para silhouette
        verbose: imprimir progreso
    """
    X = np.asarray(X)
    rows: List[Dict] = []

    for k in sorted(set(k_values)):
        if verbose:
            print(f"→ K={k}")

        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        labels = km.fit_predict(X)
        sil, dbi, chi = _safe_internal_metrics(X, labels, sample_size, random_state)

        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "n_iter": float(km.n_iter_),
                "silhouette": sil,
                "davies_bouldin": dbi,
                "calinski_harabasz": chi,
            }
        )

    return pd.DataFrame(rows).sort_values("k")


def compute_pairwise_label_metrics(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> Dict[str, float]:
    """
    Mide similitud/estabilidad entre dos particiones de clustering.

    Usa ARI, AMI, NMI y Fowlkes–Mallows más coincidencia exacta.
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    if labels_a.shape[0] != labels_b.shape[0]:
        raise ValueError("labels_a y labels_b deben tener el mismo número de muestras")

    metrics = {
        "ari": adjusted_rand_score(labels_a, labels_b),
        "ami": adjusted_mutual_info_score(labels_a, labels_b),
        "nmi": normalized_mutual_info_score(labels_a, labels_b),
        "fowlkes_mallows": fowlkes_mallows_score(labels_a, labels_b),
        "match_ratio": float((labels_a == labels_b).mean()),
    }
    return metrics


def summarize_scenarios(
    latents: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Resume métricas internas y estabilidad entre múltiples escenarios.

    Args:
        latents: dict con escenario → embeddings
        labels: dict con escenario → labels (misma longitud que embeddings)
        sample_size: submuestreo opcional para silhouette
        random_state: semilla para métricas que lo usen

    Returns:
        {"internal": DataFrame, "stability": DataFrame}
    """
    internal_rows: List[Dict] = []
    for scenario, Z in latents.items():
        scenario_metrics = compute_internal_metrics(
            Z,
            labels[scenario],
            sample_size=sample_size,
            random_state=random_state,
        )
        scenario_metrics["scenario"] = scenario
        internal_rows.append(scenario_metrics)

    stability_rows: List[Dict] = []
    for a, b in combinations(labels.keys(), 2):
        pair_metrics = compute_pairwise_label_metrics(labels[a], labels[b])
        pair_metrics["pair"] = f"{a} vs {b}"
        stability_rows.append(pair_metrics)

    internal_df = pd.DataFrame(internal_rows).set_index("scenario").sort_index()
    stability_df = pd.DataFrame(stability_rows).set_index("pair").sort_index()

    return {"internal": internal_df, "stability": stability_df}
