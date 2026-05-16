"""
Módulo de utilidades para análisis de autoencoder y clustering de resiliencia climática.

Submódulos:
- plots: Funciones de visualización de espacio latente y clustering
- clustering: Funciones de clustering y métricas de resiliencia territorial
"""

from .plots import (
    plot_latent_space_overview,
    plot_base_vs_target_comparison,
    plot_latent_panel,
    plot_cluster_transition_matrix,
    plot_spatial_comparison_inline,
    plot_spatial_comparisons_all_ssp
)

from .clustering import (
    compute_inv_covs_per_cluster,
    prepare_clustering_data,
    compute_cluster_resilience_metrics,
    cluster_and_measure_resilience,
    get_cluster_summary
)
from .cluster_metrics import (
    cluster_size_stats,
    compute_internal_metrics,
    evaluate_kmeans_grid,
    compute_pairwise_label_metrics,
    summarize_scenarios,
)

__all__ = [
    # Plots
    'plot_latent_space_overview',
    'plot_base_vs_target_comparison',
    'plot_latent_panel',
    'plot_cluster_transition_matrix',
    'plot_spatial_comparison_inline',
    'plot_spatial_comparisons_all_ssp',
    # Clustering
    'compute_inv_covs_per_cluster',
    'prepare_clustering_data',
    'compute_cluster_resilience_metrics',
    'cluster_and_measure_resilience',
    'get_cluster_summary',
    # Clustering metrics
    'cluster_size_stats',
    'compute_internal_metrics',
    'evaluate_kmeans_grid',
    'compute_pairwise_label_metrics',
    'summarize_scenarios',
]
