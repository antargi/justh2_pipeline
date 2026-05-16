"""
Módulo de utilidades para visualización de espacio latente y clustering de resiliencia climática.

Funciones principales:
- plot_latent_space_overview: Visualización PCA general de latentes BASE y TARGET
- plot_base_vs_target_comparison: Comparación detallada con clusters y desplazamientos
- plot_latent_panel: Panel individual con elipses y centroides
- plot_cluster_transition_matrix: Matriz de transición entre clusters

Uso desde notebook:
    from utils.plots import plot_latent_space_overview, plot_base_vs_target_comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_latent_space_overview(
    model_key,
    latents_dict,
    n_per_scenario,
    seed=42,
    figsize=(16, 4)
):
    """
    Visualiza proyección PCA general del espacio latente (BASE + 3 TARGETS).
    
    Args:
        model_key: Identificador del modelo (e.g., "AE_128")
        latents_dict: Diccionario con latentes {"base", "T245", "T370", "T585"}
        n_per_scenario: Número de muestras por escenario
        seed: Semilla aleatoria para PCA
        figsize: Tamaño de la figura
    """
    print(f"\nVISUALIZACIÓN DEL ESPACIO LATENTE — {model_key}")
    
    # Labels para diferenciar escenarios BASE
    scenario_labels = (
        ["B245"] * n_per_scenario
        + ["B370"] * n_per_scenario
        + ["B585"] * n_per_scenario
    )
    
    color_map = {"B245": "tab:blue", "B370": "tab:orange", "B585": "tab:green"}
    future_labels = ["T245", "T370", "T585"]
    
    z_base = latents_dict["base"]
    pca = PCA(n_components=2, random_state=seed)
    z_base_2d = pca.fit_transform(z_base)
    
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True)
    
    # Panel BASE
    axes[0].scatter(
        z_base_2d[:, 0],
        z_base_2d[:, 1],
        c=[color_map[label] for label in scenario_labels],
        s=8,
        alpha=0.6,
    )
    axes[0].set_title(f"{model_key}: Base")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    
    # Paneles TARGET
    for ax, future in zip(axes[1:], future_labels):
        z_future = latents_dict[future]
        z_future_2d = pca.transform(z_future)
        ax.scatter(
            z_future_2d[:, 0],
            z_future_2d[:, 1],
            c="tab:red",
            s=8,
            alpha=0.6,
        )
        ax.set_title(f"{model_key}: {future}")
        ax.set_xlabel("PC1")
    
    explained_var = pca.explained_variance_ratio_.sum() * 100
    plt.suptitle(
        f"Proyección PCA del espacio latente ({model_key}) — Var. explicada: {explained_var:.1f}%"
    )
    plt.tight_layout()
    plt.show()


def _get_mercator_coords(coords_df):
    """Project lat/lon coordinates to Web Mercator (EPSG:3857) for basemap alignment."""
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xs, ys = transformer.transform(coords_df["lon"].values, coords_df["lat"].values)
        return xs, ys
    except ImportError:
        print("Warning: pyproj not available, using raw lon/lat (basemap may not align)")
        return coords_df["lon"].values, coords_df["lat"].values
def _infer_grid_resolution(n_points):
    """Infer appropriate interpolation grid resolution based on number of points."""
    if n_points < 100:
        return 50
    elif n_points < 500:
        return 80
    elif n_points < 1000:
        return 100
    else:
        return 150


def plot_spatial_comparison_inline(
    labels_base,
    labels_target,
    coords_df,
    title_base,
    title_target,
    suptitle,
    alpha=0.75
):
    """
    Grafica dos mapas espaciales lado a lado comparando clusters BASE vs TARGET.
    
    Utiliza KNN para interpolar clusters en una grilla regular y proyecta
    sobre un mapa base (basemap de contextily si está disponible).
    
    Args:
        labels_base: Labels de cluster BASE (N,)
        labels_target: Labels de cluster TARGET (N,)
        coords_df: DataFrame con columnas 'lat', 'lon' de las coordenadas
        title_base: Título del panel izquierdo (BASE)
        title_target: Título del panel derecho (TARGET)
        suptitle: Título general de la figura
        alpha: Transparencia de la capa de clusters (default: 0.75)
    """
    from sklearn.neighbors import KNeighborsClassifier
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.cm import ScalarMappable
    
    labels_arr_base = np.asarray(labels_base)
    valid_mask_base = ~pd.isna(labels_arr_base)
    unique_vals_base = np.sort(np.unique(labels_arr_base[valid_mask_base]))
    
    labels_arr_target = np.asarray(labels_target)
    valid_mask_target = ~pd.isna(labels_arr_target)
    unique_vals_target = np.sort(np.unique(labels_arr_target[valid_mask_target]))
    
    all_unique_vals = np.sort(np.unique(np.concatenate([unique_vals_base, unique_vals_target])))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    color_palette = plt.get_cmap('tab20', 20)
    cluster_colors = []
    for val in all_unique_vals:
        color_idx = int(val) % 20
        cluster_colors.append(color_palette(color_idx))
    discrete_cmap = ListedColormap(cluster_colors)
    
    val_to_idx = {val: idx for idx, val in enumerate(all_unique_vals)}
    
    try:
        xs, ys = _get_mercator_coords(coords_df)
        grid_res = _infer_grid_resolution(valid_mask_base.sum())
        grid_x = np.linspace(xs.min(), xs.max(), grid_res)
        grid_y = np.linspace(ys.min(), ys.max(), grid_res)
        GX, GY = np.meshgrid(grid_x, grid_y)
        extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
        
        coords_base = np.column_stack([xs[valid_mask_base], ys[valid_mask_base]])
        targets_base = labels_arr_base[valid_mask_base]
        int_targets_base = np.vectorize(val_to_idx.get)(targets_base)
        
        n_neighbors = max(1, min(len(int_targets_base), int(np.sqrt(len(int_targets_base)))))
        clf_base = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        clf_base.fit(coords_base, int_targets_base)
        
        grid_points = np.column_stack([GX.ravel(), GY.ravel()])
        pred_base = clf_base.predict(grid_points).reshape(GX.shape)
        
        axes[0].set_xlim(extent[0], extent[1])
        axes[0].set_ylim(extent[2], extent[3])
        
        try:
            import contextily as ctx
            ctx.add_basemap(
                axes[0],
                source=ctx.providers.CartoDB.Positron,
                crs="EPSG:3857",
                alpha=1.0,
                attribution_size=6
            )
        except:
            pass
        
        boundaries = np.arange(len(all_unique_vals) + 1) - 0.5
        norm = BoundaryNorm(boundaries, discrete_cmap.N)
        
        axes[0].imshow(
            pred_base,
            extent=extent,
            origin="lower",
            cmap=discrete_cmap,
            norm=norm,
            alpha=alpha,
            zorder=3
        )
        axes[0].set_axis_off()
        axes[0].set_title(title_base, fontsize=11, pad=10)
        
        coords_target = np.column_stack([xs[valid_mask_target], ys[valid_mask_target]])
        targets_target = labels_arr_target[valid_mask_target]
        int_targets_target = np.vectorize(val_to_idx.get)(targets_target)
        
        clf_target = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        clf_target.fit(coords_target, int_targets_target)
        pred_target = clf_target.predict(grid_points).reshape(GX.shape)
        
        axes[1].set_xlim(extent[0], extent[1])
        axes[1].set_ylim(extent[2], extent[3])
        
        try:
            ctx.add_basemap(
                axes[1],
                source=ctx.providers.CartoDB.Positron,
                crs="EPSG:3857",
                alpha=1.0,
                attribution_size=6
            )
        except:
            pass
        
        axes[1].imshow(
            pred_target,
            extent=extent,
            origin="lower",
            cmap=discrete_cmap,
            norm=norm,
            alpha=alpha,
            zorder=3
        )
        axes[1].set_axis_off()
        axes[1].set_title(title_target, fontsize=11, pad=10)
        
        mappable = ScalarMappable(norm=norm, cmap=discrete_cmap)
        cbar = fig.colorbar(
            mappable,
            ax=axes,
            fraction=0.03,
            pad=0.02,
            ticks=np.arange(len(all_unique_vals))
        )
        cbar.set_ticklabels([str(int(val)) for val in all_unique_vals])
        cbar.set_label("Cluster ID", fontsize=10)
        
        fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
    except Exception as err:
        print(f"Error en plot_spatial_comparison: {err}")
        plt.close(fig)


def plot_spatial_comparisons_all_ssp(
    model_key,
    clustering_results,
    coords_df,
    n_per_scenario,
    scenarios=["245", "370", "585"],
    alpha=0.75
):
    """
    Genera comparaciones espaciales BASE vs TARGET para todos los escenarios SSP.
    
    Args:
        model_key: Identificador del modelo (e.g., "AE_128")
        clustering_results: Diccionario con resultados de clustering
        coords_df: DataFrame con columnas 'lat', 'lon'
        n_per_scenario: Número de muestras por escenario
        scenarios: Lista de escenarios SSP a procesar (default: ["245", "370", "585"])
        alpha: Transparencia de la capa de clusters
        
    Returns:
        dict: Diccionario con estadísticas de transición por escenario
    """
    print(f"\nComparación BASE → TARGET — {model_key}")
    
    results = clustering_results
    labels_base = results["labels_base"]
    
    labels_targets = {ssp: results[f"labels_T{ssp}"] for ssp in scenarios}
    
    # Separar BASE por escenarios
    labels_base_splits = {}
    for idx, ssp in enumerate(scenarios):
        start = idx * n_per_scenario
        end = (idx + 1) * n_per_scenario
        labels_base_splits[ssp] = labels_base[start:end]
    
    print(f"\nClusters únicos: {np.unique(labels_base)}")
    print(f"Mostrando {len(scenarios)} comparaciones BASE → TARGET (una por trayectoria SSP)\n")
    
    transition_stats = {}
    
    for ssp in scenarios:
        labels_B = labels_base_splits[ssp]
        labels_T = labels_targets[ssp]
        
        changed = (labels_T != labels_B).sum()
        pct_changed = (changed / len(labels_T)) * 100
        
        print(f"SSP{ssp}: {changed}/{len(labels_T)} píxeles transicionan ({pct_changed:.1f}%)")
        
        plot_spatial_comparison_inline(
            labels_B,
            labels_T,
            coords_df,
            title_base="BASE (2020-2029)",
            title_target="TARGET (2090-2100)",
            suptitle=f"SSP{ssp} — {model_key}",
            alpha=alpha
        )
        
        transition_stats[ssp] = {
            "changed": changed,
            "total": len(labels_T),
            "pct_changed": pct_changed
        }
    
    print(f"\nRESUMEN: Transiciones de cluster BASE → TARGET")
    for ssp, stats in transition_stats.items():
        print(f"  SSP{ssp}: {stats['pct_changed']:.1f}% de píxeles cambian de cluster")
    print(f"\n  → Mayor % = mayor reordenamiento espacial del perfil climático")
    
    return transition_stats


def plot_latent_panel(ax, z_2d, labels, centroids_base, title, k_clusters, 
                      show_legend=False, centroids_target=None, show_displacement=False):
    """
    Grafica un panel del espacio latente con clusters, centroides y elipses envolventes.
    
    Args:
        ax: Axes de matplotlib
        z_2d: Coordenadas 2D de los puntos (N, 2)
        labels: Labels de cluster (N,)
        centroids_base: Centroides BASE en 2D (k_clusters, 2)
        title: Título del panel
        k_clusters: Número de clusters
        show_legend: Si mostrar leyenda
        centroids_target: Centroides TARGET en 2D (k_clusters, 2) - solo para panel TARGET
        show_displacement: Si dibujar líneas de desplazamiento BASE→TARGET
    """
    from matplotlib.patches import Ellipse
    
    color_palette = plt.get_cmap('tab20', 20)
    unique_clusters = np.unique(labels)
    
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        n_points = mask.sum()
        
        if n_points == 0:
            continue
        
        points = z_2d[mask]
        color_idx = int(cluster_id) % 20
        cluster_color = color_palette(color_idx)
        
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=[cluster_color],
            label=f"Cluster {cluster_id}",
            s=20,
            alpha=0.65,
            edgecolor="k",
            linewidth=0.3,
            zorder=3
        )
        
        if n_points >= 3:
            mean_x = points[:, 0].mean()
            mean_y = points[:, 1].mean()
            
            cov = np.cov(points.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * 2.45 * np.sqrt(eigenvalues)
            
            ellipse = Ellipse(
                xy=(mean_x, mean_y),
                width=width,
                height=height,
                angle=angle,
                facecolor=cluster_color,
                alpha=0.15,
                edgecolor=cluster_color,
                linewidth=1.5,
                linestyle='--',
                zorder=1
            )
            ax.add_patch(ellipse)
    
    if centroids_target is not None and show_displacement:
        for cluster_id in unique_clusters:
            centroid_base = centroids_base[cluster_id]
            centroid_target = centroids_target[cluster_id]
            
            if np.isnan(centroid_base).any() or np.isnan(centroid_target).any():
                continue
            
            ax.plot(
                [centroid_base[0], centroid_target[0]],
                [centroid_base[1], centroid_target[1]],
                'k--',
                linewidth=1.5,
                alpha=0.5,
                zorder=8
            )
            
            ax.annotate(
                '',
                xy=(centroid_target[0], centroid_target[1]),
                xytext=(centroid_base[0], centroid_base[1]),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black', alpha=0.6),
                zorder=8
            )
            
            ax.scatter(
                centroid_base[0],
                centroid_base[1],
                c="gray",
                marker="X",
                s=180,
                edgecolor="white",
                linewidth=2,
                alpha=0.7,
                zorder=9,
                label="Centroide BASE" if cluster_id == unique_clusters[0] else ""
            )
            
            ax.scatter(
                centroid_target[0],
                centroid_target[1],
                c="red",
                marker="X",
                s=280,
                edgecolor="white",
                linewidth=2.5,
                zorder=10,
                label="Centroide TARGET" if cluster_id == unique_clusters[0] else ""
            )
            
            distance = np.linalg.norm(centroid_target - centroid_base)
            mid_x = (centroid_base[0] + centroid_target[0]) / 2
            mid_y = (centroid_base[1] + centroid_target[1]) / 2
            
            ax.text(
                mid_x, mid_y,
                f'{distance:.2f}',
                fontsize=7,
                color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
                ha='center',
                va='center',
                zorder=11
            )
    else:
        for cluster_id in unique_clusters:
            centroid = centroids_base[cluster_id]
            ax.scatter(
                centroid[0],
                centroid[1],
                c="black",
                marker="X",
                s=250,
                edgecolor="white",
                linewidth=2.5,
                zorder=10
            )
    
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(alpha=0.25, linestyle="--")
    if show_legend:
        ax.legend(loc="best", fontsize=7, framealpha=0.9, ncol=2)


def plot_cluster_transition_matrix(labels_base, labels_target, k_clusters, scenario_name):
    """
    Calcula y muestra matriz de transición entre clusters BASE y TARGET.
    
    Args:
        labels_base: Labels de cluster BASE (N,)
        labels_target: Labels de cluster TARGET (N,)
        k_clusters: Número de clusters
        scenario_name: Nombre del escenario (e.g., "SSP245")
    
    Returns:
        pd.DataFrame: Matriz de transición
    """
    transition = np.zeros((k_clusters, k_clusters), dtype=int)
    for i in range(k_clusters):
        mask = labels_base == i
        for j in range(k_clusters):
            transition[i, j] = (labels_target[mask] == j).sum()
    
    df_transition = pd.DataFrame(
        transition,
        index=[f"B_C{i}" for i in range(k_clusters)],
        columns=[f"T_C{j}" for j in range(k_clusters)]
    )
    
    print(f"\n{scenario_name}:")
    print(df_transition)
    
    print(f"\n  Retención de cluster (diagonal):")
    for i in range(k_clusters):
        total = transition[i, :].sum()
        retained = transition[i, i]
        pct = retained / total * 100 if total > 0 else 0
        print(f"    Cluster {i}: {retained}/{total} píxeles ({pct:.1f}%)")
    
    return df_transition


def plot_base_vs_target_comparison(
    model_key,
    latents_dict,
    clustering_results,
    n_per_scenario,
    k_clusters,
    scenarios=["245", "370", "585"],
    seed=42
):
    """
    Genera visualizaciones comparativas BASE vs TARGET para todos los escenarios SSP.
    
    Args:
        model_key: Identificador del modelo (e.g., "AE_128")
        latents_dict: Diccionario con latentes {"base", "T245", "T370", "T585"}
        clustering_results: Diccionario con resultados de clustering
        n_per_scenario: Número de muestras por escenario
        k_clusters: Número de clusters
        scenarios: Lista de escenarios SSP a procesar (default: ["245", "370", "585"])
        seed: Semilla aleatoria para PCA
    """
    print(f"\nClusters en espacio latente — {model_key}")
    
    results = clustering_results
    scaler = results["scaler"]
    
    # Obtener y escalar latentes
    z_base = latents_dict["base"]
    z_base_scaled = scaler.transform(z_base)
    
    z_targets_scaled = {}
    for ssp in scenarios:
        z_targets_scaled[ssp] = scaler.transform(latents_dict[f"T{ssp}"])
    
    # Labels
    labels_base = results["labels_base"]
    labels_targets = {ssp: results[f"labels_T{ssp}"] for ssp in scenarios}
    
    # Separar BASE por escenarios
    base_splits = {}
    labels_base_splits = {}
    for idx, ssp in enumerate(scenarios):
        start = idx * n_per_scenario
        end = (idx + 1) * n_per_scenario
        base_splits[ssp] = z_base_scaled[start:end]
        labels_base_splits[ssp] = labels_base[start:end]
    
    # PCA
    pca = PCA(n_components=2, random_state=seed)
    z_base_2d = pca.fit_transform(z_base_scaled)
    
    z_targets_2d = {}
    for ssp in scenarios:
        z_targets_2d[ssp] = pca.transform(z_targets_scaled[ssp])
    
    # Separar BASE 2D por escenarios
    base_splits_2d = {}
    for idx, ssp in enumerate(scenarios):
        start = idx * n_per_scenario
        end = (idx + 1) * n_per_scenario
        base_splits_2d[ssp] = z_base_2d[start:end]
    
    print(f"Varianza explicada por PC1+PC2: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    
    # Iterar por cada escenario
    for ssp in scenarios:
        print(f"\nSSP{ssp}: BASE vs TARGET")
        
        # Centroides BASE específicos del escenario
        centroids_base_2d = np.array([
            base_splits_2d[ssp][labels_base_splits[ssp] == cid].mean(axis=0) 
            for cid in range(k_clusters)
        ])
        
        # Centroides TARGET
        centroids_target_2d = np.array([
            z_targets_2d[ssp][labels_targets[ssp] == cid].mean(axis=0) 
            for cid in range(k_clusters)
        ])
        
        # Crear figura
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        
        # Panel BASE
        plot_latent_panel(
            axes[0], 
            base_splits_2d[ssp], 
            labels_base_splits[ssp], 
            centroids_base_2d, 
            "BASE (2020-2029)",
            k_clusters,
            show_legend=True,
            centroids_target=None,
            show_displacement=False
        )
        axes[0].set_ylabel("PC2", fontsize=10)
        axes[0].set_xlabel("PC1", fontsize=10)
        
        # Panel TARGET
        plot_latent_panel(
            axes[1], 
            z_targets_2d[ssp], 
            labels_targets[ssp], 
            centroids_base_2d, 
            "TARGET (2090-2100)",
            k_clusters,
            show_legend=True,
            centroids_target=centroids_target_2d,
            show_displacement=True
        )
        axes[1].set_xlabel("PC1", fontsize=10)
        
        fig.suptitle(
            f"SSP{ssp}: Evolución de clusters en espacio latente (PCA) — {model_key}", 
            fontsize=13, fontweight="bold", y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    # Matrices de transición
    print(f"\nMATRICES DE TRANSICIÓN BASE → TARGET")
    
    transition_matrices = {}
    for ssp in scenarios:
        df_transition = plot_cluster_transition_matrix(
            labels_base_splits[ssp],
            labels_targets[ssp],
            k_clusters,
            f"SSP{ssp} (B{ssp} → T{ssp})"
        )
        transition_matrices[ssp] = df_transition
    
    return transition_matrices

