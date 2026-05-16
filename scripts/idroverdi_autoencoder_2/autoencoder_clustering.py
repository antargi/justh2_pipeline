# %% [markdown]
# # 07_experiments.ipynb — Experimentos de Resiliencia (AE Latente)
# 
# Este cuaderno implementa **5 experimentos** sobre el autoencoder (AE) y el espacio latente para evaluar **resiliencia territorial** bajo escenarios SSP.
# 
# **Experimentos**
# 1. Representación base (reconstrucción)
# 2. Clustering (KMeans / DBSCAN)
# 3. Sensibilidad a k (Silhouette / Davies–Bouldin)
# 4. Ablaciones (exclusión de variables)
# 5. Validación cruzada espacial
# 

# %%

import random
import os, re, pickle
import numpy as np
import pandas as pd
import xarray as xr

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, r2_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BASE_DIR = "/home/aninotna/magister/tesis/justh2_pipeline"
DATA_DIR = os.path.join(BASE_DIR, "data/autoencoder_tensors")
MODE = "test"

PATH_SSP245 = os.path.join(DATA_DIR, f"tensors_ssp245_splits_{MODE}.npz")
PATH_SSP370 = os.path.join(DATA_DIR, f"tensors_ssp370_splits_{MODE}.npz")
PATH_SSP585 = os.path.join(DATA_DIR, f"tensors_ssp585_splits_{MODE}.npz")

PATH_SSP245_ORIG = os.path.join(DATA_DIR, f"tensors_ssp245_splits_{MODE}_ORIGINAL.npz")
PATH_SSP370_ORIG = os.path.join(DATA_DIR, f"tensors_ssp370_splits_{MODE}_ORIGINAL.npz")
PATH_SSP585_ORIG = os.path.join(DATA_DIR, f"tensors_ssp585_splits_{MODE}_ORIGINAL.npz")

PATH_METADATA = os.path.join(DATA_DIR, f"metadata_{MODE}.pkl")
PATH_FEATURE_NAMES = os.path.join(DATA_DIR, f"feature_names_{MODE}.csv")

print("✓ Config cargada")


# %%

def load_npz(path):
    d = np.load(path)
    X = np.vstack([d["X_train"], d["X_val"], d["X_test"]])
    idx = np.concatenate([d["train_idx"], d["val_idx"], d["test_idx"]])
    return X, idx

def load_npz_orig(path):
    d = np.load(path)
    X = np.vstack([d["X_train_original"], d["X_val_original"], d["X_test_original"]])
    idx = np.concatenate([d["train_idx"], d["val_idx"], d["test_idx"]])
    return X, idx

X245_norm, idx245 = load_npz(PATH_SSP245)
X370_norm, idx370 = load_npz(PATH_SSP370)
X585_norm, idx585 = load_npz(PATH_SSP585)

X245_orig, _ = load_npz_orig(PATH_SSP245_ORIG)
X370_orig, _ = load_npz_orig(PATH_SSP370_ORIG)
X585_orig, _ = load_npz_orig(PATH_SSP585_ORIG)

feature_names = pd.read_csv(PATH_FEATURE_NAMES)["feature_name"].tolist()

with open(PATH_METADATA, "rb") as f:
    meta = pickle.load(f)

splits = meta["splits"]
mask = meta["mask"]
lat = meta["spatial_info"]["lat"]
lon = meta["spatial_info"]["lon"]

lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
flat_lat = lat_grid[mask]
flat_lon = lon_grid[mask]
all_idx = np.concatenate([splits["train_idx"], splits["val_idx"], splits["test_idx"]])
coords_df = pd.DataFrame({
    "lat": flat_lat[all_idx],
    "lon": flat_lon[all_idx],
    "pixel_id": np.arange(flat_lat[all_idx].shape[0])
})

print("Shapes norm:", X245_norm.shape, X370_norm.shape, X585_norm.shape)
print("Coords:", coords_df.shape)


# %%

DEC_BASE = [2020]
DEC_TARGET = [2040, 2060, 2080]

def cols_for_decades(names, decades):
    pat = re.compile(r"_decadal_(?:mean|max|min)_(\d{4})$")
    idx = []
    for j, n in enumerate(names):
        m = pat.search(n)
        if m and int(m.group(1)) in decades:
            idx.append(j)
    return idx

def non_decadal_idx(names):
    return [j for j, n in enumerate(names) if "_decadal_" not in n]

base_idx = cols_for_decades(feature_names, DEC_BASE)
tgt_idx  = cols_for_decades(feature_names, DEC_TARGET)
non_idx  = non_decadal_idx(feature_names)

def h2_cols(names, decades):
    pat = re.compile(r"calliope_h2_prod_ton_decadal_(?:mean|max|min)_(\d{4})$")
    idx = []
    for j, n in enumerate(names):
        m = pat.search(n)
        if m and int(m.group(1)) in decades:
            idx.append(j)
    if idx:
        return idx
    if "calliope_h2_prod_ton" in names:
        return [names.index("calliope_h2_prod_ton")]
    raise ValueError("No encontré columnas de H2.")

h2_base_idx = h2_cols(feature_names, DEC_BASE)
h2_tgt_idx  = h2_cols(feature_names, DEC_TARGET)

def take_h2(X, idxs):
    H = X[:, idxs]
    return H.mean(axis=1) if H.ndim == 2 and H.shape[1] > 1 else H.squeeze()


# %%
def cols_for_decade_and_stat(names, decades, stat="mean"):
    pat = re.compile(rf"_decadal_({stat})_(\d{{4}})$")  # ej: _decadal_mean_2020
    idx = []
    for j, n in enumerate(names):
        m = pat.search(n)
        if m and int(m.group(2)) in decades:
            idx.append(j)
    return idx

# => Índices por estadístico (decadal)
base_mean_idx = cols_for_decade_and_stat(feature_names, DEC_BASE, stat="mean")
base_max_idx  = cols_for_decade_and_stat(feature_names, DEC_BASE, stat="max")
base_min_idx  = cols_for_decade_and_stat(feature_names, DEC_BASE, stat="min")

tgt_mean_idx  = cols_for_decade_and_stat(feature_names, DEC_TARGET, stat="mean")

# => NO decanales se mantienen igual
non_idx  = [j for j, n in enumerate(feature_names) if "_decadal_" not in n]


def build_base_augmented(X_norm):
    """
    Construye BASE con:
      - medias decenales (DEC_BASE)
      - std_proxy_base ≈ (max - min) / 2   => capta 'spread' intra-decada
      - features no decenales
    """
    mean_base = X_norm[:, base_mean_idx]
    if base_max_idx and base_min_idx and len(base_max_idx) == len(base_mean_idx) == len(base_min_idx):
        std_proxy_base = 0.5 * (X_norm[:, base_max_idx] - X_norm[:, base_min_idx])
    else:
        # => Fallback si faltan max/min: std_proxy=0 (no rompe shapes)
        std_proxy_base = np.zeros_like(mean_base)
    non_features = X_norm[:, non_idx]
    return np.hstack([mean_base, std_proxy_base, non_features])


def build_target_augmented(X_norm):
    """
    Construye TARGET con:
      - media de las medias (promedio entre DEC_TARGET)
      - std real a través de décadas: std(mean_{2040,2060,2080})
      - features no decenales
    """
    # => Reorganiza target means: [N, V*D] -> [N, V, D] para std a través de décadas
    V = len(base_mean_idx)              # misma cantidad de variables climáticas que en base
    D = len(DEC_TARGET)                 # típicamente 3: 2040,2060,2080
    if len(tgt_mean_idx) != V * D:
        # => Si no calza, usa agregación simple (no debería pasar si nombres están OK)
        tgt_means = X_norm[:, tgt_mean_idx]
        mean_of_means = tgt_means.reshape(tgt_means.shape[0], -1).mean(axis=1, keepdims=True)
        std_across_dec = np.zeros((X_norm.shape[0], 1))
    else:
        tgt_means = X_norm[:, tgt_mean_idx].reshape(X_norm.shape[0], V, D)
        mean_of_means = tgt_means.mean(axis=2)          # [N, V]
        std_across_dec = tgt_means.std(axis=2, ddof=1)  # [N, V] => std entre décadas
    non_features = X_norm[:, non_idx]
    return np.hstack([mean_of_means, std_across_dec, non_features])


B245 = build_base_augmented(X245_norm)   # => ahora incluye mean_base + std_proxy_base + non
B370 = build_base_augmented(X370_norm)
B585 = build_base_augmented(X585_norm)

T245 = build_target_augmented(X245_norm) # => ahora incluye mean_target + std_target + non
T370 = build_target_augmented(X370_norm)
T585 = build_target_augmented(X585_norm)

X_BASE = np.vstack([B245, B370, B585])

print("BASE/TARGET shapes (AUG):", X_BASE.shape, T245.shape, T370.shape, T585.shape)
print(f"  base_mean_idx: {len(base_mean_idx)}, tgt_mean_idx: {len(tgt_mean_idx)}, non_idx: {len(non_idx)}")

# %%
def build_target_augmented_from_subset(X_subset, feat_names_subset):
    """
    Versión de build_target_augmented() que funciona con tensores filtrados (ablaciones).
    Recalcula los índices basándose en feat_names_subset en lugar de usar índices globales.
    
    Args:
        X_subset: tensor filtrado (n_samples, n_features_subset)
        feat_names_subset: lista de nombres de features en X_subset
    
    Returns:
        Tensor augmented con [mean_of_means, std_across_dec, non_features]
    """
    # Recalcular índices relativos al subset
    base_mean_idx_subset = cols_for_decade_and_stat(feat_names_subset, DEC_BASE, stat="mean")
    tgt_mean_idx_subset = cols_for_decade_and_stat(feat_names_subset, DEC_TARGET, stat="mean")
    non_idx_subset = [j for j, n in enumerate(feat_names_subset) if "_decadal_" not in n]
    
    V = len(base_mean_idx_subset)
    D = len(DEC_TARGET)
    
    if len(tgt_mean_idx_subset) != V * D:
        # Agregación simple si no calza
        tgt_means = X_subset[:, tgt_mean_idx_subset]
        mean_of_means = tgt_means.reshape(tgt_means.shape[0], -1).mean(axis=1, keepdims=True)
        std_across_dec = np.zeros((X_subset.shape[0], 1))
    else:
        tgt_means = X_subset[:, tgt_mean_idx_subset].reshape(X_subset.shape[0], V, D)
        mean_of_means = tgt_means.mean(axis=2)          # [N, V]
        std_across_dec = tgt_means.std(axis=2, ddof=1)  # [N, V]
    
    non_features = X_subset[:, non_idx_subset]
    return np.hstack([mean_of_means, std_across_dec, non_features])


# %%

def _get_mercator_coords():
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

def get_scenario_slice(scenario_name):
    """Get slice indices for a scenario from stacked array (B245, B370, B585)."""
    N_PER_SCENARIO = B245.shape[0]
    scenario_map = {
        "B245": slice(0, N_PER_SCENARIO),
        "B370": slice(N_PER_SCENARIO, 2 * N_PER_SCENARIO),
        "B585": slice(2 * N_PER_SCENARIO, 3 * N_PER_SCENARIO),
    }
    if scenario_name not in scenario_map:
        raise ValueError(f"Escenario '{scenario_name}' no reconocido. Usa 'B245', 'B370' o 'B585'.")
    return scenario_map[scenario_name]

def plot_spatial_scalar(values, title, cmap="magma", colorbar_label="Valor", 
                       alpha=0.65, vmin=None, vmax=None):
    """Plot continuous-value spatial heatmap with KNN interpolation and basemap."""
    values_arr = np.asarray(values)
    valid_mask = ~np.isnan(values_arr)
    
    if not valid_mask.any():
        print(f"Sin datos válidos para {title}")
        return
    
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    
    try:
        xs, ys = _get_mercator_coords()
        
        grid_res = _infer_grid_resolution(valid_mask.sum())
        grid_x = np.linspace(xs.min(), xs.max(), grid_res)
        grid_y = np.linspace(ys.min(), ys.max(), grid_res)
        GX, GY = np.meshgrid(grid_x, grid_y)
        
        coords = np.column_stack([xs[valid_mask], ys[valid_mask]])
        targets = values_arr[valid_mask]
        
        grid_points = np.column_stack([GX.ravel(), GY.ravel()])
        
        n_neighbors = max(3, min(len(targets), int(np.sqrt(len(targets)) * 1.5)))
        reg = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
        reg.fit(coords, targets)
        GZ = reg.predict(grid_points).reshape(GX.shape)
        
        extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
        try:
            import contextily as ctx
            ctx.add_basemap(
                ax,
                source=ctx.providers.CartoDB.Positron,
                crs="EPSG:3857",
                alpha=1.0,
                attribution_size=6,
            )
        except Exception as basemap_err:
            ax.text(
                0.02,
                0.02,
                f"Basemap no disponible: {basemap_err}",
                transform=ax.transAxes,
                fontsize=8,
                color="red",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )
        
        heat = ax.imshow(
            GZ,
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            zorder=3,
        )
        
        ax.set_axis_off()
        
        cbar = fig.colorbar(heat, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(colorbar_label)
        
        ax.set_title(title)
        fig.tight_layout()
        plt.show()
        return
        
    except Exception as err:
        ax.text(
            0.02,
            0.02,
            f"Heatmap no disponible ({err}); usando dispersión.",
            transform=ax.transAxes,
            fontsize=8,
            color="red",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )
        print(f"Error en plot_spatial_scalar: {err}")
    
    # Fallback: scatter plot
    sc = ax.scatter(
        coords_df["lon"],
        coords_df["lat"],
        c=values_arr,
        cmap=cmap,
        s=20,
        alpha=0.9,
        edgecolor="k",
        linewidth=0.2,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    plt.show()


# %%
def _get_mercator_coords():
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

def get_scenario_slice(scenario_name):
    """Get slice indices for a scenario from stacked array (B245, B370, B585)."""
    N_PER_SCENARIO = B245.shape[0]
    scenario_map = {
        "B245": slice(0, N_PER_SCENARIO),
        "B370": slice(N_PER_SCENARIO, 2 * N_PER_SCENARIO),
        "B585": slice(2 * N_PER_SCENARIO, 3 * N_PER_SCENARIO),
    }
    if scenario_name not in scenario_map:
        raise ValueError(f"Escenario '{scenario_name}' no reconocido. Usa 'B245', 'B370' o 'B585'.")
    return scenario_map[scenario_name]

def plot_spatial_scalar(values, title, cmap="magma", colorbar_label="Valor", 
                       alpha=0.65, vmin=None, vmax=None):
    """Plot continuous-value spatial heatmap with KNN interpolation and basemap."""
    values_arr = np.asarray(values)
    valid_mask = ~np.isnan(values_arr)
    
    if not valid_mask.any():
        print(f"Sin datos válidos para {title}")
        return
    
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    
    try:
        xs, ys = _get_mercator_coords()
        
        grid_res = _infer_grid_resolution(valid_mask.sum())
        grid_x = np.linspace(xs.min(), xs.max(), grid_res)
        grid_y = np.linspace(ys.min(), ys.max(), grid_res)
        GX, GY = np.meshgrid(grid_x, grid_y)
        
        coords = np.column_stack([xs[valid_mask], ys[valid_mask]])
        targets = values_arr[valid_mask]
        
        grid_points = np.column_stack([GX.ravel(), GY.ravel()])
        
        n_neighbors = max(3, min(len(targets), int(np.sqrt(len(targets)) * 1.5)))
        reg = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
        reg.fit(coords, targets)
        GZ = reg.predict(grid_points).reshape(GX.shape)
        
        extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
        try:
            import contextily as ctx
            ctx.add_basemap(
                ax,
                source=ctx.providers.CartoDB.Positron,
                crs="EPSG:3857",
                alpha=1.0,
                attribution_size=6,
            )
        except Exception as basemap_err:
            ax.text(
                0.02,
                0.02,
                f"Basemap no disponible: {basemap_err}",
                transform=ax.transAxes,
                fontsize=8,
                color="red",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )
        
        heat = ax.imshow(
            GZ,
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            zorder=3,
        )
        
        ax.set_axis_off()
        
        cbar = fig.colorbar(heat, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(colorbar_label)
        
        ax.set_title(title)
        fig.tight_layout()
        plt.show()
        return
        
    except Exception as err:
        ax.text(
            0.02,
            0.02,
            f"Heatmap no disponible ({err}); usando dispersión.",
            transform=ax.transAxes,
            fontsize=8,
            color="red",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )
        print(f"Error en plot_spatial_scalar: {err}")
    
    # Fallback: scatter plot
    sc = ax.scatter(
        coords_df["lon"],
        coords_df["lat"],
        c=values_arr,
        cmap=cmap,
        s=20,
        alpha=0.9,
        edgecolor="k",
        linewidth=0.2,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    plt.show()


# %%

def plot_spatial_categories(labels, title, cmap="tab10", alpha=0.75, s=20):
    labels_arr = np.asarray(labels)
    valid_mask = ~pd.isna(labels_arr)
    if not valid_mask.any():
        print(f"Sin datos válidos para {title}")
        return
    unique_vals = np.sort(np.unique(labels_arr[valid_mask]))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    try:
        xs, ys = _get_mercator_coords()
        grid_res = _infer_grid_resolution(valid_mask.sum())
        grid_x = np.linspace(xs.min(), xs.max(), grid_res)
        grid_y = np.linspace(ys.min(), ys.max(), grid_res)
        GX, GY = np.meshgrid(grid_x, grid_y)
        coords = np.column_stack([xs[valid_mask], ys[valid_mask]])
        targets = labels_arr[valid_mask]
        cat_to_int = {val: idx for idx, val in enumerate(unique_vals)}
        int_targets = np.vectorize(cat_to_int.get)(targets)
        grid_points = np.column_stack([GX.ravel(), GY.ravel()])
        n_neighbors = max(1, min(len(int_targets), int(np.sqrt(len(int_targets)))))
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        clf.fit(coords, int_targets)
        pred_int = clf.predict(grid_points).reshape(GX.shape)
        extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        try:
            import contextily as ctx
            ctx.add_basemap(
                ax,
                source=ctx.providers.CartoDB.Positron,
                crs="EPSG:3857",
                alpha=1.0,
                attribution_size=6,
            )
        except Exception as basemap_err:
            ax.text(
                0.02,
                0.02,
                f"Basemap no disponible: {basemap_err}",
                transform=ax.transAxes,
                fontsize=8,
                color="red",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.cm import ScalarMappable
        base_cmap = plt.get_cmap(cmap, len(unique_vals))
        discrete_cmap = ListedColormap(base_cmap(np.arange(len(unique_vals))))
        boundaries = np.arange(len(unique_vals) + 1) - 0.5
        norm = BoundaryNorm(boundaries, discrete_cmap.N)
        heat = ax.imshow(
            pred_int,
            extent=extent,
            origin="lower",
            cmap=discrete_cmap,
            norm=norm,
            alpha=alpha,
            zorder=3,
        )
        ax.set_axis_off()
        mappable = ScalarMappable(norm=norm, cmap=discrete_cmap)
        cbar = fig.colorbar(
            mappable,
            ax=ax,
            fraction=0.035,
            pad=0.02,
            ticks=np.arange(len(unique_vals)),
        )
        cbar.set_ticklabels([str(val) for val in unique_vals])
        cbar.set_label("Categoría")
        ax.set_title(title)
        fig.tight_layout()
        plt.show()
        return
    except Exception as err:
        ax.text(
            0.02,
            0.02,
            f"Heatmap categórico no disponible ({err}); usando dispersión.",
            transform=ax.transAxes,
            fontsize=8,
            color="red",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )
        print(f"Error en plot_spatial_categories: {err}")
    
    # Fallback: scatter plot
    sc = ax.scatter(
        coords_df["lon"],
        coords_df["lat"],
        c=labels_arr,
        cmap=plt.get_cmap(cmap, len(unique_vals)),
        s=s,
        alpha=0.9,
        edgecolor="k",
        linewidth=0.2,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Categoría")
    fig.tight_layout()
    plt.show()

# %%

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim=8, p_drop=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

@torch.no_grad()
def encode_ae(model, X):
    tensor = torch.tensor(X, dtype=torch.float32)
    return model.encoder(tensor).numpy()

@torch.no_grad()
def get_z(model, X):
    return encode_ae(model, X)

def train_ae(model, X_tr, X_val, epochs=400, lr=1e-3, batch_size=64,
             weight_decay=1e-4, noise_std=0.05, patience=30, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    best, best_val, best_epoch = None, float("inf"), -1
    wait = 0
    history = {"early_stop": False}
    train_curve, val_curve = [], []

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(Xtr.size(0))
        train_loss_sum = 0.0
        n_seen = 0
        for i in range(0, Xtr.size(0), batch_size):
            batch = Xtr[idx[i:i+batch_size]]
            noisy = batch + noise_std * torch.randn_like(batch)
            opt.zero_grad()
            x_hat, _ = model(noisy)
            loss = ((x_hat - batch)**2).mean()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * batch.size(0)
            n_seen += batch.size(0)

        avg_train = train_loss_sum / max(1, n_seen)
        train_curve.append(avg_train)

        model.eval()
        with torch.no_grad():
            xhat_val, _ = model(Xva)
            val_loss = ((xhat_val - Xva)**2).mean().item()
        val_curve.append(val_loss)

        if verbose and ep % 25 == 0:
            print(f"[AE] ep {ep:03d} | val_mse={val_loss:.6f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = ep
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                history["early_stop"] = True
                if verbose:
                    print(f"[AE] Early stop @ {ep} | best_val={best_val:.6f}")
                break

    if best is not None:
        model.load_state_dict(best)

    history.update({
        "val_mse": best_val,
        "epochs": ep + 1,
        "best_epoch": best_epoch,
        "train_curve": train_curve,
        "val_curve": val_curve,
    })
    return model, history

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=12, p_drop=0.05):  # latent_dim ↑
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.mu     = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, input_dim),
        )
    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)
    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar

def elbo_loss(x, x_hat, mu, logvar, beta=1.0):
    recon = F.mse_loss(x_hat, x, reduction="mean")
    # KL(N(mu, sigma)||N(0,I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta*kl, recon, kl

def beta_schedule(ep, warmup=60, max_beta=1.0):
    # lineal: 0.1 → 1.0
    if ep < 5: return 0.1
    if ep < warmup: return 0.1 + (max_beta-0.1)*(ep-5)/(warmup-5)
    return max_beta

def train_vae(model, X_tr, X_val, epochs=400, lr=1e-3, batch_size=64,
              beta=1.0, weight_decay=1e-4, noise_std=0.05, patience=30, verbose=True, warmup=80, cap_warmup=60):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    
    best, best_val, best_epoch = None, float("inf"), -1
    best_recon, best_kl = None, None
    wait = 0
    
    history = {"early_stop": False}
    train_curve, val_curve = [], []
    train_recon_curve, train_kl_curve = [], []
    val_recon_curve, val_kl_curve = [], []
    capacity_curve = []
    
    for ep in range(epochs):
        model.train()
        cur_beta = beta_schedule(ep, warmup=warmup, max_beta=beta)
        # Capacity schedule simple (opcional, puede dejarse en 0 si no se usa)
        cur_C = 0.0  # No usamos capacity en versión simple, pero mantenemos para compatibilidad
        
        idx = torch.randperm(Xtr.size(0))
        train_loss_sum = train_recon_sum = train_kl_sum = 0.0
        n_seen = 0
        
        for i in range(0, Xtr.size(0), batch_size):
            batch = Xtr[idx[i:i+batch_size]]
            noisy = batch + noise_std*torch.randn_like(batch)
            opt.zero_grad()
            x_hat, mu, logvar = model(noisy)
            loss, recon, kl = elbo_loss(batch, x_hat, mu, logvar, beta=cur_beta)
            loss.backward()
            opt.step()
            
            bs = batch.size(0)
            train_loss_sum += loss.item() * bs
            train_recon_sum += recon.item() * bs
            train_kl_sum += kl.item() * bs
            n_seen += bs
        
        # Promedios de entrenamiento
        train_curve.append(train_loss_sum / max(1, n_seen))
        train_recon_curve.append(train_recon_sum / max(1, n_seen))
        train_kl_curve.append(train_kl_sum / max(1, n_seen))
        
        # Validación
        model.eval()
        with torch.no_grad():
            xh, mu_v, lv_v = model(Xva)
            val_loss, val_recon, val_kl = elbo_loss(Xva, xh, mu_v, lv_v, beta=cur_beta)
            val_loss = val_loss.item()
            val_recon = val_recon.item()
            val_kl = float(val_kl)
        
        val_curve.append(val_loss)
        val_recon_curve.append(val_recon)
        val_kl_curve.append(val_kl)
        capacity_curve.append(cur_C)
        
        if verbose and ep % 25 == 0:
            print(f"[VAE] ep {ep:03d} | val={val_loss:.6f} (recon={val_recon:.6f}, kl={val_kl:.6f})")
        
        # Early stopping basado en val_loss total
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = ep
            best_recon = val_recon
            best_kl = val_kl
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                history["early_stop"] = True
                if verbose:
                    print(f"[VAE] Early stop @ {ep} | best={best_val:.6f}")
                break
    
    if best is not None:
        model.load_state_dict(best)
    
    history.update({
        "val_loss": best_val,
        "val_recon": best_recon,
        "val_kl": best_kl,
        "epochs": ep + 1,
        "best_epoch": best_epoch,
        "train_curve": train_curve,
        "val_curve": val_curve,
        "train_recon_curve": train_recon_curve,
        "train_kl_curve": train_kl_curve,
        "val_recon_curve": val_recon_curve,
        "val_kl_curve": val_kl_curve,
        "capacity_curve": capacity_curve,
    })
    return model, history

@torch.no_grad()
def encode_vae(model, X, return_logvar=False):
    tensor = torch.tensor(X, dtype=torch.float32)
    x_hat, mu, logvar = model(tensor)
    if return_logvar:
        return mu.numpy(), logvar.numpy()
    return mu.numpy()

@torch.no_grad()
def get_mu_logvar(model, X):
    X_t = torch.tensor(X, dtype=torch.float32)
    _, mu, logvar = model(X_t)
    return mu.numpy(), logvar.numpy()

@torch.no_grad()
def reconstruction_mse(model_key, model, X):
    tensor = torch.tensor(X, dtype=torch.float32)
    if model_key == "AE":
        x_hat, _ = model(tensor)
    elif model_key == "VAE":
        x_hat, _, _ = model(tensor)
    else:
        raise ValueError(f"Modelo no soportado: {model_key}")
    return ((x_hat - tensor)**2).mean().item()

def get_latent_vectors(model_key, model, X, return_logvar=False):
    if model_key == "AE":
        return encode_ae(model, X)
    if model_key == "VAE":
        if return_logvar:
            return encode_vae(model, X, return_logvar=True)
        return encode_vae(model, X)
    raise ValueError(f"Modelo no soportado: {model_key}")

print("Arquitecturas AE y VAE definidas (version 05_resilience_vae)")


# %% [markdown]
# ## Experimento 1 — Representación base (reconstrucción)

# %%
perm = np.random.permutation(X_BASE.shape[0])
n_val = int(0.2 * len(perm))
val_idx, tr_idx = perm[:n_val], perm[n_val:]
X_tr, X_val = X_BASE[tr_idx], X_BASE[val_idx]

LATENT_DIM_AE = 8
LATENT_DIM_VAE = 8
LR = 1e-3
N_PER_SCENARIO = B245.shape[0]

MODEL_ORDER = ["AE", "VAE"]
MODEL_CONFIG = {
    "AE": {
        "latent_dim": LATENT_DIM_AE,
        "build_fn": lambda input_dim: AE(input_dim, latent_dim=LATENT_DIM_AE, p_drop=0.1),
        "train_fn": train_ae,
        "train_kwargs": {
            "epochs": 400,
            "lr": LR,
            "batch_size": 64,
            "weight_decay": 1e-4,
            "noise_std": 0.05,
            "patience": 30,
            "verbose": True,
        },
    },
    "VAE": {
        "latent_dim": LATENT_DIM_VAE,
        "build_fn": lambda input_dim: VAE(input_dim, latent_dim=LATENT_DIM_VAE, p_drop=0.1),
        "train_fn": train_vae,
        "train_kwargs": {
            "epochs": 400,
            "lr": 1e-3,          # antes 5e-4
            "batch_size": 64,
            "weight_decay": 5e-5,
            "noise_std": 0.01,   # antes 0.02
            "patience": 90,      # antes 60
            "verbose": True,
            "beta": 1.0,         # antes 0.5
            "warmup": 60,        # antes 150
            "cap_warmup": 60, 
        },
    },
}

MODELS = {}
MODEL_TRAIN_LOGS = {}
LATENTS = {}
LATENT_LOGVARS = {}
RECON_ROWS = []

DATA_BLOCKS = {
    "base": X_BASE,
    "B245": B245,
    "B370": B370,
    "B585": B585,
    "T245": T245,
    "T370": T370,
    "T585": T585,
}

for model_key in MODEL_ORDER:
    cfg = MODEL_CONFIG[model_key]
    model = cfg["build_fn"](X_BASE.shape[1])
    model, history = cfg["train_fn"](model, X_tr, X_val, **cfg["train_kwargs"])
    model.eval()

    MODELS[model_key] = model
    MODEL_TRAIN_LOGS[model_key] = history

    with torch.no_grad():
        Xv_t = torch.tensor(X_val, dtype=torch.float32)
        if model_key == "AE":
            xhat_val, _ = model(Xv_t)
        else:
            xhat_val, _, _ = model(Xv_t)
        val_pred = xhat_val.detach().cpu().numpy()
    val_mse = reconstruction_mse(model_key, model, X_val)
    val_r2 = r2_score(X_val, val_pred, multioutput="variance_weighted")
    MODEL_TRAIN_LOGS[model_key]["val_r2"] = val_r2

    row = {
        "model": model_key,
        "latent_dim": cfg["latent_dim"],
        "val_mse": val_mse,
        "val_r2": val_r2,
        "epochs_trained": history.get("epochs"),
        "best_epoch": history.get("best_epoch"),
        "early_stop": history.get("early_stop"),
    }
    if model_key == "VAE":
        row.update({
            "val_elbo": history.get("val_loss"),
            "val_recon_component": history.get("val_recon"),
            "val_kl_component": history.get("val_kl"),
        })

    latents = {}
    logvars = {}
    for label, data in DATA_BLOCKS.items():
        if model_key == "VAE":
            mu, logvar = get_latent_vectors(model_key, model, data, return_logvar=True)
            latents[label] = mu
            logvars[label] = logvar
        else:
            latents[label] = get_latent_vectors(model_key, model, data)
    LATENTS[model_key] = latents
    if logvars:
        LATENT_LOGVARS[model_key] = logvars

    RECON_ROWS.append(row)

recon_df = pd.DataFrame(RECON_ROWS)
print("Reconstrucción (validación) por modelo:")
print(recon_df)

for model_key in MODEL_ORDER:
    base_lat_shape = LATENTS[model_key]["base"].shape
    print(f"  {model_key}: latent base shape={base_lat_shape}")


# %%
curve_tables = {}
for model_key in MODEL_ORDER:
    hist = MODEL_TRAIN_LOGS[model_key]
    train_curve = hist.get("train_curve", [])
    val_curve = hist.get("val_curve", [])
    epochs_hist = np.arange(1, len(train_curve) + 1)
    data = {
        "epoch": epochs_hist,
        "train_loss": train_curve,
        "val_loss": val_curve,
    }
    if model_key == "VAE":
        data.update({
            "train_recon": hist.get("train_recon_curve", []),
            "train_kl": hist.get("train_kl_curve", []),
            "val_recon": hist.get("val_recon_curve", []),
            "val_kl": hist.get("val_kl_curve", []),
        })
    df_curve = pd.DataFrame(data)
    curve_tables[model_key] = df_curve
    print(f"\nCurvas de entrenamiento — {model_key} (primeros 5 epochs):")
    print(df_curve.head())
    print(f"\nCurvas de entrenamiento — {model_key} (últimos 5 epochs):")
    print(df_curve.tail())

fig, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(7, 4 * len(MODEL_ORDER)), sharex=True)
if len(MODEL_ORDER) == 1:
    axes = [axes]
for ax, model_key in zip(axes, MODEL_ORDER):
    df_curve = curve_tables[model_key]
    ax.plot(df_curve["epoch"], df_curve["train_loss"], label="train_loss")
    ax.plot(df_curve["epoch"], df_curve["val_loss"], label="val_loss")
    if model_key == "VAE":
        ax.plot(df_curve["epoch"], df_curve.get("val_recon"), label="val_recon", linestyle="--")
        ax.plot(df_curve["epoch"], df_curve.get("val_kl"), label="val_kl", linestyle=":")
    ax.set_title(f"Curvas de entrenamiento — {model_key}")
    ax.set_ylabel("Loss")
    ax.legend()
axes[-1].set_xlabel("Epoch")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(recon_df["model"], recon_df["val_r2"], color=["tab:blue", "tab:orange"][:len(MODEL_ORDER)])
plt.ylabel("R² validación")
plt.title("Accuracy (R²) por modelo")
plt.ylim(0, 1)
plt.show()

scenario_labels = (
    ["B245"] * N_PER_SCENARIO
    + ["B370"] * N_PER_SCENARIO
    + ["B585"] * N_PER_SCENARIO
 )
color_map = {"B245": "tab:blue", "B370": "tab:orange", "B585": "tab:green"}
future_labels = ["T245", "T370", "T585"]

for model_key in MODEL_ORDER:
    z_base = LATENTS[model_key]["base"]
    pca = PCA(n_components=2, random_state=SEED)
    z_base_2d = pca.fit_transform(z_base)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
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
    for ax, future in zip(axes[1:], future_labels):
        z_future = LATENTS[model_key][future]
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
    plt.suptitle(f"Proyección PCA del espacio latente ({model_key})")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ### Interpretación de los gráficos PCA
# 
# **Panel 1 (Base)**: Todos los escenarios base (B245, B370, B585) del año 2020 mezclados en el espacio latente. La superposición indica que en el período histórico los tres escenarios SSP aún no han divergido significativamente (esperado, ya que comparten historia hasta ~2015).
# 
# **Paneles 2-4 (Futuros)**: Cada escenario futuro (T245, T370, T585) proyectado en el mismo espacio PCA entrenado con base. El desplazamiento respecto al cluster base indica:
# - **Dirección**: hacia dónde evoluciona el clima
# - **Magnitud**: qué tan distinto es el futuro del histórico (proxy de pérdida de resiliencia)
# - **Dispersión**: si el futuro se expande o se mantiene compacto
# 
# **Hipótesis clave**: Si los futuros se alejan mucho del base → pérdida de resiliencia climática territorial.

# %% [markdown]
# ## Análisis de Desplazamiento Latente (Base → Futuros)
# 
# Cuantificamos qué tan lejos se mueven los escenarios futuros respecto al período base (2020). Esta métrica es un **proxy de pérdida de resiliencia**: mayor desplazamiento = mayor cambio climático = menor resiliencia.

# %%
from sklearn.metrics import adjusted_rand_score

# Calcular centroide del BASE (histórico común)
displacement_metrics = {}

for model_key in MODEL_ORDER:
    print(f"\n{'='*60}")
    print(f"Modelo: {model_key}")
    print(f"{'='*60}")
    
    z_base = LATENTS[model_key]["base"]
    z_T245 = LATENTS[model_key]["T245"]
    z_T370 = LATENTS[model_key]["T370"]
    z_T585 = LATENTS[model_key]["T585"]
    
    # Centroide global del base (2020)
    centroid_base = z_base.mean(axis=0)
    
    # Distancia euclidiana promedio de cada futuro respecto al centroide base
    dist_T245 = np.linalg.norm(z_T245 - centroid_base, axis=1).mean()
    dist_T370 = np.linalg.norm(z_T370 - centroid_base, axis=1).mean()
    dist_T585 = np.linalg.norm(z_T585 - centroid_base, axis=1).mean()
    
    # Dispersión (std de las distancias)
    dispersion_T245 = np.linalg.norm(z_T245 - centroid_base, axis=1).std()
    dispersion_T370 = np.linalg.norm(z_T370 - centroid_base, axis=1).std()
    dispersion_T585 = np.linalg.norm(z_T585 - centroid_base, axis=1).std()
    
    displacement_metrics[model_key] = {
        "dist_T245": dist_T245,
        "dist_T370": dist_T370,
        "dist_T585": dist_T585,
        "dispersion_T245": dispersion_T245,
        "dispersion_T370": dispersion_T370,
        "dispersion_T585": dispersion_T585,
    }
    
    print(f"\nDesplazamiento promedio desde base (2020) [unidades latentes]:")
    print(f"  T245 (2040-2080): {dist_T245:.4f} ± {dispersion_T245:.4f}")
    print(f"  T370 (2040-2080): {dist_T370:.4f} ± {dispersion_T370:.4f}")
    print(f"  T585 (2040-2080): {dist_T585:.4f} ± {dispersion_T585:.4f}")
    
    print(f"\nIncremento relativo respecto a T245:")
    print(f"  T370 vs T245: {((dist_T370 / dist_T245 - 1) * 100):.1f}% más desplazado")
    print(f"  T585 vs T245: {((dist_T585 / dist_T245 - 1) * 100):.1f}% más desplazado")
    
    print(f"\nInterpretación:")
    if dist_T585 > dist_T370 > dist_T245:
        print("  ✓ Orden esperado: SSP585 > SSP370 > SSP245")
        print("    → Mayor forzamiento = mayor desplazamiento latente = menor resiliencia")
    else:
        print("  ⚠ Orden inesperado en desplazamientos")

# Comparación visual
df_displacement = pd.DataFrame([
    {"model": mk, "scenario": "T245", "distance": displacement_metrics[mk]["dist_T245"]}
    for mk in MODEL_ORDER
] + [
    {"model": mk, "scenario": "T370", "distance": displacement_metrics[mk]["dist_T370"]}
    for mk in MODEL_ORDER
] + [
    {"model": mk, "scenario": "T585", "distance": displacement_metrics[mk]["dist_T585"]}
    for mk in MODEL_ORDER
])

fig, ax = plt.subplots(figsize=(8, 5))
width = 0.25
x = np.arange(len(MODEL_ORDER))
scenarios = ["T245", "T370", "T585"]
colors = ["tab:blue", "tab:orange", "tab:red"]

for i, scenario in enumerate(scenarios):
    values = [displacement_metrics[mk][f"dist_{scenario}"] for mk in MODEL_ORDER]
    ax.bar(x + i * width, values, width, label=scenario, color=colors[i], alpha=0.8)

ax.set_xlabel("Modelo")
ax.set_ylabel("Desplazamiento latente promedio")
ax.set_title("Desplazamiento de futuros SSP respecto al base (2020)")
ax.set_xticks(x + width)
ax.set_xticklabels(MODEL_ORDER)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nTabla resumen:")
print(df_displacement.pivot(index="model", columns="scenario", values="distance"))

# %% [markdown]
# ## Clustering en BASE + Proyección a Futuros
# 
# **Estrategia para medir resiliencia por zonas:**
# 
# 1. **Clusterizar el espacio latente BASE (2020)** → identificar regiones con comportamiento climático similar
# 2. **Proyectar los futuros (T245, T370, T585)** a esos mismos clusters → ver si los píxeles se mantienen en su cluster o migran
# 3. **Métricas de resiliencia por cluster:**
#    - Compacidad: distancia intra-cluster (menor = más resiliente)
#    - Estabilidad: % de píxeles que permanecen en su cluster base
#    - Expansión: aumento de la dispersión del cluster en futuros

# %% [markdown]
# ### Método del Codo: Determinando el número óptimo de clusters
# 
# Antes de clusterizar, evaluamos K entre 4 y 10 usando:
# - **Inertia (WCSS)**: suma de distancias intra-cluster (menor = mejor compacidad)
# - **Silhouette Score**: coherencia de clustering (-1 a 1, mayor = mejor separación)
# - **Davies-Bouldin Index**: ratio de dispersión intra/inter-cluster (menor = mejor)

# %%
# Método del Codo + Silhouette para determinar K óptimo

K_RANGE = range(4, 20)  # Probar de 4 a 20 clusters

ELBOW_RESULTS = {}

for model_key in MODEL_ORDER:
    print(f"\n{'='*60}")
    print(f"Evaluando K óptimo — {model_key}")
    print(f"{'='*60}")
    
    z_base = LATENTS[model_key]["base"]
    
    inertias = []
    silhouettes = []
    davies_bouldins = []
    
    for k in K_RANGE:
        # KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=50, max_iter=300)
        labels = kmeans.fit_predict(z_base)
        
        # Métricas
        inertias.append(kmeans.inertia_)  # WCSS (Within-Cluster Sum of Squares)
        silhouettes.append(silhouette_score(z_base, labels))
        davies_bouldins.append(davies_bouldin_score(z_base, labels))
    
    ELBOW_RESULTS[model_key] = {
        "k_values": list(K_RANGE),
        "inertia": inertias,
        "silhouette": silhouettes,
        "davies_bouldin": davies_bouldins,
    }
    
    # Visualización
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Panel 1: Inertia (Elbow)
    ax = axes[0]
    ax.plot(K_RANGE, inertias, marker="o", linewidth=2, markersize=8, color="tab:blue")
    ax.set_xlabel("Número de clusters (K)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Método del Codo")
    ax.grid(alpha=0.3)
    ax.set_xticks(K_RANGE)
    
    # Panel 2: Silhouette Score (mayor = mejor)
    ax = axes[1]
    ax.plot(K_RANGE, silhouettes, marker="o", linewidth=2, markersize=8, color="tab:green")
    ax.set_xlabel("Número de clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Cohesión de Clusters")
    ax.grid(alpha=0.3)
    ax.set_xticks(K_RANGE)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    
    # Panel 3: Davies-Bouldin Index (menor = mejor)
    ax = axes[2]
    ax.plot(K_RANGE, davies_bouldins, marker="o", linewidth=2, markersize=8, color="tab:orange")
    ax.set_xlabel("Número de clusters (K)")
    ax.set_ylabel("Davies-Bouldin Index")
    ax.set_title("Separación de Clusters")
    ax.grid(alpha=0.3)
    ax.set_xticks(K_RANGE)
    
    plt.suptitle(f"Evaluación de K óptimo — {model_key}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    
    # Tabla resumen
    df_metrics = pd.DataFrame({
        "K": K_RANGE,
        "Inertia": inertias,
        "Silhouette": silhouettes,
        "Davies-Bouldin": davies_bouldins,
    })
    
    print(f"\nMétricas por K:")
    print(df_metrics.to_string(index=False))
    
    # Sugerencias
    best_silhouette_k = K_RANGE[np.argmax(silhouettes)]
    best_db_k = K_RANGE[np.argmin(davies_bouldins)]
    
    print(f"\nRecomendaciones basadas en métricas:")
    print(f"  Mejor Silhouette Score: K={best_silhouette_k} ({max(silhouettes):.3f})")
    print(f"  Mejor Davies-Bouldin: K={best_db_k} ({min(davies_bouldins):.3f})")
    
    # Detectar "codo" en inertia (cambio de pendiente)
    # Usamos la segunda derivada discreta
    if len(inertias) > 2:
        deltas = np.diff(inertias)
        second_deltas = np.diff(deltas)
        elbow_idx = np.argmax(second_deltas) + 2  # +2 porque perdemos 2 puntos con diff
        elbow_k = K_RANGE[min(elbow_idx, len(K_RANGE)-1)]
        print(f"  Codo detectado (inertia): K≈{elbow_k}")
    
    print(f"\n→ Interpretación:")
    print(f"  - Silhouette > 0.5: excelente separación")
    print(f"  - Silhouette 0.25-0.5: razonable")
    print(f"  - Silhouette < 0.25: clusters débiles")
    print(f"  - Davies-Bouldin < 1.0: buena separación")


# %%
# Experimento: Clustering en BASE + Proyección de Futuros
# Objetivo: Medir resiliencia territorial mediante estabilidad de clusters

K_CLUSTERS = 8

CLUSTERING_RESULTS = {}

for model_key in MODEL_ORDER:
    print(f"\n{'='*60}")
    print(f"Clustering en BASE — Modelo: {model_key}")
    print(f"{'='*60}")
    
    z_base = LATENTS[model_key]["base"]
    z_T245 = LATENTS[model_key]["T245"]
    z_T370 = LATENTS[model_key]["T370"]
    z_T585 = LATENTS[model_key]["T585"]
    
    # 1. Clustering en BASE (2020)
    kmeans_base = KMeans(n_clusters=K_CLUSTERS, random_state=SEED, n_init=50)
    labels_base = kmeans_base.fit_predict(z_base)
    centroids_base = kmeans_base.cluster_centers_
    
    print(f"\nDistribución de píxeles en clusters BASE:")
    for cluster_id in range(K_CLUSTERS):
        count = (labels_base == cluster_id).sum()
        pct = count / len(labels_base) * 100
        print(f"  Cluster {cluster_id}: {count} píxeles ({pct:.1f}%)")
    
    # 2. Proyectar futuros a los clusters BASE
    labels_T245 = kmeans_base.predict(z_T245)
    labels_T370 = kmeans_base.predict(z_T370)
    labels_T585 = kmeans_base.predict(z_T585)
    
    # 3. Métricas de resiliencia por cluster
    cluster_resilience = []
    
    # Recordar: z_base = stack de [B245, B370, B585], entonces labels_base también
    # Necesitamos separar las labels por escenario para obtener máscaras correctas
    N_PER_SCENARIO = len(z_T245)  # 661 píxeles por escenario
    
    # Dividir labels_base en 3 escenarios
    labels_B245 = labels_base[:N_PER_SCENARIO]
    labels_B370 = labels_base[N_PER_SCENARIO:2*N_PER_SCENARIO]
    labels_B585 = labels_base[2*N_PER_SCENARIO:]
    
    for cluster_id in range(K_CLUSTERS):
        # Máscara de píxeles en este cluster para cada escenario BASE
        mask_B245 = (labels_B245 == cluster_id)
        mask_B370 = (labels_B370 == cluster_id)
        mask_B585 = (labels_B585 == cluster_id)
        
        # Usar promedio de los 3 escenarios base para calcular n_pixels representativo
        n_pixels_245 = mask_B245.sum()
        n_pixels_370 = mask_B370.sum()
        n_pixels_585 = mask_B585.sum()
        n_pixels = int((n_pixels_245 + n_pixels_370 + n_pixels_585) / 3)
        
        if n_pixels == 0:
            continue
        
        # Centroide del cluster en BASE
        centroid = centroids_base[cluster_id]
        
        # Compacidad BASE (promedio de los 3 escenarios base)
        z_B245 = z_base[:N_PER_SCENARIO]
        z_B370 = z_base[N_PER_SCENARIO:2*N_PER_SCENARIO]
        z_B585 = z_base[2*N_PER_SCENARIO:]
        
        comp_base_245 = np.linalg.norm(z_B245[mask_B245] - centroid, axis=1).mean() if mask_B245.any() else 0
        comp_base_370 = np.linalg.norm(z_B370[mask_B370] - centroid, axis=1).mean() if mask_B370.any() else 0
        comp_base_585 = np.linalg.norm(z_B585[mask_B585] - centroid, axis=1).mean() if mask_B585.any() else 0
        compactness_base = (comp_base_245 + comp_base_370 + comp_base_585) / 3
        
        # Compacidad en FUTUROS
        # Usamos la máscara de cada escenario BASE correspondiente para trackear los mismos píxeles
        compactness_T245 = np.linalg.norm(z_T245[mask_B245] - centroid, axis=1).mean() if mask_B245.any() else 0
        compactness_T370 = np.linalg.norm(z_T370[mask_B370] - centroid, axis=1).mean() if mask_B370.any() else 0
        compactness_T585 = np.linalg.norm(z_T585[mask_B585] - centroid, axis=1).mean() if mask_B585.any() else 0
        
        # Expansión relativa (qué tanto crece la distancia al centroide)
        expansion_T245 = (compactness_T245 / compactness_base - 1) * 100 if compactness_base > 0 else 0
        expansion_T370 = (compactness_T370 / compactness_base - 1) * 100 if compactness_base > 0 else 0
        expansion_T585 = (compactness_T585 / compactness_base - 1) * 100 if compactness_base > 0 else 0
        
        # Estabilidad: % de píxeles que permanecen en el mismo cluster
        stability_T245 = (labels_T245[mask_B245] == cluster_id).sum() / n_pixels_245 * 100 if n_pixels_245 > 0 else 0
        stability_T370 = (labels_T370[mask_B370] == cluster_id).sum() / n_pixels_370 * 100 if n_pixels_370 > 0 else 0
        stability_T585 = (labels_T585[mask_B585] == cluster_id).sum() / n_pixels_585 * 100 if n_pixels_585 > 0 else 0
        
        cluster_resilience.append({
            "cluster_id": cluster_id,
            "n_pixels": n_pixels,
            "compactness_base": compactness_base,
            "compactness_T245": compactness_T245,
            "compactness_T370": compactness_T370,
            "compactness_T585": compactness_T585,
            "expansion_T245": expansion_T245,
            "expansion_T370": expansion_T370,
            "expansion_T585": expansion_T585,
            "stability_T245": stability_T245,
            "stability_T370": stability_T370,
            "stability_T585": stability_T585,
        })
    
    df_cluster_resilience = pd.DataFrame(cluster_resilience)
    
    CLUSTERING_RESULTS[model_key] = {
        "kmeans": kmeans_base,
        "labels_base": labels_base,
        "labels_T245": labels_T245,
        "labels_T370": labels_T370,
        "labels_T585": labels_T585,
        "resilience_df": df_cluster_resilience,
    }
    
    print(f"\nMétricas de resiliencia por cluster:")
    print(df_cluster_resilience[["cluster_id", "n_pixels", "expansion_T585", "stability_T585"]].to_string(index=False))
    
    print(f"\nInterpretación:")
    print(f"  Expansión T585: {df_cluster_resilience['expansion_T585'].mean():.1f}% promedio")
    print(f"  Estabilidad T585: {df_cluster_resilience['stability_T585'].mean():.1f}% promedio")
    print(f"  → Cluster más resiliente: {df_cluster_resilience.loc[df_cluster_resilience['expansion_T585'].idxmin(), 'cluster_id']}")
    print(f"  → Cluster menos resiliente: {df_cluster_resilience.loc[df_cluster_resilience['expansion_T585'].idxmax(), 'cluster_id']}")

# %% [markdown]
# ### Visualización de clusters en espacio latente (PCA)
# 
# Proyectamos el espacio latente 8D a 2D usando PCA para ver cómo KMeans separó los clusters.

# %%
# Visualización de clusters en espacio latente 2D (PCA)

for model_key in MODEL_ORDER:
    print(f"\n{'='*60}")
    print(f"Clusters en espacio latente — {model_key}")
    print(f"{'='*60}")
    
    results = CLUSTERING_RESULTS[model_key]
    z_base = LATENTS[model_key]["base"]
    z_T245 = LATENTS[model_key]["T245"]
    z_T370 = LATENTS[model_key]["T370"]
    z_T585 = LATENTS[model_key]["T585"]
    
    labels_base = results["labels_base"]
    labels_T245 = results["labels_T245"]
    labels_T370 = results["labels_T370"]
    labels_T585 = results["labels_T585"]
    
    centroids_base = results["kmeans"].cluster_centers_
    
    # PCA 2D para visualización
    pca = PCA(n_components=2, random_state=SEED)
    z_base_2d = pca.fit_transform(z_base)
    z_T245_2d = pca.transform(z_T245)
    z_T370_2d = pca.transform(z_T370)
    z_T585_2d = pca.transform(z_T585)
    centroids_2d = pca.transform(centroids_base)
    
    print(f"Varianza explicada por PC1+PC2: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    
    # 4 paneles: BASE + 3 futuros
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Colormap discreto para clusters
    cluster_colors = plt.get_cmap("tab10", K_CLUSTERS)
    
    # Panel 1: BASE con clusters coloreados
    ax = axes[0]
    for cluster_id in range(K_CLUSTERS):
        mask = labels_base == cluster_id
        ax.scatter(
            z_base_2d[mask, 0],
            z_base_2d[mask, 1],
            c=[cluster_colors(cluster_id)],
            label=f"Cluster {cluster_id}",
            s=15,
            alpha=0.6,
            edgecolor="k",
            linewidth=0.3
        )
    # Centroides
    ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c="black",
        marker="X",
        s=200,
        edgecolor="white",
        linewidth=2,
        label="Centroides",
        zorder=10
    )
    ax.set_title(f"{model_key}: Clusters BASE (2020)", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    # Panel 2: T245 proyectado
    ax = axes[1]
    for cluster_id in range(K_CLUSTERS):
        mask = labels_T245 == cluster_id
        ax.scatter(
            z_T245_2d[mask, 0],
            z_T245_2d[mask, 1],
            c=[cluster_colors(cluster_id)],
            label=f"Cluster {cluster_id}",
            s=15,
            alpha=0.6,
            edgecolor="k",
            linewidth=0.3
        )
    ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c="black",
        marker="X",
        s=200,
        edgecolor="white",
        linewidth=2,
        zorder=10
    )
    ax.set_title(f"{model_key}: T245 (2040-2080)", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    # Panel 3: T370 proyectado
    ax = axes[2]
    for cluster_id in range(K_CLUSTERS):
        mask = labels_T370 == cluster_id
        ax.scatter(
            z_T370_2d[mask, 0],
            z_T370_2d[mask, 1],
            c=[cluster_colors(cluster_id)],
            label=f"Cluster {cluster_id}",
            s=15,
            alpha=0.6,
            edgecolor="k",
            linewidth=0.3
        )
    ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c="black",
        marker="X",
        s=200,
        edgecolor="white",
        linewidth=2,
        zorder=10
    )
    ax.set_title(f"{model_key}: T370 (2040-2080)", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    # Panel 4: T585 proyectado
    ax = axes[3]
    for cluster_id in range(K_CLUSTERS):
        mask = labels_T585 == cluster_id
        ax.scatter(
            z_T585_2d[mask, 0],
            z_T585_2d[mask, 1],
            c=[cluster_colors(cluster_id)],
            label=f"Cluster {cluster_id}",
            s=15,
            alpha=0.6,
            edgecolor="k",
            linewidth=0.3
        )
    ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c="black",
        marker="X",
        s=200,
        edgecolor="white",
        linewidth=2,
        zorder=10
    )
    ax.set_title(f"{model_key}: T585 (2040-2080)", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.suptitle(f"Evolución de clusters en espacio latente (PCA) — {model_key}", 
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.show()
    
    # Estadísticas de migración entre clusters
    print(f"\nMatriz de transición BASE → T585:")
    print(f"(filas=cluster BASE, columnas=cluster T585)")
    
    # Usamos B585 (últimos N_PER_SCENARIO) para comparación justa
    labels_B585 = labels_base[2*N_PER_SCENARIO:]
    transition_matrix = np.zeros((K_CLUSTERS, K_CLUSTERS), dtype=int)
    for i in range(K_CLUSTERS):
        mask = labels_B585 == i
        for j in range(K_CLUSTERS):
            transition_matrix[i, j] = (labels_T585[mask] == j).sum()
    
    df_transition = pd.DataFrame(
        transition_matrix,
        index=[f"B585_C{i}" for i in range(K_CLUSTERS)],
        columns=[f"T585_C{j}" for j in range(K_CLUSTERS)]
    )
    print(df_transition)
    
    # Porcentajes de retención por cluster
    print(f"\nRetención de cluster (diagonal de la matriz):")
    for i in range(K_CLUSTERS):
        total = transition_matrix[i, :].sum()
        retained = transition_matrix[i, i]
        pct = retained / total * 100 if total > 0 else 0
        print(f"  Cluster {i}: {retained}/{total} píxeles ({pct:.1f}%)")


# %%
# Visualización espacial de los clusters BASE y futuros

for model_key in MODEL_ORDER:
    print(f"\n{'='*60}")
    print(f"Mapeando clusters espaciales — {model_key}")
    print(f"{'='*60}")
    
    results = CLUSTERING_RESULTS[model_key]
    labels_base = results["labels_base"]
    labels_T245 = results["labels_T245"]
    labels_T370 = results["labels_T370"]
    labels_T585 = results["labels_T585"]
    
    # Extraer los 3 escenarios base por separado
    labels_B245 = labels_base[:N_PER_SCENARIO]
    labels_B370 = labels_base[N_PER_SCENARIO:2*N_PER_SCENARIO]
    labels_B585 = labels_base[2*N_PER_SCENARIO:]
    
    # 1. Mapa de clusters BASE (usando B245 como representante visual del 2020)
    plot_spatial_categories(
        labels_B245,
        title=f"Clusters BASE (2020, B245) — {model_key}",
        cmap="tab10",
        alpha=0.75
    )
    
    # 2. Migraciones de clusters para cada SSP (Base → Futuro)
    print(f"\nMigración de clusters por escenario SSP:")
    
    # SSP245: B245 → T245
    cluster_change_245 = labels_T245 - labels_B245
    changed_245 = (cluster_change_245 != 0).sum()
    stability_245 = (cluster_change_245 == 0).sum() / len(cluster_change_245) * 100
    print(f"  SSP245 (B245 → T245): {changed_245}/{len(cluster_change_245)} píxeles cambiaron ({stability_245:.1f}% estables)")
    
    plot_spatial_scalar(
        cluster_change_245,
        title=f"Migración de clusters: SSP245 (B245→T245) — {model_key}",
        cmap="RdBu_r",
        colorbar_label="Cambio de cluster",
        alpha=0.75,
        vmin=-2,
        vmax=2
    )
    
    # SSP370: B370 → T370
    cluster_change_370 = labels_T370 - labels_B370
    changed_370 = (cluster_change_370 != 0).sum()
    stability_370 = (cluster_change_370 == 0).sum() / len(cluster_change_370) * 100
    print(f"  SSP370 (B370 → T370): {changed_370}/{len(cluster_change_370)} píxeles cambiaron ({stability_370:.1f}% estables)")
    
    plot_spatial_scalar(
        cluster_change_370,
        title=f"Migración de clusters: SSP370 (B370→T370) — {model_key}",
        cmap="RdBu_r",
        colorbar_label="Cambio de cluster",
        alpha=0.75,
        vmin=-2,
        vmax=2
    )
    
    # SSP585: B585 → T585
    cluster_change_585 = labels_T585 - labels_B585
    changed_585 = (cluster_change_585 != 0).sum()
    stability_585 = (cluster_change_585 == 0).sum() / len(cluster_change_585) * 100
    print(f"  SSP585 (B585 → T585): {changed_585}/{len(cluster_change_585)} píxeles cambiaron ({stability_585:.1f}% estables)")
    
    plot_spatial_scalar(
        cluster_change_585,
        title=f"Migración de clusters: SSP585 (B585→T585) — {model_key}",
        cmap="RdBu_r",
        colorbar_label="Cambio de cluster",
        alpha=0.75,
        vmin=-2,
        vmax=2
    )
    
    # Resumen comparativo
    print(f"\nResumen de estabilidad por SSP:")
    print(f"  SSP245: {stability_245:.1f}% estable")
    print(f"  SSP370: {stability_370:.1f}% estable")
    print(f"  SSP585: {stability_585:.1f}% estable")
    print(f"\n→ Interpretación: Menor estabilidad = mayor cambio climático territorial")

# %% [markdown]
# ### Verificación: ¿Son consistentes los 3 escenarios base (2020)?
# 
# Antes de analizar los futuros, verificamos que B245, B370 y B585 del año 2020 producen clusters similares.

# %%
# Comparación de los 3 escenarios base (2020): B245, B370, B585

for model_key in MODEL_ORDER:
    print(f"\n{'='*60}")
    print(f"Verificando consistencia de escenarios base — {model_key}")
    print(f"{'='*60}")
    
    results = CLUSTERING_RESULTS[model_key]
    labels_base = results["labels_base"]
    
    # Extraer labels de cada escenario base
    labels_B245 = labels_base[:N_PER_SCENARIO]
    labels_B370 = labels_base[N_PER_SCENARIO:2*N_PER_SCENARIO]
    labels_B585 = labels_base[2*N_PER_SCENARIO:]
    
    # Métricas de acuerdo entre escenarios base
    agree_245_370 = (labels_B245 == labels_B370).sum()
    agree_245_585 = (labels_B245 == labels_B585).sum()
    agree_370_585 = (labels_B370 == labels_B585).sum()
    
    print(f"\nAcuerdo entre escenarios base (2020):")
    print(f"  B245 vs B370: {agree_245_370}/{N_PER_SCENARIO} píxeles ({agree_245_370/N_PER_SCENARIO*100:.1f}%)")
    print(f"  B245 vs B585: {agree_245_585}/{N_PER_SCENARIO} píxeles ({agree_245_585/N_PER_SCENARIO*100:.1f}%)")
    print(f"  B370 vs B585: {agree_370_585}/{N_PER_SCENARIO} píxeles ({agree_370_585/N_PER_SCENARIO*100:.1f}%)")
    
    # Adjusted Rand Index (métrica más robusta)
    from sklearn.metrics import adjusted_rand_score
    ari_245_370 = adjusted_rand_score(labels_B245, labels_B370)
    ari_245_585 = adjusted_rand_score(labels_B245, labels_B585)
    ari_370_585 = adjusted_rand_score(labels_B370, labels_B585)
    
    print(f"\nAdjusted Rand Index (1.0 = idéntico, 0.0 = aleatorio):")
    print(f"  B245 vs B370: {ari_245_370:.3f}")
    print(f"  B245 vs B585: {ari_245_585:.3f}")
    print(f"  B370 vs B585: {ari_370_585:.3f}")
    
    if ari_245_370 > 0.7 and ari_245_585 > 0.7 and ari_370_585 > 0.7:
        print(f"\n✓ Los 3 escenarios base (2020) son consistentes → válido usar cualquiera como proxy")
    else:
        print(f"\n⚠ Los escenarios base (2020) difieren significativamente → revisar datos")
    
    # Visualización espacial de los 3 escenarios base
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, labels, title in zip(axes, 
                                  [labels_B245, labels_B370, labels_B585],
                                  ["B245 (2020)", "B370 (2020)", "B585 (2020)"]):
        
        # Intentar hacer heatmap con KNN
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            xs, ys = transformer.transform(coords_df["lon"].values, coords_df["lat"].values)
            
            grid_res = 100
            grid_x = np.linspace(xs.min(), xs.max(), grid_res)
            grid_y = np.linspace(ys.min(), ys.max(), grid_res)
            GX, GY = np.meshgrid(grid_x, grid_y)
            
            coords = np.column_stack([xs, ys])
            grid_points = np.column_stack([GX.ravel(), GY.ravel()])
            
            n_neighbors = max(1, int(np.sqrt(len(labels))))
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
            clf.fit(coords, labels)
            pred_labels = clf.predict(grid_points).reshape(GX.shape)
            
            extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            
            try:
                import contextily as ctx
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs="EPSG:3857", alpha=1.0, attribution_size=6)
            except:
                pass
            
            from matplotlib.colors import ListedColormap, BoundaryNorm
            cluster_cmap = ListedColormap(plt.get_cmap("tab10", K_CLUSTERS)(np.arange(K_CLUSTERS)))
            boundaries = np.arange(K_CLUSTERS + 1) - 0.5
            norm = BoundaryNorm(boundaries, cluster_cmap.N)
            
            heat = ax.imshow(pred_labels, extent=extent, origin="lower", 
                           cmap=cluster_cmap, norm=norm, alpha=0.75, zorder=3)
            ax.set_axis_off()
            ax.set_title(f"{title} — Clusters", fontsize=12, fontweight="bold")
            
        except Exception as e:
            # Fallback: scatter plot
            sc = ax.scatter(coords_df["lon"], coords_df["lat"], 
                          c=labels, cmap=plt.get_cmap("tab10", K_CLUSTERS),
                          s=20, alpha=0.8, edgecolor="k", linewidth=0.2)
            ax.set_title(f"{title} — Clusters", fontsize=12, fontweight="bold")
            ax.set_xlabel("Longitud")
            ax.set_ylabel("Latitud")
            ax.set_aspect("equal", adjustable="box")
    
    plt.suptitle(f"Comparación de clusters base (2020) — {model_key}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    
    print(f"\n→ Si los 3 mapas lucen similares, confirma que B245 es un buen proxy del 2020")


# %%
# Comparación de resiliencia entre clusters

for model_key in MODEL_ORDER:
    df_res = CLUSTERING_RESULTS[model_key]["resilience_df"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Expansión por cluster
    ax = axes[0]
    x = df_res["cluster_id"]
    width = 0.25
    ax.bar(x - width, df_res["expansion_T245"], width, label="T245", color="tab:blue", alpha=0.8)
    ax.bar(x, df_res["expansion_T370"], width, label="T370", color="tab:orange", alpha=0.8)
    ax.bar(x + width, df_res["expansion_T585"], width, label="T585", color="tab:red", alpha=0.8)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Expansión (%)")
    ax.set_title(f"Expansión de clusters por SSP — {model_key}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # Panel 2: Estabilidad por cluster
    ax = axes[1]
    ax.bar(x - width, df_res["stability_T245"], width, label="T245", color="tab:blue", alpha=0.8)
    ax.bar(x, df_res["stability_T370"], width, label="T370", color="tab:orange", alpha=0.8)
    ax.bar(x + width, df_res["stability_T585"], width, label="T585", color="tab:red", alpha=0.8)
    ax.axhline(100, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Estabilidad (% píxeles que permanecen)")
    ax.set_title(f"Estabilidad de clusters por SSP — {model_key}")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{model_key} — Ranking de resiliencia por cluster (menor expansión = más resiliente):")
    df_sorted = df_res.sort_values("expansion_T585")
    print(df_sorted[["cluster_id", "expansion_T585", "stability_T585"]].to_string(index=False))

# %% [markdown]
# ## Índice de Resiliencia Climática Territorial (IRCT)
# 
# Construimos un índice compuesto que integra múltiples dimensiones de resiliencia:
# 1. **Estabilidad de cluster**: ¿Se mantiene en su patrón climático?
# 2. **Magnitud de cambio latente**: ¿Qué tan lejos se desplaza en el espacio climático?
# 3. **Consistencia entre SSPs**: ¿El cambio es similar en diferentes escenarios?
# 
# **Escala del índice**: 0 (vulnerabilidad extrema) a 100 (resiliencia máxima)

# %%
# Construcción del Índice de Resiliencia Climática Territorial (IRCT)

RESILIENCE_INDICES = {}

for model_key in MODEL_ORDER:
    print(f"\n{'='*60}")
    print(f"Calculando IRCT — {model_key}")
    print(f"{'='*60}")
    
    results = CLUSTERING_RESULTS[model_key]
    z_base = LATENTS[model_key]["base"]
    z_T245 = LATENTS[model_key]["T245"]
    z_T370 = LATENTS[model_key]["T370"]
    z_T585 = LATENTS[model_key]["T585"]
    
    labels_base = results["labels_base"]
    labels_T245 = results["labels_T245"]
    labels_T370 = results["labels_T370"]
    labels_T585 = results["labels_T585"]
    
    centroids_base = results["kmeans"].cluster_centers_
    
    # Extraer escenarios base por separado
    labels_B245 = labels_base[:N_PER_SCENARIO]
    labels_B370 = labels_base[N_PER_SCENARIO:2*N_PER_SCENARIO]
    labels_B585 = labels_base[2*N_PER_SCENARIO:]
    
    z_B245 = z_base[:N_PER_SCENARIO]
    z_B370 = z_base[N_PER_SCENARIO:2*N_PER_SCENARIO]
    z_B585 = z_base[2*N_PER_SCENARIO:]
    
    # =====================================================
    # COMPONENTE 1: Estabilidad de Cluster (0-100)
    # =====================================================
    # Promedio de estabilidad entre los 3 SSPs
    stable_245 = (labels_T245 == labels_B245).astype(float)
    stable_370 = (labels_T370 == labels_B370).astype(float)
    stable_585 = (labels_T585 == labels_B585).astype(float)
    
    stability_score = (stable_245 + stable_370 + stable_585) / 3 * 100
    
    # =====================================================
    # COMPONENTE 2: Magnitud de Cambio Latente (0-100)
    # =====================================================
    # Distancia euclidiana en espacio latente (invertida y normalizada)
    dist_245 = np.linalg.norm(z_T245 - z_B245, axis=1)
    dist_370 = np.linalg.norm(z_T370 - z_B370, axis=1)
    dist_585 = np.linalg.norm(z_T585 - z_B585, axis=1)
    
    # Promedio de distancias
    avg_distance = (dist_245 + dist_370 + dist_585) / 3
    
    # Normalizar distancias: 0 = máxima distancia (vulnerable), 100 = sin movimiento (resiliente)
    max_dist = np.percentile(avg_distance, 95)  # Usa percentil 95 para evitar outliers
    distance_score = 100 * (1 - np.clip(avg_distance / max_dist, 0, 1))
    
    # =====================================================
    # COMPONENTE 3: Consistencia entre SSPs (0-100)
    # =====================================================
    # Versión mejorada: medimos la divergencia entre SSPs usando pares
    # Si 2 o más SSPs coinciden → más consistente
    # Si los 3 difieren → menos consistente
    
    # Contamos coincidencias de pares
    match_245_370 = (labels_T245 == labels_T370).astype(int)
    match_245_585 = (labels_T245 == labels_T585).astype(int)
    match_370_585 = (labels_T370 == labels_T585).astype(int)
    
    # Score basado en cuántos pares coinciden:
    # 3 coincidencias (todos iguales) = 100
    # 2 coincidencias (mayoría) = 66.7
    # 1 coincidencia (un par igual) = 33.3
    # 0 coincidencias (todos diferentes) = 0
    total_matches = match_245_370 + match_245_585 + match_370_585
    consistency_score = (total_matches / 3) * 100
    
    # Alternativa más suave: usar distancia entre clusters (si están "cerca" aunque no iguales)
    # Varianza de los 3 cluster IDs (0 = todos iguales, mayor = más dispersos)
    cluster_array = np.column_stack([labels_T245, labels_T370, labels_T585])
    cluster_variance = np.var(cluster_array, axis=1)
    max_variance = ((K_CLUSTERS - 1) ** 2) / 3  # Varianza máxima teórica
    
    # Score basado en baja varianza (invertido y normalizado)
    consistency_score_variance = 100 * (1 - np.clip(cluster_variance / max_variance, 0, 1))
    
    # Usamos el promedio de ambas métricas para balance
    consistency_score = (consistency_score + consistency_score_variance) / 2
    
    # =====================================================
    # COMPONENTE 4: Distancia al centroide futuro (0-100)
    # =====================================================
    # Qué tan compacto está el píxel en su cluster futuro (SSP585 como worst-case)
    assigned_centroids = centroids_base[labels_T585]
    dist_to_centroid = np.linalg.norm(z_T585 - assigned_centroids, axis=1)
    
    max_centroid_dist = np.percentile(dist_to_centroid, 95)
    compactness_score = 100 * (1 - np.clip(dist_to_centroid / max_centroid_dist, 0, 1))
    
    # =====================================================
    # ÍNDICE COMPUESTO (pesos ajustables)
    # =====================================================
    # Pesos por componente (deben sumar 1.0)
    w_stability = 0.40      # Más peso a estabilidad (principal indicador)
    w_distance = 0.30       # Magnitud de cambio
    w_consistency = 0.20    # Consistencia entre escenarios
    w_compactness = 0.10    # Compacidad en cluster futuro
    
    IRCT = (
        w_stability * stability_score +
        w_distance * distance_score +
        w_consistency * consistency_score +
        w_compactness * compactness_score
    )
    
    # Guardar resultados
    resilience_df = pd.DataFrame({
        "pixel_id": np.arange(N_PER_SCENARIO),
        "lat": coords_df["lat"].values,
        "lon": coords_df["lon"].values,
        "cluster_B245": labels_B245,
        "cluster_T245": labels_T245,
        "cluster_T370": labels_T370,
        "cluster_T585": labels_T585,
        "stability_score": stability_score,
        "distance_score": distance_score,
        "consistency_score": consistency_score,
        "compactness_score": compactness_score,
        "IRCT": IRCT,
    })
    
    RESILIENCE_INDICES[model_key] = resilience_df
    
    # Estadísticas descriptivas
    print(f"\nEstadísticas del IRCT (0-100):")
    print(f"  Media: {IRCT.mean():.2f}")
    print(f"  Mediana: {np.median(IRCT):.2f}")
    print(f"  Std: {IRCT.std():.2f}")
    print(f"  Min: {IRCT.min():.2f}")
    print(f"  Max: {IRCT.max():.2f}")
    
    # Categorización
    high_resilience = (IRCT >= 70).sum()
    medium_resilience = ((IRCT >= 40) & (IRCT < 70)).sum()
    low_resilience = (IRCT < 40).sum()
    
    print(f"\nCategorización por nivel de resiliencia:")
    print(f"  Alta (IRCT ≥ 70): {high_resilience} píxeles ({high_resilience/N_PER_SCENARIO*100:.1f}%)")
    print(f"  Media (40 ≤ IRCT < 70): {medium_resilience} píxeles ({medium_resilience/N_PER_SCENARIO*100:.1f}%)")
    print(f"  Baja (IRCT < 40): {low_resilience} píxeles ({low_resilience/N_PER_SCENARIO*100:.1f}%)")
    
    # Correlaciones entre componentes
    print(f"\nCorrelaciones entre componentes del índice:")
    corr_matrix = resilience_df[["stability_score", "distance_score", "consistency_score", "compactness_score"]].corr()
    print(corr_matrix.round(2))
    
    # Top 10 zonas más resilientes
    print(f"\nTop 10 zonas más resilientes:")
    top10 = resilience_df.nlargest(10, "IRCT")[["pixel_id", "lat", "lon", "IRCT", "cluster_B245", "cluster_T585"]]
    print(top10.to_string(index=False))
    
    # Top 10 zonas más vulnerables
    print(f"\nTop 10 zonas más vulnerables:")
    bottom10 = resilience_df.nsmallest(10, "IRCT")[["pixel_id", "lat", "lon", "IRCT", "cluster_B245", "cluster_T585"]]
    print(bottom10.to_string(index=False))

print(f"\n{'='*60}")
print(f"Índice IRCT calculado para {len(MODEL_ORDER)} modelos")
print(f"{'='*60}")


# %% [markdown]
# ### Visualización espacial del IRCT
# 
# Mapeamos el índice de resiliencia para identificar zonas prioritarias.

# %%
# Visualización espacial del Índice de Resiliencia Climática Territorial

for model_key in MODEL_ORDER:
    print(f"\n{'='*60}")
    print(f"Mapeando IRCT — {model_key}")
    print(f"{'='*60}")
    
    resilience_df = RESILIENCE_INDICES[model_key]
    
    # 1. Mapa del IRCT continuo (0-100)
    plot_spatial_scalar(
        resilience_df["IRCT"].values,
        title=f"Índice de Resiliencia Climática Territorial — {model_key}",
        cmap="RdYlGn",  # Rojo (vulnerable) → Amarillo → Verde (resiliente)
        colorbar_label="IRCT (0=vulnerable, 100=resiliente)",
        alpha=0.75,
        vmin=0,
        vmax=100
    )
    
    # 2. Mapa categórico (Alta/Media/Baja resiliencia)
    resilience_category = pd.cut(
        resilience_df["IRCT"],
        bins=[0, 40, 70, 100],
        labels=["Baja", "Media", "Alta"],
        include_lowest=True
    )
    
    plot_spatial_categories(
        resilience_category,
        title=f"Categorías de Resiliencia — {model_key}",
        cmap="RdYlGn",
        alpha=0.75
    )
    
    # 3. Mapas de componentes individuales
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    components = [
        ("stability_score", "Estabilidad de Cluster"),
        ("distance_score", "Baja Magnitud de Cambio"),
        ("consistency_score", "Consistencia entre SSPs"),
        ("compactness_score", "Compacidad en Cluster Futuro"),
    ]
    
    for ax, (col, label) in zip(axes, components):
        values = resilience_df[col].values
        
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            xs, ys = transformer.transform(resilience_df["lon"].values, resilience_df["lat"].values)
            
            grid_res = 100
            grid_x = np.linspace(xs.min(), xs.max(), grid_res)
            grid_y = np.linspace(ys.min(), ys.max(), grid_res)
            GX, GY = np.meshgrid(grid_x, grid_y)
            
            coords = np.column_stack([xs, ys])
            grid_points = np.column_stack([GX.ravel(), GY.ravel()])
            
            n_neighbors = max(3, int(np.sqrt(len(values)) * 1.5))
            reg = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
            reg.fit(coords, values)
            GZ = reg.predict(grid_points).reshape(GX.shape)
            
            extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            
            try:
                import contextily as ctx
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs="EPSG:3857", alpha=1.0, attribution_size=6)
            except:
                pass
            
            heat = ax.imshow(GZ, extent=extent, origin="lower", cmap="viridis", 
                           vmin=0, vmax=100, alpha=0.75, zorder=3)
            ax.set_axis_off()
            ax.set_title(label, fontsize=11, fontweight="bold")
            
            from matplotlib.colorbar import ColorbarBase
            from matplotlib.colors import Normalize
            cbar = plt.colorbar(heat, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label("Score (0-100)", fontsize=8)
            
        except Exception as e:
            # Fallback: scatter plot
            sc = ax.scatter(resilience_df["lon"], resilience_df["lat"], 
                          c=values, cmap="viridis", s=20, alpha=0.8, 
                          edgecolor="k", linewidth=0.2, vmin=0, vmax=100)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Longitud", fontsize=9)
            ax.set_ylabel("Latitud", fontsize=9)
            ax.set_aspect("equal", adjustable="box")
            plt.colorbar(sc, ax=ax, label="Score (0-100)")
    
    plt.suptitle(f"Componentes del IRCT — {model_key}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    
    # 4. Histograma del IRCT
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(resilience_df["IRCT"], bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(resilience_df["IRCT"].mean(), color="red", linestyle="--", linewidth=2, label=f"Media: {resilience_df['IRCT'].mean():.1f}")
    ax.axvline(resilience_df["IRCT"].median(), color="orange", linestyle="--", linewidth=2, label=f"Mediana: {resilience_df['IRCT'].median():.1f}")
    ax.axvspan(0, 40, alpha=0.2, color="red", label="Baja resiliencia")
    ax.axvspan(40, 70, alpha=0.2, color="yellow", label="Media resiliencia")
    ax.axvspan(70, 100, alpha=0.2, color="green", label="Alta resiliencia")
    ax.set_xlabel("IRCT", fontsize=12)
    ax.set_ylabel("Frecuencia (píxeles)", fontsize=12)
    ax.set_title(f"Distribución del Índice de Resiliencia — {model_key}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ### IRCT Normalizado (contraste mejorado)
# 
# Aplicamos normalización min-max para expandir el rango dinámico del índice y hacer más evidentes las diferencias territoriales.

# %%
# IRCT Normalizado para aumentar contraste visual

for model_key in MODEL_ORDER:
    print(f"\n{'='*60}")
    print(f"IRCT Normalizado — {model_key}")
    print(f"{'='*60}")
    
    resilience_df = RESILIENCE_INDICES[model_key].copy()
    
    # =====================================================
    # MÉTODO 1: Min-Max Normalization (0-100)
    # =====================================================
    # Expande el rango para que el mínimo sea 0 y el máximo 100
    irct_original = resilience_df["IRCT"].values
    irct_min = irct_original.min()
    irct_max = irct_original.max()
    
    irct_minmax = 100 * (irct_original - irct_min) / (irct_max - irct_min)
    
    print(f"\nMin-Max Normalization:")
    print(f"  Rango original: [{irct_min:.2f}, {irct_max:.2f}]")
    print(f"  Rango normalizado: [0.00, 100.00]")
    print(f"  Media normalizada: {irct_minmax.mean():.2f}")
    
    # =====================================================
    # MÉTODO 2: Percentile-based Normalization
    # =====================================================
    # Usa percentiles 5-95 para evitar outliers
    p5 = np.percentile(irct_original, 5)
    p95 = np.percentile(irct_original, 95)
    
    irct_percentile = 100 * np.clip((irct_original - p5) / (p95 - p5), 0, 1)
    
    print(f"\nPercentile Normalization (5-95):")
    print(f"  P5={p5:.2f}, P95={p95:.2f}")
    print(f"  Media normalizada: {irct_percentile.mean():.2f}")
    
    # =====================================================
    # MÉTODO 3: Z-score + sigmoid (distribución más uniforme)
    # =====================================================
    # Estandariza y luego aplica sigmoid para 0-1 suave
    mean_irct = irct_original.mean()
    std_irct = irct_original.std()
    
    z_scores = (irct_original - mean_irct) / std_irct
    irct_sigmoid = 100 / (1 + np.exp(-z_scores))  # Sigmoid escalado a 0-100
    
    print(f"\nZ-score + Sigmoid:")
    print(f"  Media normalizada: {irct_sigmoid.mean():.2f}")
    print(f"  Std normalizada: {irct_sigmoid.std():.2f}")
    
    # =====================================================
    # MÉTODO 4: Power Transform (gamma correction)
    # =====================================================
    # Gamma < 1: expande valores bajos, comprime altos (más contraste en zonas vulnerables)
    # Gamma > 1: expande valores altos, comprime bajos (más contraste en zonas resilientes)
    gamma = 0.7  # Ajusta entre 0.5 (muy agresivo) y 1.5 (suave)
    
    irct_norm_01 = (irct_original - irct_min) / (irct_max - irct_min)  # Normalizar a [0,1]
    irct_power = 100 * (irct_norm_01 ** gamma)  # Aplicar gamma
    
    print(f"\nPower Transform (gamma={gamma}):")
    print(f"  Media normalizada: {irct_power.mean():.2f}")
    
    # =====================================================
    # Comparación visual de métodos
    # =====================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    methods = [
        (irct_original, "Original", "viridis", irct_min, irct_max),
        (irct_minmax, "Min-Max [0-100]", "RdYlGn", 0, 100),
        (irct_percentile, "Percentile [5-95]", "RdYlGn", 0, 100),
        (irct_sigmoid, "Z-score + Sigmoid", "RdYlGn", 0, 100),
        (irct_power, f"Power (γ={gamma})", "RdYlGn", 0, 100),
    ]
    
    for idx, (values, title, cmap, vmin, vmax) in enumerate(methods):
        if idx >= 5:  # Solo 5 métodos
            break
        
        ax = axes.flatten()[idx]
        
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            xs, ys = transformer.transform(resilience_df["lon"].values, resilience_df["lat"].values)
            
            grid_res = 100
            grid_x = np.linspace(xs.min(), xs.max(), grid_res)
            grid_y = np.linspace(ys.min(), ys.max(), grid_res)
            GX, GY = np.meshgrid(grid_x, grid_y)
            
            coords = np.column_stack([xs, ys])
            grid_points = np.column_stack([GX.ravel(), GY.ravel()])
            
            n_neighbors = max(3, int(np.sqrt(len(values)) * 1.5))
            reg = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
            reg.fit(coords, values)
            GZ = reg.predict(grid_points).reshape(GX.shape)
            
            extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            
            try:
                import contextily as ctx
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs="EPSG:3857", alpha=1.0, attribution_size=6)
            except:
                pass
            
            heat = ax.imshow(GZ, extent=extent, origin="lower", cmap=cmap, 
                           vmin=vmin, vmax=vmax, alpha=0.75, zorder=3)
            ax.set_axis_off()
            ax.set_title(title, fontsize=11, fontweight="bold")
            
            cbar = plt.colorbar(heat, ax=ax, fraction=0.035, pad=0.02)
            cbar.ax.tick_params(labelsize=8)
            
        except Exception as e:
            # Fallback: scatter
            sc = ax.scatter(resilience_df["lon"], resilience_df["lat"], 
                          c=values, cmap=cmap, s=20, alpha=0.8, 
                          edgecolor="k", linewidth=0.2, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_aspect("equal", adjustable="box")
            plt.colorbar(sc, ax=ax)
    
    # Histogramas comparativos
    ax = axes.flatten()[5]
    ax.hist(irct_original, bins=30, alpha=0.5, label="Original", color="gray", density=True)
    ax.hist(irct_minmax, bins=30, alpha=0.5, label="Min-Max", color="blue", density=True)
    ax.hist(irct_power, bins=30, alpha=0.5, label=f"Power (γ={gamma})", color="red", density=True)
    ax.set_xlabel("IRCT")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribuciones comparadas", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle(f"Comparación de Normalizaciones del IRCT — {model_key}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    
    # Recomendación
    print(f"\n{'='*60}")
    print(f"RECOMENDACIONES:")
    print(f"{'='*60}")
    print(f"  • Min-Max: Máximo contraste, sensible a outliers")
    print(f"  • Percentile: Robusto a outliers, buen compromiso")
    print(f"  • Sigmoid: Distribución más uniforme, suaviza extremos")
    print(f"  • Power (γ<1): Enfatiza diferencias en zonas vulnerables")
    print(f"  • Power (γ>1): Enfatiza diferencias en zonas resilientes")
    print(f"\n→ Para tu tesis, recomiendo Percentile o Power (γ=0.7)")
    
    # Guardar versión normalizada preferida
    resilience_df["IRCT_normalized"] = irct_power  # Puedes cambiar a irct_percentile si prefieres
    RESILIENCE_INDICES[model_key] = resilience_df


# %% [markdown]
# ### Resumen: Estrategia de Clustering para Medir Resiliencia
# 
# **Enfoque implementado: Clustering en BASE + Proyección a Futuros**
# 
# **¿Por qué clusterizar el BASE?**
# - El período base (2020) representa el estado climático histórico/actual
# - Clustering en BASE identifica **zonas con comportamiento climático similar** antes del cambio climático proyectado
# - Es la referencia contra la cual medimos el cambio
# 
# **Métricas de resiliencia por cluster:**
# 
# 1. **Expansión** (`expansion_T585`): 
#    - Cuánto aumenta la distancia promedio al centroide base en el futuro
#    - **Interpretación**: Mayor expansión = mayor dispersión climática = menor resiliencia
#    - Valores bajos (0-20%) → resiliente
#    - Valores altos (>50%) → vulnerable
# 
# 2. **Estabilidad** (`stability_T585`):
#    - % de píxeles que permanecen en su cluster original
#    - **Interpretación**: Alta estabilidad = comportamiento climático consistente = mayor resiliencia
#    - Valores altos (>80%) → resiliente
#    - Valores bajos (<50%) → transición climática significativa
# 
# **Para tu tesis:**
# - Clusters con **baja expansión + alta estabilidad** → zonas resilientes (candidatas para producción H₂)
# - Clusters con **alta expansión + baja estabilidad** → zonas vulnerables (requieren medidas de adaptación)
# - El análisis espacial muestra **dónde** están estas zonas en el Valle de Aconcagua
# 
# **Hipótesis validable:**
# > "El espacio latente del autoencoder permite identificar agrupamientos espaciales que reflejan distintos niveles de resiliencia climática bajo escenarios SSP, medida mediante la estabilidad de clusters y su expansión en el tiempo."

# %% [markdown]
# ## Caracterización de Clusters: ¿Qué define cada patrón climático?
# 
# Perfilamos cada cluster analizando las variables climáticas y energéticas originales (antes del autoencoder) para entender qué características distinguen cada grupo territorial.

# %%
# Caracterización de Clusters: Perfiles climáticos y energéticos

CLUSTER_PROFILES = {}

for model_key in MODEL_ORDER:
    print(f"\n{'='*80}")
    print(f"CARACTERIZACIÓN DE CLUSTERS — {model_key}")
    print(f"{'='*80}")
    
    results = CLUSTERING_RESULTS[model_key]
    labels_base = results["labels_base"]
    
    # Usar solo B245 para caracterización (661 píxeles)
    labels_B245 = labels_base[:N_PER_SCENARIO]
    
    # Datos originales (sin normalizar) para interpretación directa
    X_B245_orig = B245  # Ya construido con build_base_augmented (incluye mean + std_proxy + non-decadal)
    
    # Identificar nombres de features correspondientes a X_B245_orig
    # Recordar: X_B245_orig = [mean_base, std_proxy_base, non_features]
    n_climate_vars = len(base_mean_idx)  # Número de variables climáticas con estadísticas decadales
    n_non = len(non_idx)
    
    # Construir nombres de features para la base augmentada
    feature_names_base = []
    
    # 1. Medias decadales (2020)
    for idx in base_mean_idx:
        feature_names_base.append(feature_names[idx])
    
    # 2. Std proxy (max-min)/2 para 2020
    for idx in base_mean_idx:
        original_name = feature_names[idx]
        std_name = original_name.replace("_mean_", "_stdproxy_")
        feature_names_base.append(std_name)
    
    # 3. Features no decadales
    for idx in non_idx:
        feature_names_base.append(feature_names[idx])
    
    # Crear DataFrame con datos + labels
    df_profile = pd.DataFrame(X_B245_orig, columns=feature_names_base)
    df_profile["cluster"] = labels_B245
    df_profile["lat"] = coords_df["lat"].values
    df_profile["lon"] = coords_df["lon"].values
    
    # Calcular estadísticas por cluster
    cluster_profiles = []
    
    for cluster_id in range(K_CLUSTERS):
        mask = df_profile["cluster"] == cluster_id
        n_pixels = mask.sum()
        
        if n_pixels == 0:
            continue
        
        cluster_data = df_profile[mask]
        
        # Resumen geográfico
        lat_center = cluster_data["lat"].mean()
        lon_center = cluster_data["lon"].mean()
        lat_std = cluster_data["lat"].std()
        lon_std = cluster_data["lon"].std()
        
        profile = {
            "cluster_id": cluster_id,
            "n_pixels": n_pixels,
            "pct_territory": n_pixels / len(labels_B245) * 100,
            "lat_center": lat_center,
            "lon_center": lon_center,
            "lat_std": lat_std,
            "lon_std": lon_std,
        }
        
        # Calcular medias de todas las variables por cluster
        for col in feature_names_base:
            if col not in ["lat", "lon"]:
                profile[f"{col}_mean"] = cluster_data[col].mean()
                profile[f"{col}_std"] = cluster_data[col].std()
        
        cluster_profiles.append(profile)
    
    df_cluster_profiles = pd.DataFrame(cluster_profiles)
    CLUSTER_PROFILES[model_key] = df_cluster_profiles
    
    # =====================================================
    # ANÁLISIS 1: Top variables discriminantes
    # =====================================================
    print(f"\nVariables más discriminantes entre clusters:")
    print(f"(Coeficiente de variación inter-cluster)")
    
    # Calcular coef. de variación entre clusters para cada variable
    discrimination_scores = {}
    
    for col in feature_names_base:
        if col not in ["lat", "lon"]:
            means_col = df_cluster_profiles[[c for c in df_cluster_profiles.columns if c.startswith(col) and c.endswith("_mean")]]
            if not means_col.empty:
                cluster_means = means_col.iloc[:, 0].values  # Tomar la columna de medias
                cv = cluster_means.std() / (abs(cluster_means.mean()) + 1e-6)  # Coef. variación
                discrimination_scores[col] = cv
    
    # Top 15 variables que más diferencian clusters
    top_discriminators = sorted(discrimination_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    
    print(f"\nTop 15 variables discriminantes:")
    for var, score in top_discriminators:
        print(f"  {var}: {score:.4f}")
    
    # =====================================================
    # ANÁLISIS 2: Caracterización verbal de cada cluster
    # =====================================================
    print(f"\n{'='*80}")
    print(f"PERFILES DE CLUSTERS (interpretación)")
    print(f"{'='*80}")
    
    for _, row in df_cluster_profiles.iterrows():
        cluster_id = int(row["cluster_id"])
        n_pixels = int(row["n_pixels"])
        pct = row["pct_territory"]
        
        print(f"\nCLUSTER {cluster_id} ({n_pixels} píxeles, {pct:.1f}% del territorio)")
        print(f"  Ubicación: lat={row['lat_center']:.3f}°, lon={row['lon_center']:.3f}°")
        print(f"  Dispersión espacial: ±{row['lat_std']:.3f}° lat, ±{row['lon_std']:.3f}° lon")
        
        # Identificar características distintivas vs. promedio global
        print(f"\n  Características distintivas:")
        
        # Variables clave para caracterizar (ajusta según tus features)
        key_vars = [
            ("tasmax", "Temp. máxima"),
            ("tasmin", "Temp. mínima"),
            ("pr", "Precipitación"),
            ("wind", "Viento"),
            ("rsds", "Radiación solar"),
            ("h2_prod", "Producción H₂"),
        ]
        
        for var_pattern, var_label in key_vars:
            matching_cols = [c for c in df_cluster_profiles.columns if var_pattern in c.lower() and c.endswith("_mean")]
            if matching_cols:
                cluster_val = row[matching_cols[0]]
                global_mean = df_profile[[c for c in df_profile.columns if var_pattern in c.lower()][0]].mean()
                diff_pct = ((cluster_val - global_mean) / (abs(global_mean) + 1e-6)) * 100
                
                if abs(diff_pct) > 10:  # Solo mostrar diferencias significativas (>10%)
                    direction = "↑" if diff_pct > 0 else "↓"
                    print(f"    • {var_label}: {cluster_val:.2f} ({direction} {abs(diff_pct):.1f}% vs. promedio)")
    
    # =====================================================
    # ANÁLISIS 3: Heatmap de características por cluster
    # =====================================================
    # Seleccionar las top 10 variables discriminantes
    top_10_vars = [var for var, _ in top_discriminators[:10]]
    
    # Construir matriz de medias normalizadas (z-score por variable)
    heatmap_data = []
    heatmap_labels = []
    
    for var in top_10_vars:
        col_name = f"{var}_mean"
        if col_name in df_cluster_profiles.columns:
            values = df_cluster_profiles[col_name].values
            # Normalizar (z-score)
            values_norm = (values - values.mean()) / (values.std() + 1e-6)
            heatmap_data.append(values_norm)
            heatmap_labels.append(var.replace("_decadal_mean_2020", "").replace("_", " "))
    
    heatmap_matrix = np.array(heatmap_data)
    
    # Usar número real de clusters (algunos pueden estar vacíos)
    n_clusters_real = len(df_cluster_profiles)
    cluster_ids_real = df_cluster_profiles["cluster_id"].values.astype(int)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heatmap_matrix, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    
    ax.set_xticks(np.arange(n_clusters_real))
    ax.set_xticklabels([f"C{cid}" for cid in cluster_ids_real])
    ax.set_yticks(np.arange(len(heatmap_labels)))
    ax.set_yticklabels(heatmap_labels, fontsize=9)
    
    ax.set_xlabel("Cluster ID", fontsize=12)
    ax.set_ylabel("Variable", fontsize=12)
    ax.set_title(f"Perfil de Características por Cluster (z-scores) — {model_key}", fontsize=14, fontweight="bold")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Z-score (desviación del promedio)", fontsize=10)
    
    # Anotaciones en celdas
    for i in range(len(heatmap_labels)):
        for j in range(n_clusters_real):
            text = ax.text(j, i, f"{heatmap_matrix[i, j]:.1f}",
                         ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*80}")
    print(f"Interpretación del heatmap:")
    print(f"  • Rojo: Valor alto respecto al promedio territorial")
    print(f"  • Azul: Valor bajo respecto al promedio territorial")
    print(f"  • Blanco: Cercano al promedio")
    print(f"{'='*80}")
    
    # Guardar DataFrame completo y discriminadores para mapas (al final, después de calcularlos)
    CLUSTER_PROFILES[model_key + "_df_profile"] = df_profile
    CLUSTER_PROFILES[model_key + "_discriminators"] = top_discriminators



# %% [markdown]
# ### Mapas de Características Distintivas por Cluster
# 
# Visualizamos espacialmente las variables que más diferencian los clusters para entender la distribución territorial de cada característica.

# %%
# Comparación lado a lado: AE vs VAE (clusters base sin interpolación)

fig, axes = plt.subplots(1, 2, figsize=(24, 10))

for idx, model_key in enumerate(["AE", "VAE"]):
    ax = axes[idx]
    
    if model_key + "_df_profile" in CLUSTER_PROFILES:
        df_profile = CLUSTER_PROFILES[model_key + "_df_profile"]
        
        # Obtener labels del clustering base
        results = CLUSTERING_RESULTS[model_key]
        labels_base = results["labels_base"]
        labels_B245 = labels_base[:N_PER_SCENARIO]
        
        n_clusters_unique = len(np.unique(labels_B245))
        
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            xs, ys = transformer.transform(df_profile["lon"].values, df_profile["lat"].values)
            
            extent = (xs.min(), xs.max(), ys.min(), ys.max())
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            
            try:
                import contextily as ctx
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs="EPSG:3857", alpha=1.0, attribution_size=6)
            except:
                pass
            
            # Scatter directo
            from matplotlib.colors import ListedColormap
            cluster_cmap = ListedColormap(plt.get_cmap("tab10", n_clusters_unique)(np.arange(n_clusters_unique)))
            
            scatter = ax.scatter(xs, ys, c=labels_B245, cmap=cluster_cmap, 
                               s=50, alpha=0.8, edgecolor='black', linewidth=0.3, zorder=5)
            
            ax.set_axis_off()
            ax.set_title(f"{model_key} — {n_clusters_unique} clusters únicos", fontsize=14, fontweight="bold")
            
            # Leyenda
            from matplotlib.patches import Patch
            unique_labels = sorted(np.unique(labels_B245))
            legend_elements = [Patch(facecolor=cluster_cmap(i), label=f"Cluster {lbl}") 
                             for i, lbl in enumerate(unique_labels)]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, f"No hay datos para {model_key}", ha="center", va="center", transform=ax.transAxes)

plt.suptitle("Comparación de Clusters BASE (2020) — AE vs VAE", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout()
plt.show()

# Estadísticas comparativas
print("\nEstadísticas de clustering:")
print(f"{'='*60}")
for model_key in ["AE", "VAE"]:
    if model_key in CLUSTERING_RESULTS:
        results = CLUSTERING_RESULTS[model_key]
        labels_base = results["labels_base"]
        labels_B245 = labels_base[:N_PER_SCENARIO]
        unique_clusters = sorted(np.unique(labels_B245))
        counts = [np.sum(labels_B245 == c) for c in unique_clusters]
        
        print(f"\n{model_key}:")
        print(f"  Clusters activos: {unique_clusters}")
        print(f"  Distribución de píxeles:")
        for c, count in zip(unique_clusters, counts):
            pct = count / len(labels_B245) * 100
            print(f"    Cluster {c}: {count} píxeles ({pct:.1f}%)")


