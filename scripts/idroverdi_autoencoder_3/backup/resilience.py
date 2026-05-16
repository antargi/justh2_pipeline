# %% [markdown]
# # Índice de Resiliencia Climático-Territorial (IRCT)
# 
# Pipeline completo para calcular el **IRCT** por píxel y agregado por clúster siguiendo las definiciones matemáticas formales.
# 
# **Componentes del IRCT:**
# 1. Anomalía de reconstrucción $A_i^{(s)}$
# 2. Desplazamiento latente $S_{D,i}^{(s)}$
# 3. Estabilidad de pertenencia (softmax) $S_{C,i}^{(s)}$
# 4. Expansión del clúster $S_{E,i}^{(s)}$
# 5. Estabilidad energética H₂ $S_{H2,i}^{(s)}$ (opcional)
# 
# **Índice final:**
# $$
# \text{IRCT}_i^{(s)} = \left(A_i^{(s)}\right)^{w_a} \left(S_{D,i}^{(s)}\right)^{w_d} \left(S_{C,i}^{(s)}\right)^{w_c} \left(S_{E,i}^{(s)}\right)^{w_e} \left(S_{H2,i}^{(s)}\right)^{w_h}
# $$
# 
# $w_d=0.30, w_c=0.25, w_e=0.20, w_a=0.15, w_h=0.10$

# %% [markdown]
# ## 1. Imports y configuración

# %%
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import percentileofscore
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/home/aninotna/magister/tesis/justh2_pipeline"
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots", "resilience_analysis")
os.makedirs(PLOTS_DIR, exist_ok=True)
MODEL_ORDER = ["AE", "VAE"]

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Imports completados")
print(f"Directorio de plots: {PLOTS_DIR}")

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


# %% [markdown]
# ## 2. Definición de arquitecturas AE y VAE

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


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=12, p_drop=0.05):
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
        self.mu = nn.Linear(64, latent_dim)
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar

print("Arquitecturas AE y VAE definidas")

# %% [markdown]
# ## 3. Carga de datos del Experimento 1

# %%
print("CARGANDO DATOS DESDE EXPERIMENTO 1")
print()

# Buscar el archivo más reciente automáticamente
import glob
trained_dir = os.path.join(DATA_DIR, "autoencoder_trained_v2")
pattern = os.path.join(trained_dir, "experiment1_clustering_*.pkl")
available_files = glob.glob(pattern)

if not available_files:
    raise FileNotFoundError(f"No se encontraron archivos en: {pattern}")

# Ordenar por fecha de modificación (más reciente primero)
export_path = max(available_files, key=os.path.getmtime)

print(f"Archivo más reciente encontrado: {os.path.basename(export_path)}")
print(f"Ruta completa: {export_path}")
print()

if os.path.exists(export_path):
    print(f"Cargando datos...")
    print()
    
    with open(export_path, "rb") as f:
        exp1_data = pickle.load(f)
    
    # 1. Cargar modelos PyTorch
    print("1. Cargando modelos PyTorch...")
    models_path = exp1_data["models_path"]
    model_dims = exp1_data["model_dims"]
    
    MODELS = {}
    for model_key, model_file in models_path.items():
        dims = model_dims[model_key]
        
        if "AE" in model_key and "VAE" not in model_key:
            model = AE(
                input_dim=dims["input_dim"],
                latent_dim=dims["latent_dim"]
            )
        else:
            model = VAE(
                input_dim=dims["input_dim"],
                latent_dim=dims["latent_dim"]
            )
        
        model.load_state_dict(torch.load(model_file))
        model.eval()
        MODELS[model_key] = model
        print(f"  ✓ {model_key} (input={dims['input_dim']}, latent={dims['latent_dim']})")
    print()
    
    # 2. Extraer objetos del pickle
    print("2. Cargando datos adicionales...")
    LATENTS = exp1_data["LATENTS"]
    LATENT_LOGVARS = exp1_data.get("LATENT_LOGVARS", None)
    
    MODEL_ORDER = exp1_data["MODEL_ORDER"]
    LATENT_DIM_AE = exp1_data["LATENT_DIM_AE"]
    LATENT_DIM_VAE = exp1_data["LATENT_DIM_VAE"]
    N_PER_SCENARIO = exp1_data["N_PER_SCENARIO"]
    
    X_BASE = exp1_data["X_BASE"]
    X245_orig = exp1_data["X245_orig"]
    X370_orig = exp1_data["X370_orig"]
    X585_orig = exp1_data["X585_orig"]
    
    X245_norm = exp1_data.get("X245_norm", None)
    X370_norm = exp1_data.get("X370_norm", None)
    X585_norm = exp1_data.get("X585_norm", None)
    
    feature_names = exp1_data["feature_names"]
    coords_df = exp1_data["coords_df"]
    
    file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
    
    print(f"  ✓ Datos cargados ({file_size_mb:.2f} MB)")
    print()
    
    print("Objetos cargados:")
    print(f"  • MODELS: {len(MODELS)} modelos ({', '.join(MODELS.keys())})")
    print(f"  • LATENTS: {len(LATENTS)} conjuntos de embeddings")
    print(f"  • N_PER_SCENARIO: {N_PER_SCENARIO} puntos espaciales")
    print(f"  • feature_names: {len(feature_names)} variables")
    print(f"  • coords_df: {coords_df.shape[0]} píxeles")
    print()
    
else:
    raise FileNotFoundError(f"No se encontró {export_path}")

# %%
from sklearn.decomposition import PCA

print("VISUALIZACIÓN DEL ESPACIO LATENTE ACTUAL")
print("="*80)
print()

# Verificar qué modelo fue el último procesado
print(f"Último modelo procesado: {model_key}")
print()

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
# ## 4. KMeans con K=9
# 
# Aplicamos clustering con K=9 clusters siguiendo la misma metodología del notebook 08

# %%
def compute_inv_covs_per_cluster(Z_base_scaled, labels_base, n_clusters, eps=1e-6):
    """
    Calcula matrices de covarianza inversa (Mahalanobis) por clúster.
    Retorna un dict: cluster_id → inv_cov_matrix
    """
    Z = np.asarray(Z_base_scaled)
    inv_covs = {}

    for k in range(n_clusters):
        # Muestras del clúster k
        Zk = Z[labels_base == k]

        # Si el clúster es pequeño, usar identidad
        if len(Zk) < Z.shape[1] + 2:
            inv_covs[k] = np.eye(Z.shape[1])
            continue

        # Covarianza regularizada
        cov = np.cov(Zk.T)

        # Regularización para evitar matrices singulares
        cov = cov + eps * np.eye(cov.shape[0])

        # Inversa (Mahalanobis)
        inv_covs[k] = np.linalg.inv(cov)

    return inv_covs

# %%
# Experimento: Clustering en BASE + Proyección de Futuros
# Objetivo: Medir resiliencia territorial mediante estabilidad de clusters

from sklearn.preprocessing import StandardScaler

K_CLUSTERS = 10

CLUSTERING_RESULTS = {}

for model_key in MODEL_ORDER:
    print(f"\n{'='*60}")
    print(f"Clustering en BASE — Modelo: {model_key}")
    print(f"{'='*60}")
    
    z_base = LATENTS[model_key]["base"]
    z_T245 = LATENTS[model_key]["T245"]
    z_T370 = LATENTS[model_key]["T370"]
    z_T585 = LATENTS[model_key]["T585"]
    

    scaler = StandardScaler()
    scaler.fit(z_base)
    print(f"\nCreando nuevo scaler para estandarización")
    
    # Estandarizar todos los espacios latentes con los mismos parámetros
    z_base_scaled = scaler.transform(z_base)
    z_T245_scaled = scaler.transform(z_T245)
    z_T370_scaled = scaler.transform(z_T370)
    z_T585_scaled = scaler.transform(z_T585)
    
    print(f"\nVerificación de estandarización:")
    print(f"  z_base_scaled — Media: {z_base_scaled.mean(axis=0).mean():.4f}, Std: {z_base_scaled.std(axis=0).mean():.4f}")
    print(f"  z_T245_scaled — Media: {z_T245_scaled.mean(axis=0).mean():.4f}, Std: {z_T245_scaled.std(axis=0).mean():.4f}")
    print(f"  ✓ Todas las dimensiones contribuyen equitativamente al cálculo de distancias")
    
    # 1. Clustering en BASE estandarizado (2020)
    kmeans_base = KMeans(n_clusters=K_CLUSTERS, random_state=SEED, n_init=50)
    labels_base = kmeans_base.fit_predict(z_base_scaled)
    centroids_base = kmeans_base.cluster_centers_  # Centroides en espacio escalado

    inv_covs = compute_inv_covs_per_cluster(z_base_scaled, labels_base, K_CLUSTERS)
    
    print(f"\nDistribución de píxeles en clusters BASE:")
    for cluster_id in range(K_CLUSTERS):
        count = (labels_base == cluster_id).sum()
        pct = count / len(labels_base) * 100
        print(f"  Cluster {cluster_id}: {count} píxeles ({pct:.1f}%)")
    
    # 2. Proyectar futuros estandarizados a los clusters BASE
    labels_T245 = kmeans_base.predict(z_T245_scaled)
    labels_T370 = kmeans_base.predict(z_T370_scaled)
    labels_T585 = kmeans_base.predict(z_T585_scaled)
    
    # 3. Métricas de resiliencia por cluster
    cluster_resilience = []
    
    # Recordar: z_base = stack de [B245, B370, B585], entonces labels_base también
    # Necesitamos separar las labels por escenario para obtener máscaras correctas
    N_PER_SCENARIO = len(z_T245)  # 661 píxeles por escenario
    
    # Dividir labels_base en 3 escenarios
    labels_B245 = labels_base[:N_PER_SCENARIO]
    labels_B370 = labels_base[N_PER_SCENARIO:2*N_PER_SCENARIO]
    labels_B585 = labels_base[2*N_PER_SCENARIO:]
    
    # Dividir espacios latentes escalados
    z_B245_scaled = z_base_scaled[:N_PER_SCENARIO]
    z_B370_scaled = z_base_scaled[N_PER_SCENARIO:2*N_PER_SCENARIO]
    z_B585_scaled = z_base_scaled[2*N_PER_SCENARIO:]
    
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
        
        # Centroide del cluster en BASE (espacio escalado)
        centroid = centroids_base[cluster_id]
        
        # Compacidad BASE (promedio de los 3 escenarios base) — EN ESPACIO ESCALADO
        comp_base_245 = np.linalg.norm(z_B245_scaled[mask_B245] - centroid, axis=1).mean() if mask_B245.any() else 0
        comp_base_370 = np.linalg.norm(z_B370_scaled[mask_B370] - centroid, axis=1).mean() if mask_B370.any() else 0
        comp_base_585 = np.linalg.norm(z_B585_scaled[mask_B585] - centroid, axis=1).mean() if mask_B585.any() else 0
        compactness_base = (comp_base_245 + comp_base_370 + comp_base_585) / 3
        
        # Compacidad en FUTUROS — EN ESPACIO ESCALADO
        # Usamos la máscara de cada escenario BASE correspondiente para trackear los mismos píxeles
        compactness_T245 = np.linalg.norm(z_T245_scaled[mask_B245] - centroid, axis=1).mean() if mask_B245.any() else 0
        compactness_T370 = np.linalg.norm(z_T370_scaled[mask_B370] - centroid, axis=1).mean() if mask_B370.any() else 0
        compactness_T585 = np.linalg.norm(z_T585_scaled[mask_B585] - centroid, axis=1).mean() if mask_B585.any() else 0
        
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
        "labels_B245": labels_B245,
        "labels_B370": labels_B370,
        "labels_B585": labels_B585,
        "resilience_df": df_cluster_resilience,
        "scaler": scaler,
        "centroids": centroids_base,
        "z_B245_scaled": z_B245_scaled,
        "z_B370_scaled": z_B370_scaled,
        "z_B585_scaled": z_B585_scaled,
        "z_T245_scaled": z_T245_scaled,
        "z_T370_scaled": z_T370_scaled,
        "z_T585_scaled": z_T585_scaled,
    }
    
    print(f"\nMétricas de resiliencia por cluster:")
    print(df_cluster_resilience[["cluster_id", "n_pixels", "expansion_T585", "stability_T585"]].to_string(index=False))
    
    print(f"\nInterpretación:")
    print(f"  Expansión T585: {df_cluster_resilience['expansion_T585'].mean():.1f}% promedio")
    print(f"  Estabilidad T585: {df_cluster_resilience['stability_T585'].mean():.1f}% promedio")
    print(f"  → Cluster más resiliente: {df_cluster_resilience.loc[df_cluster_resilience['expansion_T585'].idxmin(), 'cluster_id']}")
    print(f"  → Cluster menos resiliente: {df_cluster_resilience.loc[df_cluster_resilience['expansion_T585'].idxmax(), 'cluster_id']}")
    print(f"\n→ Clustering realizado en espacio latente estandarizado para distancias equilibradas")


# %%
# Visualización de clusters en espacio latente 2D (PCA)
# Formato: Comparación BASE vs TARGET lado a lado

for model_key in MODEL_ORDER:
    print(f"\n{'='*60}")
    print(f"Clusters en espacio latente — {model_key}")
    print(f"{'='*60}")
    
    results = CLUSTERING_RESULTS[model_key]
    scaler = results["scaler"]
    
    # Obtener latentes ORIGINALES (sin escalar)
    z_base = LATENTS[model_key]["base"]
    z_T245 = LATENTS[model_key]["T245"]
    z_T370 = LATENTS[model_key]["T370"]
    z_T585 = LATENTS[model_key]["T585"]
    
    # Escalar con el mismo scaler usado en clustering
    z_base_scaled = scaler.transform(z_base)
    z_T245_scaled = scaler.transform(z_T245)
    z_T370_scaled = scaler.transform(z_T370)
    z_T585_scaled = scaler.transform(z_T585)
    
    labels_base = results["labels_base"]
    labels_T245 = results["labels_T245"]
    labels_T370 = results["labels_T370"]
    labels_T585 = results["labels_T585"]
    
    # Separar BASE por escenarios
    labels_B245 = labels_base[:N_PER_SCENARIO]
    labels_B370 = labels_base[N_PER_SCENARIO:2*N_PER_SCENARIO]
    labels_B585 = labels_base[2*N_PER_SCENARIO:]
    
    z_B245_scaled = z_base_scaled[:N_PER_SCENARIO]
    z_B370_scaled = z_base_scaled[N_PER_SCENARIO:2*N_PER_SCENARIO]
    z_B585_scaled = z_base_scaled[2*N_PER_SCENARIO:]
    
    centroids_base = results["kmeans"].cluster_centers_  # Ya en espacio escalado
    
    # PCA 2D para visualización - APLICAR AL ESPACIO ESCALADO
    pca = PCA(n_components=2, random_state=SEED)
    z_base_2d = pca.fit_transform(z_base_scaled)
    z_T245_2d = pca.transform(z_T245_scaled)
    z_T370_2d = pca.transform(z_T370_scaled)
    z_T585_2d = pca.transform(z_T585_scaled)
    centroids_2d = pca.transform(centroids_base)
    
    # Separar BASE 2D por escenarios
    z_B245_2d = z_base_2d[:N_PER_SCENARIO]
    z_B370_2d = z_base_2d[N_PER_SCENARIO:2*N_PER_SCENARIO]
    z_B585_2d = z_base_2d[2*N_PER_SCENARIO:]
    
    print(f"Varianza explicada por PC1+PC2: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    
    # Colormap consistente usando tab20
    color_palette = plt.get_cmap('tab20', 20)
    
    # Función auxiliar para graficar un panel
    def plot_latent_panel(ax, z_2d, labels, centroids_base, title, show_legend=False, 
                          centroids_target=None, show_displacement=False):
        """
        Grafica un panel del espacio latente con clusters, centroides y elipses envolventes.
        
        Args:
            ax: Axes de matplotlib
            z_2d: Coordenadas 2D de los puntos (N, 2)
            labels: Labels de cluster (N,)
            centroids_base: Centroides BASE en 2D (K_CLUSTERS, 2)
            title: Título del panel
            show_legend: Si mostrar leyenda
            centroids_target: Centroides TARGET en 2D (K_CLUSTERS, 2) - solo para panel TARGET
            show_displacement: Si dibujar líneas de desplazamiento BASE→TARGET
        """
        from matplotlib.patches import Ellipse
        
        # Identificar clusters presentes en este panel
        unique_clusters = np.unique(labels)
        
        for cluster_id in unique_clusters:
            mask = labels == cluster_id
            n_points = mask.sum()
            
            if n_points == 0:
                continue
            
            # Puntos del cluster
            points = z_2d[mask]
            color_idx = int(cluster_id) % 20
            cluster_color = color_palette(color_idx)
            
            # Scatter de puntos
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
            
            # Calcular elipse envolvente (95% de confianza)
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
        
        # Dibujar centroides y desplazamientos
        if centroids_target is not None and show_displacement:
            # Panel TARGET: mostrar centroides BASE (gris) y TARGET (rojo) con líneas
            for cluster_id in unique_clusters:
                centroid_base = centroids_base[cluster_id]
                centroid_target = centroids_target[cluster_id]
                
                # Validar que ambos centroides sean válidos (no NaN)
                if np.isnan(centroid_base).any() or np.isnan(centroid_target).any():
                    continue
                
                # Línea de desplazamiento BASE → TARGET
                ax.plot(
                    [centroid_base[0], centroid_target[0]],
                    [centroid_base[1], centroid_target[1]],
                    'k--',
                    linewidth=1.5,
                    alpha=0.5,
                    zorder=8
                )
                
                # Flecha en la punta
                ax.annotate(
                    '',
                    xy=(centroid_target[0], centroid_target[1]),
                    xytext=(centroid_base[0], centroid_base[1]),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black', alpha=0.6),
                    zorder=8
                )
                
                # Centroide BASE (gris, más pequeño)
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
                
                # Centroide TARGET (rojo, más grande)
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
                
                # Distancia euclidiana
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
            # Panel BASE: solo centroides BASE (negro)
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
    
    # ===== SSP245: B245 (izq) vs T245 (der) =====
    print(f"\nSSP245: BASE vs TARGET")
    
    # Calcular centroides BASE específicos de B245 (mismo subset que se muestra a la izquierda)
    centroids_B245_2d = np.array([z_B245_2d[labels_B245 == cid].mean(axis=0) for cid in range(K_CLUSTERS)])
    
    # Calcular centroides TARGET para SSP245
    unique_T245 = np.unique(labels_T245)
    centroids_T245_2d = np.array([z_T245_2d[labels_T245 == cid].mean(axis=0) for cid in range(K_CLUSTERS)])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    # Panel BASE - usar centroides B245
    plot_latent_panel(
        axes[0], z_B245_2d, labels_B245, centroids_B245_2d, 
        "BASE (2020-2029)", 
        show_legend=True,
        centroids_target=None,
        show_displacement=False
    )
    axes[0].set_ylabel("PC2", fontsize=10)
    axes[0].set_xlabel("PC1", fontsize=10)
    
    # Panel TARGET con desplazamientos - usar centroides B245 como referencia BASE
    plot_latent_panel(
        axes[1], z_T245_2d, labels_T245, centroids_B245_2d, 
        "TARGET (2090-2100)", 
        show_legend=True,
        centroids_target=centroids_T245_2d,
        show_displacement=True
    )
    axes[1].set_xlabel("PC1", fontsize=10)
    
    fig.suptitle(f"SSP245: Evolución de clusters en espacio latente (PCA) — {model_key}", 
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    # ===== SSP370: B370 (izq) vs T370 (der) =====
    print(f"\nSSP370: BASE vs TARGET")
    
    # Calcular centroides BASE específicos de B370
    centroids_B370_2d = np.array([z_B370_2d[labels_B370 == cid].mean(axis=0) for cid in range(K_CLUSTERS)])
    
    # Calcular centroides TARGET para SSP370
    unique_T370 = np.unique(labels_T370)
    centroids_T370_2d = np.array([z_T370_2d[labels_T370 == cid].mean(axis=0) for cid in range(K_CLUSTERS)])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    # Panel BASE - usar centroides B370
    plot_latent_panel(
        axes[0], z_B370_2d, labels_B370, centroids_B370_2d, 
        "BASE (2020-2029)", 
        show_legend=True,
        centroids_target=None,
        show_displacement=False
    )
    axes[0].set_ylabel("PC2", fontsize=10)
    axes[0].set_xlabel("PC1", fontsize=10)
    
    # Panel TARGET con desplazamientos - usar centroides B370 como referencia BASE
    plot_latent_panel(
        axes[1], z_T370_2d, labels_T370, centroids_B370_2d, 
        "TARGET (2090-2100)", 
        show_legend=True,
        centroids_target=centroids_T370_2d,
        show_displacement=True
    )
    axes[1].set_xlabel("PC1", fontsize=10)
    
    fig.suptitle(f"SSP370: Evolución de clusters en espacio latente (PCA) — {model_key}", 
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    # ===== SSP585: B585 (izq) vs T585 (der) =====
    print(f"\nSSP585: BASE vs TARGET")
    
    # Calcular centroides BASE específicos de B585
    centroids_B585_2d = np.array([z_B585_2d[labels_B585 == cid].mean(axis=0) for cid in range(K_CLUSTERS)])
    
    # Calcular centroides TARGET para SSP585
    unique_T585 = np.unique(labels_T585)
    centroids_T585_2d = np.array([z_T585_2d[labels_T585 == cid].mean(axis=0) for cid in range(K_CLUSTERS)])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    # Panel BASE - usar centroides B585
    plot_latent_panel(
        axes[0], z_B585_2d, labels_B585, centroids_B585_2d, 
        "BASE (2020-2029)", 
        show_legend=True,
        centroids_target=None,
        show_displacement=False
    )
    axes[0].set_ylabel("PC2", fontsize=10)
    axes[0].set_xlabel("PC1", fontsize=10)
    
    # Panel TARGET con desplazamientos - usar centroides B585 como referencia BASE
    plot_latent_panel(
        axes[1], z_T585_2d, labels_T585, centroids_B585_2d, 
        "TARGET (2090-2100)", 
        show_legend=True,
        centroids_target=centroids_T585_2d,
        show_displacement=True
    )
    axes[1].set_xlabel("PC1", fontsize=10)
    
    fig.suptitle(f"SSP585: Evolución de clusters en espacio latente (PCA) — {model_key}", 
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    # Estadísticas de migración entre clusters por escenario
    print(f"\n{'─'*60}")
    print(f"MATRICES DE TRANSICIÓN BASE → TARGET")
    print(f"{'─'*60}")
    
    # SSP245: B245 → T245
    print(f"\nSSP245 (B245 → T245):")
    transition_245 = np.zeros((K_CLUSTERS, K_CLUSTERS), dtype=int)
    for i in range(K_CLUSTERS):
        mask = labels_B245 == i
        for j in range(K_CLUSTERS):
            transition_245[i, j] = (labels_T245[mask] == j).sum()
    
    df_transition_245 = pd.DataFrame(
        transition_245,
        index=[f"B_C{i}" for i in range(K_CLUSTERS)],
        columns=[f"T_C{j}" for j in range(K_CLUSTERS)]
    )
    print(df_transition_245)
    
    print(f"\n  Retención de cluster (diagonal):")
    for i in range(K_CLUSTERS):
        total = transition_245[i, :].sum()
        retained = transition_245[i, i]
        pct = retained / total * 100 if total > 0 else 0
        print(f"    Cluster {i}: {retained}/{total} píxeles ({pct:.1f}%)")
    
    # SSP370: B370 → T370
    print(f"\nSSP370 (B370 → T370):")
    transition_370 = np.zeros((K_CLUSTERS, K_CLUSTERS), dtype=int)
    for i in range(K_CLUSTERS):
        mask = labels_B370 == i
        for j in range(K_CLUSTERS):
            transition_370[i, j] = (labels_T370[mask] == j).sum()
    
    df_transition_370 = pd.DataFrame(
        transition_370,
        index=[f"B_C{i}" for i in range(K_CLUSTERS)],
        columns=[f"T_C{j}" for j in range(K_CLUSTERS)]
    )
    print(df_transition_370)
    
    print(f"\n  Retención de cluster (diagonal):")
    for i in range(K_CLUSTERS):
        total = transition_370[i, :].sum()
        retained = transition_370[i, i]
        pct = retained / total * 100 if total > 0 else 0
        print(f"    Cluster {i}: {retained}/{total} píxeles ({pct:.1f}%)")
    
    # SSP585: B585 → T585
    print(f"\nSSP585 (B585 → T585):")
    transition_585 = np.zeros((K_CLUSTERS, K_CLUSTERS), dtype=int)
    for i in range(K_CLUSTERS):
        mask = labels_B585 == i
        for j in range(K_CLUSTERS):
            transition_585[i, j] = (labels_T585[mask] == j).sum()
    
    df_transition_585 = pd.DataFrame(
        transition_585,
        index=[f"B_C{i}" for i in range(K_CLUSTERS)],
        columns=[f"T_C{j}" for j in range(K_CLUSTERS)]
    )
    print(df_transition_585)
    
    print(f"\n  Retención de cluster (diagonal):")
    for i in range(K_CLUSTERS):
        total = transition_585[i, :].sum()
        retained = transition_585[i, i]
        pct = retained / total * 100 if total > 0 else 0
        print(f"    Cluster {i}: {retained}/{total} píxeles ({pct:.1f}%)")



# %%
# Visualización espacial de los clusters BASE y futuros (TARGET)
# Formato: Comparación lado a lado (BASE | TARGET)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

def plot_spatial_comparison_inline(labels_base, labels_target, title_base, title_target, suptitle, alpha=0.75):
    """
    Grafica dos mapas lado a lado: BASE (izquierda) y TARGET (derecha)
    """
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
        xs, ys = _get_mercator_coords()
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
            ctx.add_basemap(axes[0], source=ctx.providers.CartoDB.Positron, crs="EPSG:3857", alpha=1.0, attribution_size=6)
        except:
            pass
        
        boundaries = np.arange(len(all_unique_vals) + 1) - 0.5
        norm = BoundaryNorm(boundaries, discrete_cmap.N)
        
        axes[0].imshow(pred_base, extent=extent, origin="lower", cmap=discrete_cmap, norm=norm, alpha=alpha, zorder=3)
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
            ctx.add_basemap(axes[1], source=ctx.providers.CartoDB.Positron, crs="EPSG:3857", alpha=1.0, attribution_size=6)
        except:
            pass
        
        axes[1].imshow(pred_target, extent=extent, origin="lower", cmap=discrete_cmap, norm=norm, alpha=alpha, zorder=3)
        axes[1].set_axis_off()
        axes[1].set_title(title_target, fontsize=11, pad=10)
        
        mappable = ScalarMappable(norm=norm, cmap=discrete_cmap)
        cbar = fig.colorbar(mappable, ax=axes, fraction=0.03, pad=0.02, ticks=np.arange(len(all_unique_vals)))
        cbar.set_ticklabels([str(int(val)) for val in all_unique_vals])
        cbar.set_label("Cluster ID", fontsize=10)
        
        fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
    except Exception as err:
        print(f"Error en plot_spatial_comparison: {err}")
        plt.close(fig)

# Comparaciones BASE → TARGET
for model_key in MODEL_ORDER:
    print(f"\n{'='*60}")
    print(f"Comparación BASE → TARGET — {model_key}")
    print(f"{'='*60}")
    
    results = CLUSTERING_RESULTS[model_key]
    labels_base = results["labels_base"]
    labels_T245 = results["labels_T245"]
    labels_T370 = results["labels_T370"]
    labels_T585 = results["labels_T585"]
    
    labels_B245 = labels_base[:N_PER_SCENARIO]
    labels_B370 = labels_base[N_PER_SCENARIO:2*N_PER_SCENARIO]
    labels_B585 = labels_base[2*N_PER_SCENARIO:]
    
    print(f"\nClusters únicos: {np.unique(labels_base)}")
    print(f"Mostrando 3 comparaciones BASE → TARGET (una por trayectoria SSP)\n")
    
    changed_245 = (labels_T245 != labels_B245).sum()
    pct_changed_245 = (changed_245 / len(labels_T245)) * 100
    print(f"SSP245: {changed_245}/{len(labels_T245)} píxeles transicionan ({pct_changed_245:.1f}%)")
    
    plot_spatial_comparison_inline(
        labels_B245, labels_T245,
        title_base="BASE (2020-2029)",
        title_target="TARGET (2090-2100)",
        suptitle=f"SSP245 — {model_key}",
        alpha=0.75
    )
    
    changed_370 = (labels_T370 != labels_B370).sum()
    pct_changed_370 = (changed_370 / len(labels_T370)) * 100
    print(f"\nSSP370: {changed_370}/{len(labels_T370)} píxeles transicionan ({pct_changed_370:.1f}%)")
    
    plot_spatial_comparison_inline(
        labels_B370, labels_T370,
        title_base="BASE (2020-2029)",
        title_target="TARGET (2090-2100)",
        suptitle=f"SSP370 — {model_key}",
        alpha=0.75
    )
    
    changed_585 = (labels_T585 != labels_B585).sum()
    pct_changed_585 = (changed_585 / len(labels_T585)) * 100
    print(f"\nSSP585: {changed_585}/{len(labels_T585)} píxeles transicionan ({pct_changed_585:.1f}%)")
    
    plot_spatial_comparison_inline(
        labels_B585, labels_T585,
        title_base="BASE (2020-2029)",
        title_target="TARGET (2090-2100)",
        suptitle=f"SSP585 — {model_key}",
        alpha=0.75
    )
    
    print(f"\n{'─'*60}")
    print(f"RESUMEN: Transiciones de cluster BASE → TARGET")
    print(f"{'─'*60}")
    print(f"  SSP245: {pct_changed_245:.1f}% de píxeles cambian de cluster")
    print(f"  SSP370: {pct_changed_370:.1f}% de píxeles cambian de cluster")
    print(f"  SSP585: {pct_changed_585:.1f}% de píxeles cambian de cluster")
    print(f"\n  → Mayor % = mayor reordenamiento espacial del perfil climático")


# %% [markdown]
# ## 5. Implementación del IRCT
# 
# Definición de funciones para calcular cada componente del índice siguiendo las ecuaciones matemáticas

# %% [markdown]
# ### 5.1 Función principal de cálculo del IRCT

# %%
# 5) Percentile rank vectorizado (más rápido que loop + percentileofscore)
def percentile_rank(arr):
    arr = np.asarray(arr)
    n = len(arr)
    if n <= 1:
        return np.zeros_like(arr, dtype=float)
    # => promedio en empates y mapeo [0,1] con extremos exactos
    #    rank: 1..n (average ties)  → (rank-1)/(n-1) ∈ [0,1]
    import pandas as _pd  # => usar rank con empates promediados
    ranks = _pd.Series(arr).rank(method="average").to_numpy()
    return (ranks - 1.0) / (n - 1.0)

def compute_cluster_stability_softmax(z_scaled, centroids_base, labels_base, tau=1.0):
    z = np.asarray(z_scaled)
    C = np.asarray(centroids_base)

    # Distancias euclidianas (vectorizado)
    zz = np.sum(z*z, axis=1, keepdims=True)
    cc = np.sum(C*C, axis=1, keepdims=True).T
    all_distances = np.sqrt(np.maximum(zz + cc - 2*np.dot(z, C.T), 0.0))

    # => normalizar distancias por su desviación típica global para comparabilidad
    #    (así AE/VAE entran al softmax en escalas similares)
    d_std = np.std(all_distances)
    d_std = d_std if d_std > 0 else 1.0
    norm_d = all_distances / d_std  # => nuevo

    # Softmax estable con temperatura
    logits = -norm_d / max(tau, 1e-6)  # => usar distancias normalizadas
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    softmax_probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)

    idx = np.arange(len(z))
    S_C = softmax_probs[idx, labels_base]
    return S_C, all_distances


def compute_latent_displacement(z_scaled, centroids_base, labels_base, *, inv_covs_base=None):
    z = np.asarray(z_scaled)
    C = np.asarray(centroids_base)
    n = len(z)
    distances = np.empty(n)

    if inv_covs_base is None:
        # Euclídea (tu versión)
        for i in range(n):
            k = labels_base[i]
            distances[i] = np.linalg.norm(z[i] - C[k])
    else:
        # Mahalanobis por clúster base
        for i in range(n):
            k = labels_base[i]
            d = z[i] - C[k]
            distances[i] = np.sqrt(np.dot(d, inv_covs_base[k].dot(d)))

    S_D = 1 - percentile_rank(distances)
    return S_D, distances


def compute_cluster_expansion(z_base_scaled, z_future_scaled, centroids_base, labels_base, epsilon=1e-8, p99_clip=False):
    zB, zF, C = map(np.asarray, (z_base_scaled, z_future_scaled, centroids_base))
    n = len(zB)
    B = np.empty(n); F = np.empty(n)

    for i in range(n):
        k = labels_base[i]
        B[i] = np.linalg.norm(zB[i] - C[k])
        F[i] = np.linalg.norm(zF[i] - C[k])

    ratio = F / (B + epsilon)
    # => cambio simétrico: magnitud del cambio
    delta = np.abs(np.log(np.clip(ratio, 1e-12, None)))  # => |log(F/B)|

    if p99_clip:
        p99 = np.percentile(delta, 99)
        delta = np.clip(delta, 0, p99)

    S_E = 1 - percentile_rank(delta)  # => 1 = estable (poco cambio), 0 = inestable
    return S_E, ratio  # => dejo ratio por trazabilidad


def compute_h2_stability(h2_base, h2_future, epsilon=1e-8, p99_clip=True):
    """
    Componente 5: Estabilidad energética del hidrógeno
    
    R_H2,i^(s) = clip(h2_future_i / (h2_base_i + epsilon), 0, p99)
    S_H2,i^(s) = 1 - perc(R_H2,i^(s))
    
    Retorna:
        S_H2: estabilidad energética (0-1, 1 es más resiliente)
        h2_ratios: ratios de cambio de H2
    """
    h2_ratios = h2_future / (h2_base + epsilon)
    
    if p99_clip:
        p99 = np.percentile(h2_ratios, 99)
        h2_ratios = np.clip(h2_ratios, 0, p99)
    else:
        h2_ratios = np.clip(h2_ratios, 0, None)
    
    # Percentil rank
    perc_ranks = percentile_rank(h2_ratios)
    
    # Estabilidad (invertir para que 1 = más resiliente)
    S_H2 = 1 - perc_ranks
    
    return S_H2, h2_ratios
def compute_reconstruction_anomaly(
    model,
    X_orig,
    X_normalized,
    device='cpu',
    *,
    use_normalized=True,
    inverse_transform=None,
    reduce='mse',             # => nuevo: permite 'mse' o 'mae' para la reconstrucción
    output='stability'        # => nuevo: 'stability' (1=mejor) o 'anomaly' (1=peor) para evitar invertir después
):
    """
    Calcula estabilidad/anomalía de reconstrucción por muestra.

    reduce: 'mse' (default) o 'mae'                 # => opción de métrica
    output: 'stability' (1=mejor) o 'anomaly' (1=peor)  # => controla la orientación del índice
    """
    model.eval()
    with torch.no_grad():
        X_in = X_normalized if use_normalized else X_orig          # => claridad en la entrada efectiva
        X_tensor = torch.as_tensor(X_in, dtype=torch.float32, device=device)
        
        # Intentar desempacar de forma robusta
        model_output = model(X_tensor)
        if isinstance(model_output, tuple):
            if len(model_output) == 3:
                # VAE retorna (x_hat, mu, logvar)
                x_hat = model_output[0]
            elif len(model_output) == 2:
                # AE retorna (x_hat, z)
                x_hat = model_output[0]
            else:
                # Otro formato, asumir primer elemento es reconstrucción
                x_hat = model_output[0]
        else:
            # Retorna solo x_hat
            x_hat = model_output
        
        x_hat_np = x_hat.cpu().numpy()

    # => Alinear escalas de comparación según la entrada usada
    if use_normalized:
        target = np.asarray(X_normalized)
    else:
        if inverse_transform is None:
            raise ValueError("Proporciona inverse_transform para des-normalizar x_hat_np.")  # => validación explícita
        x_hat_np = inverse_transform(x_hat_np)
        target = np.asarray(X_orig)

    # => Métrica de error seleccionable (robusta a outliers si 'mae')
    if reduce == 'mae':
        reconstruction_errors = np.mean(np.abs(target - x_hat_np), axis=1)
    else:
        reconstruction_errors = np.mean((target - x_hat_np) ** 2, axis=1)

    # => Percentil robusto: empates promediados y mapeo exacto a [0,1]
    try:
        import pandas as _pd
        ranks = _pd.Series(reconstruction_errors).rank(method="average").to_numpy()
        n = len(reconstruction_errors)
        pr = (ranks - 1.0) / (n - 1.0) if n > 1 else np.zeros_like(reconstruction_errors, dtype=float)
    except Exception:
        # Fallback simple si pandas no está disponible
        order = np.argsort(reconstruction_errors)
        pr = np.empty_like(reconstruction_errors, dtype=float)
        pr[order] = np.linspace(0.0, 1.0, len(reconstruction_errors))

    # => Salida orientada: estabilidad (alta=mejor) o anomalía (alta=peor)
    if output == 'stability':
        A = 1.0 - pr
    elif output == 'anomaly':
        A = pr
    else:
        raise ValueError("output debe ser 'stability' o 'anomaly'.")

    return A, reconstruction_errors


print("Funciones de componentes del IRCT definidas:")
print("  1. compute_reconstruction_anomaly()")
print("  2. compute_latent_displacement()")
print("  3. compute_cluster_stability_softmax()")
print("  4. compute_cluster_expansion()")
print("  5. compute_h2_stability()")

# %%
def compute_IRCT_pixel_wise(
    model,
    X_base_orig,
    X_base_norm,
    X_future_orig,
    X_future_norm,
    z_base_scaled,
    z_future_scaled,
    centroids_base,
    labels_base,
    h2_base=None,
    h2_future=None,
    weights=None,
    device='cpu',
    *,
    # extras opcionales
    recon_use_normalized=True,           # => usar misma escala en reconstrucción
    inverse_transform=None,              # => necesario si recon_use_normalized=False
    softmax_tau=1.0,                     # => temperatura para softmax
    expansion_p99_clip=False,            # => recorte p99 en expansión
    h2_p99_clip=True,                    # => recorte p99 en H2
    eps=1e-8                             # => piso numérico para media geométrica
):
    """
    IRCT_i^(s) = (A_i)^{w_a} * (S_D,i)^{w_d} * (S_C,i)^{w_c} * (S_E,i)^{w_e} * (S_H2,i)^{w_h}
    Devuelve dict con IRCT y componentes intermedios.
    """
    # 0) Pesos por defecto + normalización
    if weights is None:
        weights = {'w_a':0.15,'w_d':0.30,'w_c':0.25,'w_e':0.20,'w_h':0.10}
    wsum = sum(weights.values())
    if not np.isclose(wsum, 1.0):
        weights = {k: v / (wsum if wsum else 1.0) for k, v in weights.items()}

    # 1) Anomalía de reconstrucción (usar misma escala)
    result_A = compute_reconstruction_anomaly(
        model,
        X_future_orig,
        X_future_norm,
        device=device,
        use_normalized=recon_use_normalized,
        inverse_transform=inverse_transform
    )
    # Manejo robusto: puede retornar tupla o valor simple
    if isinstance(result_A, tuple):
        A, recon_errors = result_A
    else:
        A = result_A
        recon_errors = np.zeros_like(A)

    # 2) Desplazamiento latente
    result_S_D = compute_latent_displacement(
        z_future_scaled, centroids_base, labels_base
    )
    if isinstance(result_S_D, tuple):
        S_D, latent_distances = result_S_D
    else:
        S_D = result_S_D
        latent_distances = np.zeros_like(S_D)

    # 3) Estabilidad de pertenencia (softmax estable con tau)
    result_S_C = compute_cluster_stability_softmax(
        z_future_scaled, centroids_base, labels_base, tau=softmax_tau
    )
    if isinstance(result_S_C, tuple):
        S_C, all_distances = result_S_C
    else:
        S_C = result_S_C
        all_distances = np.zeros_like(S_C)

    # 4) Expansión del clúster
    result_S_E = compute_cluster_expansion(
        z_base_scaled, z_future_scaled, centroids_base, labels_base,
        p99_clip=expansion_p99_clip
    )
    if isinstance(result_S_E, tuple):
        S_E, expansion_ratios = result_S_E
    else:
        S_E = result_S_E
        expansion_ratios = np.zeros_like(S_E)

    # 5) Estabilidad H2
    if h2_base is not None and h2_future is not None:
        result_S_H2 = compute_h2_stability(h2_base, h2_future, p99_clip=h2_p99_clip)
        if isinstance(result_S_H2, tuple):
            S_H2, h2_ratios = result_S_H2
        else:
            S_H2 = result_S_H2
            h2_ratios = np.zeros_like(S_H2)
        use_h2 = True
    else:
        n = len(A)
        S_H2 = np.ones(n)
        h2_ratios = np.ones(n)
        use_h2 = False
        print("ADVERTENCIA: Sin datos de H2 → S_H2 = 1.0")

    # 6) Chequeos rápidos de integridad
    n = len(A)
    assert all(len(x) == n for x in [S_D, S_C, S_E, S_H2]), "Longitudes inconsistentes"
    # saneo NaN/inf
    for arr in (A, S_D, S_C, S_E, S_H2):
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(arr, 0.0, 1.0, out=arr)

    # 7) IRCT como media geométrica ponderada en locompute_latent_displacementg-espacio (estable)
    # log(IRCT) = Σ w_k * log(max(comp_k, eps))
    log_IRCT = (
        weights['w_a'] * np.log(np.maximum(A,   eps)) +
        weights['w_d'] * np.log(np.maximum(S_D, eps)) +
        weights['w_c'] * np.log(np.maximum(S_C, eps)) +
        weights['w_e'] * np.log(np.maximum(S_E, eps)) +
        weights['w_h'] * np.log(np.maximum(S_H2,eps))
    )
    IRCT = np.exp(log_IRCT)

    return {
        'IRCT': IRCT,
        'A': A,
        'S_D': S_D,
        'S_C': S_C,
        'S_E': S_E,
        'S_H2': S_H2,
        'reconstruction_errors': recon_errors,
        'latent_distances': latent_distances,
        'cluster_distances_all': all_distances,
        'expansion_ratios': expansion_ratios,
        'h2_ratios': h2_ratios,
        'weights': weights,
        'use_h2': use_h2
    }


# %% [markdown]
# ## 6. Cálculo del IRCT para todos los modelos y escenarios
# 
# Aplicamos el índice a cada modelo AE/VAE y cada escenario SSP

# %% [markdown]
# ### 6.2 Corrección de features para datos futuros
# 
# Ajustamos los datos futuros para que tengan exactamente las mismas features que se usaron en el entrenamiento

# %%
print("CORRECCIÓN DE FEATURES PARA DATOS FUTUROS")
print("="*80)
print()

# Obtener el número de features que esperan los modelos
input_dim_expected = model_dims[MODEL_ORDER[0]]['input_dim']

print(f"Los modelos fueron entrenados con {input_dim_expected} features")
print(f"X_BASE tiene {X_BASE.shape[1]} features")
print(f"X245_orig tiene {X245_orig.shape[1]} features")
print()

if X_BASE.shape[1] == input_dim_expected:
    print("✓ X_BASE tiene el número correcto de features")
    print()
    
    # X_BASE está correcto, usar sus features como referencia
    if X245_orig.shape[1] != input_dim_expected:
        print(f"⚠ Los datos futuros tienen {X245_orig.shape[1]} features, pero se esperan {input_dim_expected}")
        print()
        print("Identificando features correctas...")
        
        # Las features usadas en entrenamiento son las primeras input_dim_expected de feature_names
        features_used = feature_names[:input_dim_expected]
        
        print(f"Features usadas en entrenamiento: {features_used}")
        print()
        
        # Recortar los datos futuros a las mismas features
        X245_orig_corrected = X245_orig[:, :input_dim_expected]
        X370_orig_corrected = X370_orig[:, :input_dim_expected]
        X585_orig_corrected = X585_orig[:, :input_dim_expected]
        
        if X245_norm is not None:
            X245_norm_corrected = X245_norm[:, :input_dim_expected]
            X370_norm_corrected = X370_norm[:, :input_dim_expected]
            X585_norm_corrected = X585_norm[:, :input_dim_expected]
        else:
            X245_norm_corrected = X245_orig_corrected
            X370_norm_corrected = X370_orig_corrected
            X585_norm_corrected = X585_orig_corrected
        
        # Reemplazar los datos originales
        X245_orig = X245_orig_corrected
        X370_orig = X370_orig_corrected
        X585_orig = X585_orig_corrected
        
        X245_norm = X245_norm_corrected
        X370_norm = X370_norm_corrected
        X585_norm = X585_norm_corrected
        
        # Actualizar feature_names también
        feature_names = features_used
        
        print("✓ Datos futuros corregidos:")
        print(f"  X245_orig: {X245_orig.shape}")
        print(f"  X370_orig: {X370_orig.shape}")
        print(f"  X585_orig: {X585_orig.shape}")
        print()
        
        # Actualizar el índice de H2 si existe
        try:
            h2_idx_new = feature_names.index('h2_production')
            print(f"  Nuevo índice de H2: {h2_idx_new}")
        except ValueError:
            print("  H2 no está en las features usadas")
        print()
        
    else:
        print("✓ Los datos futuros ya tienen el número correcto de features")
        print()
        
else:
    print(f"⚠ PROBLEMA: X_BASE tiene {X_BASE.shape[1]} features pero el modelo espera {input_dim_expected}")
    print("Esto indica un problema en la carga de datos del experimento 1")
    print()

print("="*80)
print("✓ CORRECCIÓN COMPLETADA")
print("="*80)

# %% [markdown]
# ### 6.3 Identificación de features de H₂
# 
# Identificamos correctamente las columnas relacionadas con producción de hidrógeno

# %%
print("IDENTIFICACIÓN DE FEATURES DE H₂")
print("="*80)
print()

# Buscar todas las features relacionadas con H2
h2_features = [f for f in feature_names if 'h2' in f.lower() or 'hydrogen' in f.lower()]

print(f"Features de H₂ encontradas: {len(h2_features)}")
for i, feat in enumerate(h2_features):
    idx = feature_names.index(feat)
    print(f"  [{idx}] {feat}")
print()

if len(h2_features) > 0:
    # Para el cálculo del IRCT, usaremos la feature más reciente (2080 para BASE, o la última disponible)
    # Esto representa el potencial de H2 en el periodo BASE
    
    # Identificar la feature de H2 para el periodo BASE (2020)
    h2_base_feature = None
    for feat in h2_features:
        if '2020' in feat:
            h2_base_feature = feat
            break
    
    if h2_base_feature is None:
        # Si no hay 2020, usar la primera disponible
        h2_base_feature = h2_features[0]
    
    h2_base_idx = feature_names.index(h2_base_feature)
    
    print(f"Feature de H₂ para periodo BASE: {h2_base_feature} (índice {h2_base_idx})")
    print()
    
    # Para escenarios futuros, necesitamos identificar cuál columna usar
    # Dado que los datos futuros son proyecciones 2090-2100, usaremos la feature más lejana (2080)
    h2_future_feature = None
    for feat in h2_features:
        if '2080' in feat:
            h2_future_feature = feat
            break
    
    if h2_future_feature is None:
        # Si no hay 2080, usar la última disponible
        h2_future_feature = h2_features[-1]
    
    h2_future_idx = feature_names.index(h2_future_feature)
    
    print(f"Feature de H₂ para escenarios futuros: {h2_future_feature} (índice {h2_future_idx})")
    print()
    
    # IMPORTANTE: Para el cálculo del IRCT, compararemos:
    # - BASE: h2_base_feature (ej: 2020)
    # - FUTUROS: h2_future_feature (ej: 2080) del mismo escenario SSP
    
    # Sin embargo, hay una consideración: el H2 futuro debería venir de las proyecciones SSP, no del BASE
    # Dado que X_BASE contiene datos históricos, usaremos el mismo índice para ambos
    
    print("ESTRATEGIA DE CÁLCULO:")
    print(f"  • BASE: Usaremos X_BASE[:, {h2_future_idx}] = '{h2_future_feature}'")
    print(f"  • SSP245/370/585: Usaremos X[245/370/585]_orig[:, {h2_future_idx}] = '{h2_future_feature}'")
    print()
    print("  Esto permite comparar el potencial de H₂ proyectado en el periodo futuro")
    print("  bajo clima histórico (BASE) vs. bajo diferentes escenarios de cambio climático (SSP)")
    print()
    
    # Guardar los índices para usar en el cálculo del IRCT
    H2_IDX_BASE = h2_base_idx
    H2_IDX_FUTURE = h2_future_idx
    h2_available = True
    
    print(f"✓ Índices guardados:")
    print(f"  H2_IDX_BASE = {H2_IDX_BASE}")
    print(f"  H2_IDX_FUTURE = {H2_IDX_FUTURE}")
    print()
    
else:
    print("⚠ No se encontraron features de H₂")
    H2_IDX_BASE = None
    H2_IDX_FUTURE = None
    h2_available = False
    print()

print("="*80)
print("✓ IDENTIFICACIÓN COMPLETADA")
print("="*80)

# %%
IRCT_RESULTS = {}

print("CALCULANDO IRCT PIXEL-WISE PARA TODOS LOS MODELOS")
print("="*80)
print()

# Verificar disponibilidad de H2 (ya identificado en celda anterior)
if h2_available:
    print(f"✓ Variable H2 disponible:")
    print(f"    BASE: índice {H2_IDX_BASE}")
    print(f"    FUTUROS: índice {H2_IDX_FUTURE}")
else:
    print("⚠ Variable H2 NO encontrada, se omitirá en el cálculo")
print()

for model_key in MODEL_ORDER:
    print(f"\n{model_key}")
    print("-"*80)
    
    model = MODELS[model_key]
    clustering = CLUSTERING_RESULTS[model_key]
    
    centroids_base = clustering["centroids"]
    
    IRCT_RESULTS[model_key] = {}
    
    # Procesar cada escenario futuro
    scenarios = {
        'SSP245': {
            'X_orig': X245_orig,
            'X_norm': X245_norm if X245_norm is not None else X245_orig,
            'z_scaled': clustering['z_T245_scaled'],
            'labels_base_subset': clustering['labels_B245']
        },
        'SSP370': {
            'X_orig': X370_orig,
            'X_norm': X370_norm if X370_norm is not None else X370_orig,
            'z_scaled': clustering['z_T370_scaled'],
            'labels_base_subset': clustering['labels_B370']
        },
        'SSP585': {
            'X_orig': X585_orig,
            'X_norm': X585_norm if X585_norm is not None else X585_orig,
            'z_scaled': clustering['z_T585_scaled'],
            'labels_base_subset': clustering['labels_B585']
        }
    }
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n  {scenario_name}:")
        
        # Dividir X_BASE por escenario
        if scenario_name == 'SSP245':
            X_base_orig_subset = X_BASE[:N_PER_SCENARIO]
            z_base_scaled_subset = clustering['z_B245_scaled']
        elif scenario_name == 'SSP370':
            X_base_orig_subset = X_BASE[N_PER_SCENARIO:2*N_PER_SCENARIO]
            z_base_scaled_subset = clustering['z_B370_scaled']
        else:  # SSP585
            X_base_orig_subset = X_BASE[2*N_PER_SCENARIO:]
            z_base_scaled_subset = clustering['z_B585_scaled']
        
        X_base_norm_subset = X_base_orig_subset  # Ya está normalizado
        
        if h2_available:
            h2_base = X_base_orig_subset[:, H2_IDX_BASE]     # => usar el índice BASE para el denominador (periodo base)
            h2_future = scenario_data['X_orig'][:, H2_IDX_FUTURE]  # => usar el índice FUTURO para el numerador (periodo futuro)
        else:
            h2_base = None
            h2_future = None

        
        # Calcular IRCT
        irct_result = compute_IRCT_pixel_wise(
            model=model,
            X_base_orig=X_base_orig_subset,
            X_base_norm=X_base_norm_subset,
            X_future_orig=scenario_data['X_orig'],
            X_future_norm=scenario_data['X_norm'],
            z_base_scaled=z_base_scaled_subset,
            z_future_scaled=scenario_data['z_scaled'],
            centroids_base=centroids_base,
            labels_base=scenario_data['labels_base_subset'],
            h2_base=h2_base,
            h2_future=h2_future,
            device='cpu',
            recon_use_normalized=True,
            softmax_tau=1.0,
            expansion_p99_clip=False,
            eps=1e-8
        )
        
        IRCT_RESULTS[model_key][scenario_name] = irct_result
        
        # Resumen estadístico
        irct = irct_result['IRCT']
        print(f"    IRCT: mean={irct.mean():.4f}, std={irct.std():.4f}, "
              f"min={irct.min():.4f}, max={irct.max():.4f}")
        print(f"    Componentes:")
        print(f"      A (Anomalía recons.):   {irct_result['A'].mean():.4f}")
        print(f"      S_D (Desplazamiento):   {irct_result['S_D'].mean():.4f}")
        print(f"      S_C (Estabilidad):      {irct_result['S_C'].mean():.4f}")
        print(f"      S_E (Expansión):        {irct_result['S_E'].mean():.4f}")
        print(f"      S_H2 (Est. H2):         {irct_result['S_H2'].mean():.4f}")

print()
print("="*80)
print("✓ CÁLCULO PIXEL-WISE COMPLETADO")
print("="*80)

# %% [markdown]
# ## 7. Agregación por clúster
# 
# Calculamos métricas agregadas por clúster BASE siguiendo las definiciones matemáticas

# %%
def aggregate_IRCT_by_cluster(irct_result, labels_base, labels_future, k_clusters):
    """
    Agrega el IRCT por clúster BASE.
    
    Métricas calculadas:
    1. IRCT_k: mediana del IRCT de píxeles en cluster k
    2. Retention_k: % píxeles que permanecen en cluster k
    3. Expansion_k: mediana del ratio de expansión
    4. Migration_entropy_k: entropía de migración
    
    Retorna:
        DataFrame con métricas agregadas por cluster
    """
    cluster_metrics = []
    
    for cluster_id in range(k_clusters):
        # Máscara de píxeles que pertenecen al cluster BASE k
        mask_base = (labels_base == cluster_id)
        n_pixels = mask_base.sum()
        
        if n_pixels == 0:
            continue
        
        # 1. IRCT mediano del cluster
        irct_cluster = np.median(irct_result['IRCT'][mask_base])
        
        # 2. Retención (% píxeles que permanecen en el cluster)
        retention = (labels_future[mask_base] == cluster_id).sum() / n_pixels * 100
        
        # 3. Expansión mediana
        expansion_median = np.median(irct_result['expansion_ratios'][mask_base])
        
        # 4. Entropía de migración
        # Calcular distribución de destinos
        future_labels_subset = labels_future[mask_base]
        unique_labels, counts = np.unique(future_labels_subset, return_counts=True)
        probs = counts / n_pixels
        
        # H_k = -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Componentes promedio
        A_mean = np.mean(irct_result['A'][mask_base])
        S_D_mean = np.mean(irct_result['S_D'][mask_base])
        S_C_mean = np.mean(irct_result['S_C'][mask_base])
        S_E_mean = np.mean(irct_result['S_E'][mask_base])
        S_H2_mean = np.mean(irct_result['S_H2'][mask_base])
        
        cluster_metrics.append({
            'cluster_id': cluster_id,
            'n_pixels': int(n_pixels),
            'IRCT_median': irct_cluster,
            'retention_pct': retention,
            'expansion_median': expansion_median,
            'migration_entropy': entropy,
            'A_mean': A_mean,
            'S_D_mean': S_D_mean,
            'S_C_mean': S_C_mean,
            'S_E_mean': S_E_mean,
            'S_H2_mean': S_H2_mean
        })
    
    return pd.DataFrame(cluster_metrics)


IRCT_CLUSTER_RESULTS = {}

print("AGREGACIÓN POR CLÚSTER")
print("="*80)
print()

for model_key in MODEL_ORDER:
    print(f"\n{model_key}")
    print("-"*80)
    
    IRCT_CLUSTER_RESULTS[model_key] = {}
    
    scenarios = ['SSP245', 'SSP370', 'SSP585']
    labels_base_map = {
        'SSP245': CLUSTERING_RESULTS[model_key]['labels_B245'],
        'SSP370': CLUSTERING_RESULTS[model_key]['labels_B370'],
        'SSP585': CLUSTERING_RESULTS[model_key]['labels_B585']
    }
    labels_future_map = {
        'SSP245': CLUSTERING_RESULTS[model_key]['labels_T245'],
        'SSP370': CLUSTERING_RESULTS[model_key]['labels_T370'],
        'SSP585': CLUSTERING_RESULTS[model_key]['labels_T585']
    }
    
    for scenario_name in scenarios:
        print(f"\n  {scenario_name}:")
        
        irct_result = IRCT_RESULTS[model_key][scenario_name]
        labels_base = labels_base_map[scenario_name]
        labels_future = labels_future_map[scenario_name]
        
        df_cluster = aggregate_IRCT_by_cluster(
            irct_result, labels_base, labels_future, K_CLUSTERS
        )
        
        IRCT_CLUSTER_RESULTS[model_key][scenario_name] = df_cluster
        
        # Mostrar top 5 clusters más resilientes
        df_sorted = df_cluster.sort_values('IRCT_median', ascending=False)
        print(f"\n    Top 5 clusters más resilientes:")
        print(df_sorted[['cluster_id', 'IRCT_median', 'retention_pct', 'n_pixels']].head().to_string(index=False))
        
        # Estadísticas generales
        print(f"\n    Estadísticas generales:")
        print(f"      IRCT mediano global: {df_cluster['IRCT_median'].median():.4f}")
        print(f"      Retención promedio:  {df_cluster['retention_pct'].mean():.1f}%")
        print(f"      Entropía promedio:   {df_cluster['migration_entropy'].mean():.3f}")

print()
print("="*80)
print("✓ AGREGACIÓN POR CLÚSTER COMPLETADA")
print("="*80)

# %% [markdown]
# ## 8. Exportar resultados pixel-wise y cluster-wise
# 
# Guardamos los resultados en archivos CSV para análisis posterior

# %%
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(DATA_DIR, "autoencoder_results")
os.makedirs(output_dir, exist_ok=True)

print("EXPORTANDO RESULTADOS")
print("="*80)
print()

# 1. EXPORTAR PIXEL-WISE
print("1. Exportando datos pixel-wise...")

for model_key in MODEL_ORDER:
    for scenario_name in ['SSP245', 'SSP370', 'SSP585']:
        
        irct_result = IRCT_RESULTS[model_key][scenario_name]
        
        # Determinar labels_base según escenario
        if scenario_name == 'SSP245':
            labels_base = CLUSTERING_RESULTS[model_key]['labels_B245']
        elif scenario_name == 'SSP370':
            labels_base = CLUSTERING_RESULTS[model_key]['labels_B370']
        else:
            labels_base = CLUSTERING_RESULTS[model_key]['labels_B585']
        
        # Crear DataFrame
        df_pixel = pd.DataFrame({
            'lat': coords_df['lat'].values,
            'lon': coords_df['lon'].values,
            'cluster_base': labels_base,
            'IRCT': irct_result['IRCT'],
            'A_reconstruction': irct_result['A'],
            'S_D_displacement': irct_result['S_D'],
            'S_C_stability': irct_result['S_C'],
            'S_E_expansion': irct_result['S_E'],
            'S_H2_energy': irct_result['S_H2'],
            'reconstruction_error': irct_result['reconstruction_errors'],
            'latent_distance': irct_result['latent_distances'],
            'expansion_ratio': irct_result['expansion_ratios'],
            'h2_ratio': irct_result['h2_ratios']
        })
        
        # Guardar
        filename = f"IRCT_pixel_{model_key}_{scenario_name}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        df_pixel.to_csv(filepath, index=False)
        print(f"  ✓ {filename}")

print()

# 2. EXPORTAR CLUSTER-WISE
print("2. Exportando datos cluster-wise...")

for model_key in MODEL_ORDER:
    
    # Concatenar los 3 escenarios
    df_list = []
    for scenario_name in ['SSP245', 'SSP370', 'SSP585']:
        df_cluster = IRCT_CLUSTER_RESULTS[model_key][scenario_name].copy()
        df_cluster['scenario'] = scenario_name
        df_cluster['model'] = model_key
        df_list.append(df_cluster)
    
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # Reordenar columnas
    cols_order = ['model', 'scenario', 'cluster_id', 'n_pixels', 'IRCT_median', 
                  'retention_pct', 'expansion_median', 'migration_entropy',
                  'A_mean', 'S_D_mean', 'S_C_mean', 'S_E_mean', 'S_H2_mean']
    df_combined = df_combined[cols_order]
    
    # Guardar
    filename = f"IRCT_cluster_{model_key}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    df_combined.to_csv(filepath, index=False)
    print(f"  ✓ {filename}")

print()
print("="*80)
print(f"✓ ARCHIVOS EXPORTADOS EN: {output_dir}")
print("="*80)

# %% [markdown]
# ## 9. Visualización: Mapas espaciales del IRCT
# 
# Visualización geográfica del índice de resiliencia por píxel

# %%
from pyproj import Transformer
from sklearn.neighbors import KNeighborsRegressor

def plot_irct_spatial(lat_vals, lon_vals, irct_vals, title, cmap='RdYlGn', 
                      alpha=0.8, n_neighbors=15, figsize=(14, 10), save_path=None):
    """
    Visualiza el IRCT en un mapa espacial usando interpolación KNN.
    
    Args:
        lat_vals: latitudes
        lon_vals: longitudes
        irct_vals: valores del IRCT (0-1)
        title: título del mapa
        cmap: colormap (RdYlGn = rojo=malo, amarillo=medio, verde=bueno)
        alpha: transparencia del overlay
        n_neighbors: vecinos para interpolación KNN
        figsize: tamaño de la figura
        save_path: ruta para guardar (opcional)
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Transformar a Web Mercator
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xs, ys = transformer.transform(lon_vals, lat_vals)
        
        # Crear grid para interpolación
        grid_res = 100
        grid_x = np.linspace(xs.min(), xs.max(), grid_res)
        grid_y = np.linspace(ys.min(), ys.max(), grid_res)
        GX, GY = np.meshgrid(grid_x, grid_y)
        extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
        
        # Filtrar valores válidos
        valid_mask = ~(np.isnan(irct_vals) | np.isinf(irct_vals))
        coords = np.column_stack([xs[valid_mask], ys[valid_mask]])
        vals = irct_vals[valid_mask]
        
        # Interpolación KNN
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
        knn.fit(coords, vals)
        
        grid_points = np.column_stack([GX.ravel(), GY.ravel()])
        pred = knn.predict(grid_points).reshape(GX.shape)
        
        # Basemap
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
        try:
            import contextily as ctx
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, 
                          crs="EPSG:3857", alpha=1.0, attribution_size=6)
        except:
            pass
        
        # Overlay IRCT
        im = ax.imshow(pred, extent=extent, origin="lower", 
                      cmap=cmap, alpha=alpha, vmin=0, vmax=1, zorder=3)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("IRCT (0=vulnerable, 1=resiliente)", fontsize=11)
        
        ax.set_axis_off()
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  → Guardado: {save_path}")
        
        plt.show()
        
    except Exception as err:
        print(f"Error en plot_irct_spatial: {err}")
        plt.close(fig)


print("VISUALIZACIÓN ESPACIAL DEL IRCT")
print("="*80)
print()

for model_key in MODEL_ORDER:
    print(f"\n{model_key}")
    print("-"*60)
    
    for scenario_name in ['SSP245', 'SSP370', 'SSP585']:
        irct_vals = IRCT_RESULTS[model_key][scenario_name]['IRCT']
        
        title = f"IRCT — {model_key} — {scenario_name} (2090-2100)"
        
        save_filename = f"IRCT_spatial_{model_key}_{scenario_name}_{timestamp}.png"
        save_path = os.path.join(PLOTS_DIR, save_filename)
        
        plot_irct_spatial(
            lat_vals=coords_df['lat'].values,
            lon_vals=coords_df['lon'].values,
            irct_vals=irct_vals,
            title=title,
            cmap='RdYlGn',
            alpha=0.8,
            n_neighbors=15,
            figsize=(14, 10),
            save_path=save_path
        )

print()
print("="*80)
print("✓ VISUALIZACIÓN COMPLETADA")
print("="*80)

# %% [markdown]
# ## 10. Validación del IRCT
# 
# Validación simplificada y sólida del índice usando 4 pruebas esenciales:
# 
# 1. **Validación por escenario**: Coherencia lógica del índice entre SSP245, SSP370, SSP585
# 2. **Estabilidad espacial**: Autocorrelación territorial (Moran's I)
# 3. **Estabilidad estructural**: Relación entre IRCT y retención de cluster
# 4. **Robustez del índice**: Sensibilidad a perturbaciones de pesos (±20%)

# %% [markdown]
# ### 10.1 Validación por escenario (coherencia lógica)
# 
# **Objetivo**: Verificar que el IRCT disminuye cuando aumenta el estrés climático.
# 
# **Hipótesis**: IRCT(SSP245) > IRCT(SSP370) > IRCT(SSP585)
# 
# **Métrica**: Mediana del IRCT por escenario + test estadístico (Wilcoxon/KS-test)

# %%
def _paired(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    n = min(len(a), len(b))
    return a[:n], b[:n]


# %%
from scipy.stats import wilcoxon, ks_2samp

print("VALIDACIÓN 1: COHERENCIA POR ESCENARIO")
print("="*80)
print()

validation_results = {}

for model_key in MODEL_ORDER:
    print(f"\n{model_key}")
    print("-"*60)
    
    # Extraer IRCT de cada escenario
    irct_245 = IRCT_RESULTS[model_key]['SSP245']['IRCT']
    irct_370 = IRCT_RESULTS[model_key]['SSP370']['IRCT']
    irct_585 = IRCT_RESULTS[model_key]['SSP585']['IRCT']
    
    median_245 = np.median(irct_245[np.isfinite(irct_245)])
    median_370 = np.median(irct_370[np.isfinite(irct_370)])
    median_585 = np.median(irct_585[np.isfinite(irct_585)])
    
    print(f"\nMedianas del IRCT:")
    print(f"  SSP245: {median_245:.4f}")
    print(f"  SSP370: {median_370:.4f}")
    print(f"  SSP585: {median_585:.4f}")
    
    # 2. Verificar orden esperado
    order_check = (median_245 > median_370) and (median_370 > median_585)
    print(f"\nOrden esperado (245 > 370 > 585): {'✓ CORRECTO' if order_check else '✗ INCORRECTO'}")
    
    # 3. Tests estadísticos (Wilcoxon signed-rank)
    x_245_370, y_245_370 = _paired(irct_245, irct_370)  # => emparejar y filtrar
    x_370_585, y_370_585 = _paired(irct_370, irct_585)  # => idem
    x_245_585, y_245_585 = _paired(irct_245, irct_585)  # => idem
    stat_245_370, p_245_370 = wilcoxon(x_245_370, y_245_370, alternative='greater')
    stat_370_585, p_370_585 = wilcoxon(x_370_585, y_370_585, alternative='greater')
    stat_245_585, p_245_585 = wilcoxon(x_245_585, y_245_585, alternative='greater')

    
    print(f"\nTests de Wilcoxon (H1: IRCT mayor en escenario más benigno):")
    print(f"  SSP245 > SSP370: p-value = {p_245_370:.4e} {'✓' if p_245_370 < 0.05 else '✗'}")
    print(f"  SSP370 > SSP585: p-value = {p_370_585:.4e} {'✓' if p_370_585 < 0.05 else '✗'}")
    print(f"  SSP245 > SSP585: p-value = {p_245_585:.4e} {'✓' if p_245_585 < 0.05 else '✗'}")
    
    ks_stat_245_370, ks_p_245_370 = ks_2samp(x_245_370, y_245_370, alternative='greater')
    ks_stat_370_585, ks_p_370_585 = ks_2samp(x_370_585, y_370_585, alternative='greater')
    ks_stat_245_585, ks_p_245_585 = ks_2samp(x_245_585, y_245_585, alternative='greater')
    
    print(f"  SSP245 > SSP370: n={len(x_245_370)} p={p_245_370:.4e} {'✓' if p_245_370 < 0.05 else '✗'}")  # => muestra n usado
    print(f"  SSP370 > SSP585: n={len(x_370_585)} p={p_370_585:.4e} {'✓' if p_370_585 < 0.05 else '✗'}")
    print(f"  SSP245 > SSP585: n={len(x_245_585)} p={p_245_585:.4e} {'✓' if p_245_585 < 0.05 else '✗'}")

    # Guardar resultados
    validation_results[model_key] = {
        'scenario_validation': {
            'median_245': median_245,
            'median_370': median_370,
            'median_585': median_585,
            'order_correct': order_check,
            'wilcoxon_245_370_p': p_245_370,
            'wilcoxon_370_585_p': p_370_585,
            'wilcoxon_245_585_p': p_245_585,
            'ks_245_370_p': ks_p_245_370,
            'ks_370_585_p': ks_p_370_585,
        }
    }

print("\n" + "="*80)
print("✓ VALIDACIÓN 1 COMPLETADA")
print("="*80)

# %% [markdown]
# ### 10.2 Estabilidad espacial (autocorrelación territorial)
# 
# **Objetivo**: Verificar que el IRCT presenta patrones geográficos coherentes (no es ruido aleatorio).
# 
# **Hipótesis**: Moran's I > 0 y significativo (p < 0.05)
# 
# **Métrica**: Índice de Moran I con matriz de pesos espaciales basada en distancia

# %%
from math import radians, sin, cos, sqrt, asin
from scipy.stats import norm
import numpy as np

def _haversine_matrix(coords):
    # => Nueva: matriz de distancias haversine (km) para lat/lon en grados
    R = 6371.0
    n = len(coords)
    D = np.zeros((n, n), dtype=float)
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    for i in range(n):
        dlat = lat - lat[i]
        dlon = lon - lon[i]
        a = np.sin(dlat/2)**2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon/2)**2
        D[i] = 2 * R * np.arcsin(np.sqrt(a))
    return D

def compute_morans_i(values, coords, k_neighbors=8, n_perm=999, use_normal_approx=True):
    """
    Moran's I con:
      - kNN binario simétrico
      - => Row-standardization (W_ij / sum_j W_ij)
      - => p-value por permutaciones (empírico) y opcional Z normal
    Retorna: I, expected_I, z_score, p_perm, p_norm
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 4:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # => Haversine en vez de euclídea
    dist_matrix = _haversine_matrix(coords)

    # kNN binario
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        nn = np.argsort(dist_matrix[i])[1:k_neighbors+1]
        W[i, nn] = 1.0

    # Simetrizar
    W = ((W + W.T) > 0).astype(float)

    # => Row-standardize (evita sesgos por grado distinto)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums

    # Moran's I
    x_mean = x.mean()
    xc = x - x_mean
    denom = np.sum(xc**2)
    if denom == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    W_sum = W.sum()
    num = np.sum(W * np.outer(xc, xc))
    I = (n / W_sum) * (num / denom)

    # Esperado bajo H0 (aleatorio)
    expected_I = -1.0 / (n - 1)

    # => Permutaciones (p-valor empírico, unilateral positiva)
    #    p_perm = ( #{I_perm >= I_obs} + 1 ) / (n_perm + 1)
    count_ge = 1
    for _ in range(n_perm):
        xp = np.random.permutation(xc)
        num_p = np.sum(W * np.outer(xp, xp))
        I_p = (n / W_sum) * (num_p / denom)
        if I_p >= I:
            count_ge += 1
    p_perm = count_ge / (n_perm + 1)

    # => (Opcional) aproximación normal para reporte
    if use_normal_approx:
        # Varianza aproximada (normalidad); puede ser inestable con W estandarizada
        # Se reporta solo como referencia
        # S1, S2 siguiendo notación clásica con W simétrica:
        S1 = 0.5 * np.sum((W + W.T)**2)
        S2 = np.sum((W.sum(axis=0) + W.sum(axis=1))**2)
        b2 = (n * np.sum(xc**4)) / (denom**2)
        var_num = n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * (W_sum**2))
        var_num -= b2 * ((n**2 - n) * S1 - 2*n * S2 + 6 * (W_sum**2))
        var_den = (n - 1) * (n - 2) * (n - 3) * (W_sum**2)
        var_I = var_num / var_den - expected_I**2 if var_den != 0 else np.nan
        if np.isfinite(var_I) and var_I > 0:
            z_score = (I - expected_I) / np.sqrt(var_I)
            p_norm = 2 * (1 - norm.cdf(abs(z_score)))
        else:
            z_score, p_norm = np.nan, np.nan
    else:
        z_score, p_norm = np.nan, np.nan

    return I, expected_I, z_score, p_perm, p_norm

print("VALIDACIÓN 2: ESTABILIDAD ESPACIAL (MORAN'S I)")
print("="*80)
print()

coords_array = coords_df[['lat', 'lon']].values

for model_key in MODEL_ORDER:
    print(f"\n{model_key}")
    print("-"*60)
    for scenario_name in ['SSP245', 'SSP370', 'SSP585']:
        irct_vals = IRCT_RESULTS[model_key][scenario_name]['IRCT']

        I, expected_I, z_norm, p_perm, p_norm = compute_morans_i(
            irct_vals, coords_array, k_neighbors=8, n_perm=999, use_normal_approx=True
        )

        print(f"\n  {scenario_name}:")
        print(f"    Moran's I:        {I:.4f}")
        print(f"    Esperado (H0):    {expected_I:.4f}")
        print(f"    Z (aprox. normal): {z_norm if np.isfinite(z_norm) else np.nan:.4f}")
        print(f"    p (perm, one-sided +): {p_perm:.4e}")
        print(f"    p (normal, two-sided): {p_norm if np.isfinite(p_norm) else np.nan:.4e}")

        is_significant = (p_perm < 0.05)  # => criterio principal con permutaciones
        is_positive = I > 0
        status = "✓ VÁLIDO" if (is_significant and is_positive) else "✗ NO VÁLIDO"
        print(f"    Autocorrelación espacial positiva y significativa: {status}")

        if 'spatial_validation' not in validation_results[model_key]:
            validation_results[model_key]['spatial_validation'] = {}

        validation_results[model_key]['spatial_validation'][scenario_name] = {
            'morans_I': I,
            'expected_I': expected_I,
            'z_score_normal': z_norm,
            'p_value_perm_one_sided_pos': p_perm,
            'p_value_normal_two_sided': p_norm,
            'is_significant_perm': is_significant,
            'is_positive': is_positive
        }

print("\n" + "="*80)
print("✓ VALIDACIÓN 2 COMPLETADA")
print("="*80)



# %% [markdown]
# ### 10.3 Estabilidad estructural (IRCT vs. retención de cluster)
# 
# **Objetivo**: Verificar que píxeles con IRCT alto mantienen su identidad climática-energética (permanecen en su cluster BASE).
# 
# **Hipótesis**: Correlación positiva entre IRCT y tasa de retención del cluster
# 
# **Métrica**: Correlación de Spearman entre IRCT y % de retención por cluster

# %%
from scipy.stats import spearmanr, mannwhitneyu
# => NUEVO:
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

def cliffs_delta(x, y):
    # => NUEVO: tamaño de efecto para dos muestras (ret vs mig)
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    # Ordenar para O((nx+ny) log(nx+ny)) si quieres optimizar; aquí simple:
    count = 0
    for xi in x:
        count += np.sum(xi > y) - np.sum(xi < y)
    return count / (nx * ny)

def holm_bonferroni(pvals):
    # => NUEVO: ajuste Holm-Bonferroni; retorna booleanos (rechazo) y pvals ajustados
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    for k, idx in enumerate(order):
        adj[idx] = max(pvals[order[:k+1]]) * (m - k)
    adj = np.minimum.accumulate(adj[np.argsort(order)[::-1]])[::-1]  # monotonicidad
    # clamp
    adj = np.clip(adj, 0, 1)
    return adj

print("VALIDACIÓN 3: ESTABILIDAD ESTRUCTURAL (IRCT vs RETENCIÓN)")
print("="*80)
print()

validation_results.setdefault('structural_validation', {})  # => NUEVO

for model_key in MODEL_ORDER:
    print(f"\n{model_key}")
    print("-"*60)
    
    # => NUEVO: acumuladores para ajuste múltiple
    pvals_spearman = []
    pvals_u = []
    scen_keys = []
    scen_store = {}

    for scenario_name in ['SSP245', 'SSP370', 'SSP585']:
        print(f"\n  {scenario_name}:")
        df_cluster = IRCT_CLUSTER_RESULTS[model_key][scenario_name]

        # => GUARDAS: chequear columnas y número de clústeres
        if ('IRCT_median' not in df_cluster.columns) or ('retention_pct' not in df_cluster.columns):
            print("    ✗ Faltan columnas IRCT_median / retention_pct")
            continue
        if len(df_cluster) < 3 or df_cluster['IRCT_median'].nunique() < 2 or df_cluster['retention_pct'].nunique() < 2:
            print("    ✗ Pocos clústeres o variabilidad insuficiente para correlación")
            rho, p_value = np.nan, np.nan
        else:
            # => Spearman normal
            rho, p_value = spearmanr(df_cluster['IRCT_median'].values, df_cluster['retention_pct'].values)
            # => Spearman ponderado (aprox): rankea y correla con pesos por tamaño
            if 'count' in df_cluster.columns and df_cluster['count'].sum() > 0:
                w = df_cluster['count'].values.astype(float)
                # centrado ponderado en ranks
                r1 = df_cluster['IRCT_median'].rank().values
                r2 = df_cluster['retention_pct'].rank().values
                r1c = r1 - np.average(r1, weights=w)
                r2c = r2 - np.average(r2, weights=w)
                num = np.sum(w * r1c * r2c)
                den = np.sqrt(np.sum(w * r1c**2) * np.sum(w * r2c**2))
                rho_w = num / den if den > 0 else np.nan
                print(f"    Spearman ponderado (aprox): ρ_w = {rho_w:.4f}")
            else:
                rho_w = np.nan

        print(f"    Spearman (IRCT vs Retención): ρ = {rho:.4f} | p = {p_value:.4e}")
        is_sig = (p_value < 0.05) if np.isfinite(p_value) else False
        is_pos = (rho > 0) if np.isfinite(rho) else False
        print(f"      Correlación positiva y significativa: {'✓' if (is_sig and is_pos) else '✗'}")

        # A nivel píxel: retenidos vs migrados
        irct_vals = IRCT_RESULTS[model_key][scenario_name]['IRCT']
        if scenario_name == 'SSP245':
            labels_b = CLUSTERING_RESULTS[model_key]['labels_B245']
            labels_t = CLUSTERING_RESULTS[model_key]['labels_T245']
        elif scenario_name == 'SSP370':
            labels_b = CLUSTERING_RESULTS[model_key]['labels_B370']
            labels_t = CLUSTERING_RESULTS[model_key]['labels_T370']
        else:
            labels_b = CLUSTERING_RESULTS[model_key]['labels_B585']
            labels_t = CLUSTERING_RESULTS[model_key]['labels_T585']

        # => NUEVO: estabilidad de partición
        ari = adjusted_rand_score(labels_b, labels_t)
        nmi = normalized_mutual_info_score(labels_b, labels_t)
        print(f"    Estabilidad de partición: ARI = {ari:.4f} | NMI = {nmi:.4f}")

        retained_mask = (labels_b == labels_t)
        irct_retained = irct_vals[retained_mask]
        irct_migrated = irct_vals[~retained_mask]

        print(f"\n    IRCT promedio por tipo de píxel:")
        print(f"      Retenidos:  {irct_retained.mean():.4f} ({retained_mask.sum()} px, {100*retained_mask.mean():.1f}%)")
        print(f"      Migrados:   {irct_migrated.mean():.4f} ({(~retained_mask).sum()} px, {100*(~retained_mask).mean():.1f}%)")
        print(f"      Diferencia: {(irct_retained.mean() - irct_migrated.mean()):.4f}")

        # Mann-Whitney (retenidos > migrados)
        if (len(irct_retained) > 0) and (len(irct_migrated) > 0):
            stat_u, p_diff = mannwhitneyu(irct_retained, irct_migrated, alternative='greater')
            delta = cliffs_delta(irct_retained, irct_migrated)  # => NUEVO
            print(f"      Mann-Whitney U: p = {p_diff:.4e} {'✓' if p_diff < 0.05 else '✗'} | Cliff’s Δ = {delta:.3f}")
        else:
            p_diff, delta = np.nan, np.nan
            print("      (Insuficientes datos para U-test / Δ)")

        # => acumular para ajuste múltiple
        scen_keys.append((model_key, scenario_name))
        pvals_spearman.append(p_value if np.isfinite(p_value) else 1.0)
        pvals_u.append(p_diff if np.isfinite(p_diff) else 1.0)

        scen_store[scenario_name] = {
            'spearman_rho': rho, 'spearman_p': p_value, 'spearman_weighted_rho': rho_w,
            'ari': ari, 'nmi': nmi,
            'irct_retained_mean': float(np.nanmean(irct_retained)) if irct_retained.size else np.nan,
            'irct_migrated_mean': float(np.nanmean(irct_migrated)) if irct_migrated.size else np.nan,
            'retention_rate': float(retained_mask.mean()),
            'mannwhitneyu_p': p_diff, 'cliffs_delta': delta
        }

    # => AJUSTE MULTIPLE por modelo (3 escenarios)
    if pvals_spearman:
        sp_adj = holm_bonferroni(np.array(pvals_spearman))
        u_adj  = holm_bonferroni(np.array(pvals_u))
        print("\n  Ajuste Holm–Bonferroni (por modelo):")
        for i, scenario_name in enumerate(['SSP245','SSP370','SSP585']):
            if scenario_name in scen_store:
                print(f"    {scenario_name}: p_spear_adj={sp_adj[i]:.4e} | p_U_adj={u_adj[i]:.4e}")
                scen_store[scenario_name]['spearman_p_holm'] = sp_adj[i]
                scen_store[scenario_name]['mannwhitneyu_p_holm'] = u_adj[i]

    validation_results['structural_validation'][model_key] = scen_store

print("\n" + "="*80)
print("✓ VALIDACIÓN 3 COMPLETADA")
print("="*80)


# %% [markdown]
# ### 10.4 Robustez del índice (sensibilidad a perturbaciones de pesos)
# 
# **Objetivo**: Verificar que el IRCT es estable y no depende de parámetros arbitrarios.
# 
# **Hipótesis**: Perturbaciones de ±20% en los pesos no alteran significativamente el ranking de resiliencia.
# 
# **Métrica**: Correlación de Spearman entre IRCT original y IRCT perturbado (esperado: ρ > 0.9)

# %%
# => NUEVO: muestreador de pesos alrededor del vector base usando Dirichlet
def sample_dirichlet_around(weights_original, perturbation=0.2, rng=None):
    """
    Centra la Dirichlet en weights_original y controla la dispersión con 'perturbation'.
    perturbation pequeña => alta concentración (poca variación).
    """
    if rng is None:
        rng = np.random.default_rng()
    keys = ['w_a', 'w_d', 'w_c', 'w_e', 'w_h']
    w0 = np.array([weights_original[k] for k in keys], dtype=float)
    # Concentración ~ 1/perturbation^2 (heurística suave)
    conc = max(1e-3, 1.0 / (perturbation ** 2))
    alpha = np.clip(w0 * conc, 1e-3, None)
    w = rng.dirichlet(alpha)
    return {k: v for k, v in zip(keys, w)}

# => NUEVO: precomputar componentes para no recalcular el forward del modelo por perturbación
def precompute_irct_components(
    model,
    X_base_orig_subset,
    X_base_norm_subset,
    X_future_orig,
    X_future_norm,
    z_base_scaled_subset,
    z_future_scaled,
    centroids_base,
    labels_base_subset,
    h2_base,
    h2_future,
    device='cpu'
):
    A, _re = compute_reconstruction_anomaly(model, X_future_orig, X_future_norm, device)
    S_D, _ld = compute_latent_displacement(z_future_scaled, centroids_base, labels_base_subset)
    S_C, _ad = compute_cluster_stability_softmax(z_future_scaled, centroids_base, labels_base_subset)
    S_E, _er = compute_cluster_expansion(z_base_scaled_subset, z_future_scaled, centroids_base, labels_base_subset)
    if (h2_base is not None) and (h2_future is not None):
        S_H2, _hr = compute_h2_stability(h2_base, h2_future)
    else:
        S_H2 = np.ones_like(A)
    return {'A': A, 'S_D': S_D, 'S_C': S_C, 'S_E': S_E, 'S_H2': S_H2}

# => NUEVO: agrega Kendall τ y Jaccard Top-k
from scipy.stats import spearmanr, kendalltau

def topk_jaccard(a, b, k_ratio=0.10):
    k = max(1, int(len(a) * k_ratio))
    idx_a = np.argpartition(-a, k-1)[:k]
    idx_b = np.argpartition(-b, k-1)[:k]
    set_a, set_b = set(idx_a.tolist()), set(idx_b.tolist())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else np.nan


print("VALIDACIÓN 4: ROBUSTEZ DEL ÍNDICE (SENSIBILIDAD A PESOS)")
print("="*80)
print()

# => CAMBIO: usa RNG nuevo y controla N perturbaciones
N_PERTURBATIONS = 50
PERTURBATION_MAGNITUDE = 0.20  # ±20%
rng_master = np.random.default_rng(SEED)

for model_key in MODEL_ORDER:
    print(f"\n{model_key}")
    print("-"*60)
    
    model = MODELS[model_key]
    clustering = CLUSTERING_RESULTS[model_key]
    centroids_base = clustering["centroids"]
    
    for scenario_name in ['SSP245', 'SSP370', 'SSP585']:
        print(f"\n  {scenario_name}:")
        
        irct_original = IRCT_RESULTS[model_key][scenario_name]['IRCT']
        weights_original = IRCT_RESULTS[model_key][scenario_name]['weights']
        
        print(f"\n    Pesos originales:")
        for k, v in weights_original.items():
            print(f"      {k}: {v:.3f}")
        
        # === preparar datos escenario ===
        if scenario_name == 'SSP245':
            X_base_orig_subset = X_BASE[:N_PER_SCENARIO]
            X_future_orig = X245_orig
            X_future_norm = X245_norm if X245_norm is not None else X245_orig
            z_base_scaled_subset = clustering['z_B245_scaled']
            z_future_scaled = clustering['z_T245_scaled']
            labels_base_subset = clustering['labels_B245']
        elif scenario_name == 'SSP370':
            X_base_orig_subset = X_BASE[N_PER_SCENARIO:2*N_PER_SCENARIO]
            X_future_orig = X370_orig
            X_future_norm = X370_norm if X370_norm is not None else X370_orig
            z_base_scaled_subset = clustering['z_B370_scaled']
            z_future_scaled = clustering['z_T370_scaled']
            labels_base_subset = clustering['labels_B370']
        else:
            X_base_orig_subset = X_BASE[2*N_PER_SCENARIO:]
            X_future_orig = X585_orig
            X_future_norm = X585_norm if X585_norm is not None else X585_orig
            z_base_scaled_subset = clustering['z_B585_scaled']
            z_future_scaled = clustering['z_T585_scaled']
            labels_base_subset = clustering['labels_B585']
        
        X_base_norm_subset = X_base_orig_subset
        
        if h2_available:
            h2_base = X_base_orig_subset[:, H2_IDX_FUTURE]
            h2_future = X_future_orig[:, H2_IDX_FUTURE]
        else:
            h2_base = None
            h2_future = None
        
        # => NUEVO: precomputar componentes
        comps = precompute_irct_components(
            model,
            X_base_orig_subset,
            X_base_norm_subset,
            X_future_orig,
            X_future_norm,
            z_base_scaled_subset,
            z_future_scaled,
            centroids_base,
            labels_base_subset,
            h2_base,
            h2_future,
            device='cpu'
        )
        
        # === perturbaciones ===
        rhos, taus, jaccs = [], [], []
        print(f"\n    Calculando {N_PERTURBATIONS} perturbaciones...")
        
        for i in range(N_PERTURBATIONS):
            rng = np.random.default_rng(rng_master.integers(0, 2**32-1))
            weights_perturbed = sample_dirichlet_around(
                weights_original,
                perturbation=PERTURBATION_MAGNITUDE,
                rng=rng
            )
            
            # => combinar por media geométrica ponderada sin re-ejecutar modelo
            IRCT_p = (
                (comps['A']   ** weights_perturbed['w_a']) *
                (comps['S_D'] ** weights_perturbed['w_d']) *
                (comps['S_C'] ** weights_perturbed['w_c']) *
                (comps['S_E'] ** weights_perturbed['w_e']) *
                (comps['S_H2']** weights_perturbed['w_h'])
            )
            
            # robustez por ranking y top-k
            rho, _ = spearmanr(irct_original, IRCT_p)
            tau, _ = kendalltau(irct_original, IRCT_p)
            jac = topk_jaccard(irct_original, IRCT_p, k_ratio=0.10)
            rhos.append(rho); taus.append(tau); jaccs.append(jac)
        
        rhos = np.array(rhos); taus = np.array(taus); jaccs = np.array(jaccs)
        
        print(f"\n    Correlaciones Spearman (orig vs pert): mean={np.nanmean(rhos):.4f} | std={np.nanstd(rhos):.4f} | min={np.nanmin(rhos):.4f} | max={np.nanmax(rhos):.4f}")
        print(f"    Kendall τ (orig vs pert):              mean={np.nanmean(taus):.4f} | std={np.nanstd(taus):.4f}")
        print(f"    Jaccard Top-10% (orig vs pert):        mean={np.nanmean(jaccs):.4f} | std={np.nanstd(jaccs):.4f}")
        
        # => criterio de robustez informativo (puedes ajustar umbrales)
        is_robust = (np.nanmean(rhos) > 0.90) and (np.nanmean(taus) > 0.80) and (np.nanmean(jaccs) > 0.70)
        print(f"      Índice robusto (ρ>0.90, τ>0.80, Jaccard>0.70): {'✓ ROBUSTO' if is_robust else '✗ NO ROBUSTO'}")
        
        # guardar
        validation_results.setdefault(model_key, {})
        validation_results[model_key].setdefault('robustness_validation', {})
        validation_results[model_key]['robustness_validation'][scenario_name] = {
            'n_perturbations': N_PERTURBATIONS,
            'perturbation_magnitude': PERTURBATION_MAGNITUDE,
            'spearman_mean': float(np.nanmean(rhos)),
            'spearman_std': float(np.nanstd(rhos)),
            'spearman_min': float(np.nanmin(rhos)),
            'spearman_max': float(np.nanmax(rhos)),
            'kendall_tau_mean': float(np.nanmean(taus)),
            'kendall_tau_std': float(np.nanstd(taus)),
            'top10_jaccard_mean': float(np.nanmean(jaccs)),
            'top10_jaccard_std': float(np.nanstd(jaccs)),
            'is_robust': bool(is_robust),
            'weights_baseline': weights_original
        }

print("\n" + "="*80)
print("✓ VALIDACIÓN 4 COMPLETADA")
print("="*80)


# %% [markdown]
# ### 10.5 Resumen consolidado de validaciones
# 
# Síntesis de los resultados de las 4 validaciones para todos los modelos y escenarios

# %%
print("RESUMEN CONSOLIDADO DE VALIDACIONES")
print("="*80)
print()

# Crear DataFrame resumen
summary_data = []

for model_key in MODEL_ORDER:
    for scenario_name in ['SSP245', 'SSP370', 'SSP585']:
        
        # Validación 1: Por escenario
        scenario_val = validation_results[model_key].get('scenario_validation', {})
        
        # Validación 2: Espacial
        spatial_val = validation_results[model_key].get('spatial_validation', {}).get(scenario_name, {})
        
        # Validación 3: Estructural
        structural_val = validation_results[model_key].get('structural_validation', {}).get(scenario_name, {})
        
        # Validación 4: Robustez
        robustness_val = validation_results[model_key].get('robustness_validation', {}).get(scenario_name, {})
        
        row = {
            'Modelo': model_key,
            'Escenario': scenario_name,
            
            # Val 1: Coherencia por escenario (solo mostrar para SSP245 para evitar redundancia)
            'IRCT_median': scenario_val.get(f'median_{scenario_name[-3:].lower()}', np.nan),
            'Orden_correcto': '✓' if scenario_val.get('order_correct', False) else '✗',
            
            # Val 2: Espacial
            'Morans_I': spatial_val.get('morans_I', np.nan),
            'Morans_p': spatial_val.get('p_value', np.nan),
            'Espacial_OK': '✓' if spatial_val.get('is_significant', False) and spatial_val.get('is_positive', False) else '✗',
            
            # Val 3: Estructural
            'Spearman_ρ': structural_val.get('spearman_rho', np.nan),
            'Spearman_p': structural_val.get('spearman_p', np.nan),
            'Estructural_OK': '✓' if structural_val.get('is_significant', False) and structural_val.get('is_positive', False) else '✗',
            'Retención_%': structural_val.get('retention_rate', np.nan) * 100,
            
            # Val 4: Robustez
            'Robustez_ρ_mean': robustness_val.get('mean_spearman_rho', np.nan),
            'Robustez_OK': '✓' if robustness_val.get('is_robust', False) else '✗',
        }
        
        summary_data.append(row)

df_summary = pd.DataFrame(summary_data)

# Formatear para display
print("TABLA RESUMEN: VALIDACIONES DEL IRCT")
print()
print(df_summary.to_string(index=False))
print()

# Calcular tasa de éxito por validación
print("\nTASA DE ÉXITO POR VALIDACIÓN:")
print("-"*60)

n_total = len(df_summary)

# Validación 1: Orden correcto (solo contar una vez por modelo)
n_orden_correcto = (df_summary.groupby('Modelo')['Orden_correcto'].first() == '✓').sum()
n_modelos = df_summary['Modelo'].nunique()
print(f"  1. Coherencia por escenario:  {n_orden_correcto}/{n_modelos} modelos ({100*n_orden_correcto/n_modelos:.1f}%)")

# Validación 2: Espacial
n_espacial_ok = (df_summary['Espacial_OK'] == '✓').sum()
print(f"  2. Estabilidad espacial:      {n_espacial_ok}/{n_total} casos ({100*n_espacial_ok/n_total:.1f}%)")

# Validación 3: Estructural
n_estructural_ok = (df_summary['Estructural_OK'] == '✓').sum()
print(f"  3. Estabilidad estructural:   {n_estructural_ok}/{n_total} casos ({100*n_estructural_ok/n_total:.1f}%)")

# Validación 4: Robustez
n_robustez_ok = (df_summary['Robustez_OK'] == '✓').sum()
print(f"  4. Robustez del índice:       {n_robustez_ok}/{n_total} casos ({100*n_robustez_ok/n_total:.1f}%)")

print()

# Tasa de éxito global
success_columns = ['Espacial_OK', 'Estructural_OK', 'Robustez_OK']
df_summary['N_validaciones_OK'] = (df_summary[success_columns] == '✓').sum(axis=1)
mean_success = df_summary['N_validaciones_OK'].mean()

print(f"TASA DE ÉXITO GLOBAL: {mean_success:.1f}/3 validaciones por caso ({100*mean_success/3:.1f}%)")
print()

# Identificar mejor modelo
print("\nMEJOR MODELO POR ESCENARIO:")
print("-"*60)
for scenario in ['SSP245', 'SSP370', 'SSP585']:
    df_scenario = df_summary[df_summary['Escenario'] == scenario]
    best_model = df_scenario.loc[df_scenario['N_validaciones_OK'].idxmax(), 'Modelo']
    best_score = df_scenario['N_validaciones_OK'].max()
    print(f"  {scenario}: {best_model} ({best_score}/3 validaciones)")

print()
print("="*80)
print("✓ RESUMEN DE VALIDACIONES COMPLETADO")
print("="*80)

# %% [markdown]
# ### 10.6 Comparación visual: IRCT vs. Clusters
# 
# Visualización lado a lado del índice de resiliencia y la estructura de clusters para cada modelo y escenario

# %%
def plot_irct_vs_clusters_comparison(
    lat_vals, lon_vals, 
    irct_vals, 
    labels_base, labels_future,
    model_key, scenario_name,
    n_neighbors=15, 
    figsize=(20, 8), 
    save_path=None
):
    """
    Visualiza lado a lado:
    - Izquierda: Mapa del IRCT interpolado
    - Derecha: Estructura de clusters (BASE vs FUTURO)
    
    Args:
        lat_vals, lon_vals: coordenadas
        irct_vals: valores del IRCT
        labels_base: clusters BASE
        labels_future: clusters FUTURO
        model_key: nombre del modelo
        scenario_name: nombre del escenario
        n_neighbors: vecinos para interpolación
        figsize: tamaño de la figura
        save_path: ruta para guardar
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Transformar a Web Mercator
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xs, ys = transformer.transform(lon_vals, lat_vals)
        
        # Crear grid para interpolación
        grid_res = 100
        grid_x = np.linspace(xs.min(), xs.max(), grid_res)
        grid_y = np.linspace(ys.min(), ys.max(), grid_res)
        GX, GY = np.meshgrid(grid_x, grid_y)
        extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
        
        # === PANEL IZQUIERDO: IRCT ===
        valid_mask = ~(np.isnan(irct_vals) | np.isinf(irct_vals))
        coords = np.column_stack([xs[valid_mask], ys[valid_mask]])
        vals = irct_vals[valid_mask]
        
        # Interpolación KNN para IRCT
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
        knn.fit(coords, vals)
        
        grid_points = np.column_stack([GX.ravel(), GY.ravel()])
        pred_irct = knn.predict(grid_points).reshape(GX.shape)
        
        # Basemap izquierdo
        ax1.set_xlim(extent[0], extent[1])
        ax1.set_ylim(extent[2], extent[3])
        
        try:
            import contextily as ctx
            ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron, 
                          crs="EPSG:3857", alpha=1.0, attribution_size=6)
        except:
            pass
        
        # Overlay IRCT
        im1 = ax1.imshow(pred_irct, extent=extent, origin="lower", 
                        cmap='RdYlGn', alpha=0.8, vmin=0, vmax=1, zorder=3)
        
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("IRCT\n(0=vulnerable, 1=resiliente)", fontsize=11)
        
        ax1.set_title(f"Índice de Resiliencia (IRCT)\n{model_key} — {scenario_name}", 
                     fontsize=13, fontweight='bold', pad=10)
        ax1.set_axis_off()
        
        # === PANEL DERECHO: CLUSTERS ===
        # Interpolación de clusters (usar moda de vecinos)
        from scipy.stats import mode
        
        # Interpolación usando clasificación para mantener clusters discretos
        coords_all = np.column_stack([xs, ys])
        labels_combined = labels_future.copy().astype(int)
        
        # Usar KNeighborsClassifier (mismo método que celda 13)
        n_neighbors_cluster = max(1, min(len(labels_combined), 15))
        knn_cluster = KNeighborsClassifier(n_neighbors=n_neighbors_cluster, weights="distance")
        knn_cluster.fit(coords_all, labels_combined)
        
        pred_clusters = knn_cluster.predict(grid_points).reshape(GX.shape)
        
        # Basemap derecho
        ax2.set_xlim(extent[0], extent[1])
        ax2.set_ylim(extent[2], extent[3])
        
        try:
            ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron, 
                          crs="EPSG:3857", alpha=1.0, attribution_size=6)
        except:
            pass
        
        # Overlay de clusters con colormap categórico
        # Colormap consistente con celda 13
        color_palette = plt.get_cmap('tab20', 20)
        cluster_colors = []
        for cluster_id in range(K_CLUSTERS):
            color_idx = int(cluster_id) % 20
            cluster_colors.append(color_palette(color_idx))
        from matplotlib.colors import ListedColormap
        cmap_clusters = ListedColormap(cluster_colors)
        im2 = ax2.imshow(pred_clusters, extent=extent, origin="lower", 
                        cmap=cmap_clusters, alpha=0.7, vmin=0, vmax=K_CLUSTERS-1, zorder=3)
        
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, 
                            ticks=range(K_CLUSTERS))
        cbar2.set_label("Cluster ID\n(estructura territorial)", fontsize=11)
        
        # Calcular retención por cluster
        retention_rates = []
        for cluster_id in range(K_CLUSTERS):
            mask = labels_base == cluster_id
            if mask.sum() > 0:
                retention = (labels_future[mask] == cluster_id).sum() / mask.sum() * 100
                retention_rates.append(retention)
            else:
                retention_rates.append(0)
        
        avg_retention = np.mean(retention_rates)
        
        ax2.set_title(f"Estructura de Clusters (Futuro)\n{model_key} — {scenario_name}\nRetención promedio: {avg_retention:.1f}%", 
                     fontsize=13, fontweight='bold', pad=10)
        ax2.set_axis_off()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  → Guardado: {save_path}")
        
        plt.show()
        
    except Exception as err:
        print(f"Error en plot_irct_vs_clusters_comparison: {err}")
        import traceback
        traceback.print_exc()
        plt.close(fig)


print("VISUALIZACIÓN COMPARATIVA: IRCT vs CLUSTERS")
print("="*80)
print()

for model_key in MODEL_ORDER:
    print(f"\n{model_key}")
    print("-"*60)
    
    for scenario_name in ['SSP245', 'SSP370', 'SSP585']:
        print(f"\n  {scenario_name}:")
        
        # Obtener datos
        irct_vals = IRCT_RESULTS[model_key][scenario_name]['IRCT']
        
        # Determinar labels según escenario
        if scenario_name == 'SSP245':
            labels_base = CLUSTERING_RESULTS[model_key]['labels_B245']
            labels_future = CLUSTERING_RESULTS[model_key]['labels_T245']
        elif scenario_name == 'SSP370':
            labels_base = CLUSTERING_RESULTS[model_key]['labels_B370']
            labels_future = CLUSTERING_RESULTS[model_key]['labels_T370']
        else:
            labels_base = CLUSTERING_RESULTS[model_key]['labels_B585']
            labels_future = CLUSTERING_RESULTS[model_key]['labels_T585']
        
        # Generar gráfico comparativo
        save_filename = f"IRCT_vs_clusters_{model_key}_{scenario_name}_{timestamp}.png"
        save_path = os.path.join(PLOTS_DIR, save_filename)
        
        plot_irct_vs_clusters_comparison(
            lat_vals=coords_df['lat'].values,
            lon_vals=coords_df['lon'].values,
            irct_vals=irct_vals,
            labels_base=labels_base,
            labels_future=labels_future,
            model_key=model_key,
            scenario_name=scenario_name,
            n_neighbors=15,
            figsize=(20, 8),
            save_path=save_path
        )

print()
print("="*80)
print("✓ VISUALIZACIÓN COMPARATIVA COMPLETADA")
print("="*80)

# %% [markdown]
# ## Visualización detallada: IRCT y sus componentes
# 
# Descomposición espacial del IRCT mostrando cada componente individual para entender cómo se construye el índice de resiliencia

# %%
def plot_irct_decomposition(
    lat_vals, lon_vals,
    irct_result,
    labels_base, labels_future,
    model_key, scenario_name,
    n_neighbors=15,
    figsize=(24, 14),
    save_path=None
):
    """
    Visualiza el IRCT y sus 5 componentes individuales en un grid 2x3
    
    Grid layout:
    [A: Anomalía] [S_D: Desplazamiento] [S_C: Estabilidad]
    [S_E: Expansión] [S_H2: Energía] [IRCT Final]
    
    Args:
        lat_vals, lon_vals: coordenadas
        irct_result: diccionario con IRCT y componentes (A, S_D, S_C, S_E, S_H2)
        labels_base, labels_future: clusters BASE y FUTURO
        model_key: nombre del modelo
        scenario_name: nombre del escenario
        n_neighbors: vecinos para interpolación
        figsize: tamaño de la figura
        save_path: ruta para guardar
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Transformar a Web Mercator
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xs, ys = transformer.transform(lon_vals, lat_vals)
        
        # Crear grid para interpolación
        grid_res = 100
        grid_x = np.linspace(xs.min(), xs.max(), grid_res)
        grid_y = np.linspace(ys.min(), ys.max(), grid_res)
        GX, GY = np.meshgrid(grid_x, grid_y)
        extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
        grid_points = np.column_stack([GX.ravel(), GY.ravel()])
        
        # Componentes a visualizar
        components = [
            ('A', 'Anomalía Reconstrucción\n(error autoencoder)', 'RdYlGn_r', (0, 1)),
            ('S_D', 'Desplazamiento Latente\n(deriva en espacio latente)', 'RdYlGn_r', (0, 1)),
            ('S_C', 'Estabilidad Cluster\n(retención de cluster)', 'RdYlGn', (0, 1)),
            ('S_E', 'Expansión Cluster\n(compacidad espacial)', 'RdYlGn_r', (0, 1)),
            ('S_H2', 'Estabilidad Energética\n(producción H₂)', 'RdYlGn', (0, 1)),
        ]
        
        # Plotear cada componente
        for idx, (comp_key, title, cmap, vrange) in enumerate(components):
            ax = axes[idx]
            
            # Obtener valores del componente
            comp_vals = irct_result[comp_key]
            
            # Filtrar valores válidos
            valid_mask = ~(np.isnan(comp_vals) | np.isinf(comp_vals))
            coords = np.column_stack([xs[valid_mask], ys[valid_mask]])
            vals = comp_vals[valid_mask]
            
            if len(vals) == 0:
                ax.text(0.5, 0.5, f'No data for {comp_key}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            
            # Interpolación KNN
            knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
            knn.fit(coords, vals)
            pred_vals = knn.predict(grid_points).reshape(GX.shape)
            
            # Basemap
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            
            try:
                import contextily as ctx
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                              crs="EPSG:3857", alpha=1.0, attribution_size=6)
            except:
                pass
            
            # Overlay del componente
            im = ax.imshow(pred_vals, extent=extent, origin="lower",
                          cmap=cmap, alpha=0.8, vmin=vrange[0], vmax=vrange[1], zorder=3)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"{comp_key}\n(0=malo, 1=bueno)", fontsize=9)
            
            # Título con estadísticas
            mean_val = comp_vals.mean()
            std_val = comp_vals.std()
            ax.set_title(f"{title}\nμ={mean_val:.3f}, σ={std_val:.3f}",
                        fontsize=11, fontweight='bold', pad=8)
            ax.set_axis_off()
        
        # Panel 6: IRCT Final
        ax = axes[5]
        irct_vals = irct_result['IRCT']
        valid_mask = ~(np.isnan(irct_vals) | np.isinf(irct_vals))
        coords = np.column_stack([xs[valid_mask], ys[valid_mask]])
        vals = irct_vals[valid_mask]
        
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
        knn.fit(coords, vals)
        pred_irct = knn.predict(grid_points).reshape(GX.shape)
        
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                          crs="EPSG:3857", alpha=1.0, attribution_size=6)
        except:
            pass
        
        im = ax.imshow(pred_irct, extent=extent, origin="lower",
                      cmap='RdYlGn', alpha=0.85, vmin=0, vmax=1, zorder=3)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("IRCT\n(0=vulnerable, 1=resiliente)", fontsize=9)
        
        # Calcular estadísticas
        mean_irct = irct_vals.mean()
        std_irct = irct_vals.std()
        median_irct = np.median(irct_vals)
        
        ax.set_title(f"IRCT FINAL (Agregado)\nμ={mean_irct:.3f}, σ={std_irct:.3f}, med={median_irct:.3f}",
                    fontsize=11, fontweight='bold', pad=8)
        ax.set_axis_off()
        
        # Título general
        fig.suptitle(f"Descomposición del IRCT — {model_key} — {scenario_name} (2090-2100)\n" + 
                    f"IRCT = f(A, S_D, S_C, S_E, S_H2) | Cada componente contribuye al índice final",
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  → Guardado: {save_path}")
        
        plt.show()
        
    except Exception as err:
        print(f"Error en plot_irct_decomposition: {err}")
        import traceback
        traceback.print_exc()
        plt.close(fig)


print("VISUALIZACIÓN DETALLADA: DESCOMPOSICIÓN DEL IRCT")
print("="*80)
print()
print("Mostrando los 5 componentes individuales del IRCT más el índice agregado final")
print()

for model_key in MODEL_ORDER:
    print(f"\n{model_key}")
    print("-"*60)
    
    for scenario_name in ['SSP245', 'SSP370', 'SSP585']:
        print(f"\n  {scenario_name}:")
        
        # Obtener datos
        irct_result = IRCT_RESULTS[model_key][scenario_name]
        
        # Determinar labels según escenario
        if scenario_name == 'SSP245':
            labels_base = CLUSTERING_RESULTS[model_key]['labels_B245']
            labels_future = CLUSTERING_RESULTS[model_key]['labels_T245']
        elif scenario_name == 'SSP370':
            labels_base = CLUSTERING_RESULTS[model_key]['labels_B370']
            labels_future = CLUSTERING_RESULTS[model_key]['labels_T370']
        else:
            labels_base = CLUSTERING_RESULTS[model_key]['labels_B585']
            labels_future = CLUSTERING_RESULTS[model_key]['labels_T585']
        
        # Generar gráfico de descomposición
        save_filename = f"IRCT_decomposition_{model_key}_{scenario_name}_{timestamp}.png"
        save_path = os.path.join(PLOTS_DIR, save_filename)
        
        plot_irct_decomposition(
            lat_vals=coords_df['lat'].values,
            lon_vals=coords_df['lon'].values,
            irct_result=irct_result,
            labels_base=labels_base,
            labels_future=labels_future,
            model_key=model_key,
            scenario_name=scenario_name,
            n_neighbors=15,
            figsize=(24, 14),
            save_path=save_path
        )

print()
print("="*80)
print("✓ VISUALIZACIÓN DE DESCOMPOSICIÓN COMPLETADA")
print("="*80)

# %% [markdown]
# ## Visualización del espacio latente coloreado por IRCT
# 
# Proyección del espacio latente en 2D usando PCA, donde cada punto se colorea según su índice de resiliencia IRCT.

# %%
print("VISUALIZACIÓN DEL ESPACIO LATENTE COLOREADO POR IRCT")
print()

# Obtener IRCTs por escenario
irct_by_scenario = {
    "SSP245": IRCT_RESULTS["VAE"]["SSP245"]['IRCT'],
    "SSP370": IRCT_RESULTS["VAE"]["SSP370"]['IRCT'],
    "SSP585": IRCT_RESULTS["VAE"]["SSP585"]['IRCT']
}

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Calcular rango de valores IRCT
vmin_irct = min([irct_by_scenario[s].min() for s in irct_by_scenario])
vmax_irct = max([irct_by_scenario[s].max() for s in irct_by_scenario])

print(f"Rango de IRCT: [{vmin_irct:.4f}, {vmax_irct:.4f}]")

# Determinar si hay valores negativos y positivos
has_negative = vmin_irct < 0
has_positive = vmax_irct > 0

# Elegir normalización y colormap según el rango
if has_negative and has_positive:
    # Valores en ambos lados de 0 → colormap divergente centrado en 0
    norm_irct = plt.cm.colors.TwoSlopeNorm(vmin=vmin_irct, vcenter=0, vmax=vmax_irct)
    cmap_irct = 'RdYlGn'
    print("Usando colormap divergente (RdYlGn) centrado en 0")
elif has_positive:
    # Solo valores positivos → colormap secuencial
    norm_irct = plt.cm.colors.Normalize(vmin=vmin_irct, vmax=vmax_irct)
    cmap_irct = 'YlGn'
    print("Usando colormap secuencial (YlGn) - solo valores positivos")
else:
    # Solo valores negativos → colormap secuencial invertido
    norm_irct = plt.cm.colors.Normalize(vmin=vmin_irct, vmax=vmax_irct)
    cmap_irct = 'YlOrRd_r'
    print("Usando colormap secuencial invertido (YlOrRd_r) - solo valores negativos")

print()

# Panel 1: Base (coloreado por escenario original como referencia)
ax = axes[0, 0]
scenario_labels = (
    ["B245"] * N_PER_SCENARIO
    + ["B370"] * N_PER_SCENARIO
    + ["B585"] * N_PER_SCENARIO
)
color_map_base = {"B245": "tab:blue", "B370": "tab:orange", "B585": "tab:green"}
scatter = ax.scatter(
    z_base_2d[:, 0],
    z_base_2d[:, 1],
    c=[color_map_base[label] for label in scenario_labels],
    s=15,
    alpha=0.6,
    edgecolors='none'
)
ax.set_title("Base (por escenario)", fontsize=12, fontweight='bold')
ax.set_xlabel("PC1", fontsize=10)
ax.set_ylabel("PC2", fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2-4: Cada escenario futuro coloreado por su IRCT
scenarios_info = [
    ("SSP245", "T245", 0, 1),
    ("SSP370", "T370", 1, 0),
    ("SSP585", "T585", 1, 1)
]

for scenario_name, future_label, row, col in scenarios_info:
    ax = axes[row, col]
    
    # Obtener latentes y IRCT
    z_future = LATENTS["VAE"][future_label]
    z_future_2d = pca.transform(z_future)
    irct_vals = irct_by_scenario[scenario_name]
    
    # Scatter plot coloreado por IRCT
    scatter = ax.scatter(
        z_future_2d[:, 0],
        z_future_2d[:, 1],
        c=irct_vals,
        s=15,
        alpha=0.7,
        cmap=cmap_irct,
        norm=norm_irct,
        edgecolors='none'
    )
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('IRCT', fontsize=10)
    
    # Estadísticas del IRCT
    mean_irct = irct_vals.mean()
    std_irct = irct_vals.std()
    
    ax.set_title(
        f"{scenario_name}\nIRCT: μ={mean_irct:.3f}, σ={std_irct:.3f}",
        fontsize=12,
        fontweight='bold'
    )
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    "Espacio Latente (VAE) Coloreado por Índice de Resiliencia (IRCT)\nVerde = Mayor resiliencia | Rojo = Menor resiliencia",
    fontsize=14,
    fontweight='bold',
    y=0.995
)
plt.tight_layout()

# Guardar
save_path = os.path.join(PLOTS_DIR, f"latent_space_irct_{timestamp}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Gráfico guardado: {save_path}")

plt.show()

print()
print("Interpretación del gráfico:")
print("- Verde: Puntos con IRCT positivo (alta resiliencia, cambios favorables)")
print("- Amarillo: Puntos con IRCT cercano a 0 (resiliencia neutral)")
print("- Rojo: Puntos con IRCT negativo (baja resiliencia, cambios desfavorables)")
print()
print("Patrones espaciales en el espacio latente revelan:")
print("  • Agrupamientos de resiliencia similar")
print("  • Separación entre zonas resilientes y vulnerables")
print("  • Distribución espacial de la capacidad adaptativa")


