"""
Librería de clustering para modelos de autoencoder

Este módulo proporciona funciones para realizar clustering en espacios latentes
de autoencoders, calculando métricas de resiliencia territorial mediante
análisis de estabilidad de clusters entre escenarios base y futuros.

Funciones principales:
- compute_inv_covs_per_cluster: Calcula matrices de covarianza inversa (Mahalanobis)
- cluster_and_measure_resilience: Pipeline completo de clustering + métricas de resiliencia
- prepare_clustering_data: Prepara datos para clustering (estandarización + splits)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from typing import Dict, Tuple, Optional, Union, List

class SimpleSOM:
    """
    Implementación simple de Self-Organizing Maps (SOM) usando NumPy.
    Diseñada para ser compatible con la interfaz de sklearn (fit, predict).
    """
    def __init__(
        self, 
        x_dim: int, 
        y_dim: int, 
        input_len: int, 
        sigma: float = 1.0, 
        learning_rate: float = 0.5,
        random_state: int = 42,
        n_iterations: int = 1000
    ):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_iterations = n_iterations
        
        np.random.seed(random_state)
        self.weights = np.random.random((x_dim, y_dim, input_len))
        
    def _activate(self, x):
        """Calcula la distancia euclidiana entre x y todos los pesos."""
        # x shape: (input_len,)
        # weights shape: (x_dim, y_dim, input_len)
        # delta shape: (x_dim, y_dim, input_len)
        delta = x - self.weights
        # dist shape: (x_dim, y_dim)
        return np.linalg.norm(delta, axis=2)
        
    def _winner(self, x):
        """Encuentra las coordenadas (x, y) de la neurona ganadora (BMU)."""
        activation_map = self._activate(x)
        return np.unravel_index(activation_map.argmin(), activation_map.shape)
        
    def _update(self, x, winner, t, max_iter):
        """Actualiza los pesos de las neuronas vecinas."""
        # Decaimiento de learning rate y sigma
        lr = self.learning_rate * (1 - t / max_iter)
        sig = self.sigma * (1 - t / max_iter)
        
        # Rango de vecindad
        width = int(round(sig * 3)) # 3 sigma rule
        
        start_x = max(0, winner[0] - width)
        end_x = min(self.x_dim, winner[0] + width + 1)
        start_y = max(0, winner[1] - width)
        end_y = min(self.y_dim, winner[1] + width + 1)
        
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                # Distancia al ganador en la grilla
                dist_sq = (i - winner[0])**2 + (j - winner[1])**2
                
                # Función de vecindad gaussiana
                influence = np.exp(-dist_sq / (2 * (sig**2 + 1e-8)))
                
                # Actualización de pesos
                self.weights[i, j, :] += lr * influence * (x - self.weights[i, j, :])
                
    def fit(self, X):
        """Entrena el SOM."""
        n_samples = X.shape[0]
        
        for t in range(self.n_iterations):
            # Seleccionar muestra aleatoria
            idx = np.random.randint(0, n_samples)
            x = X[idx]
            
            # Encontrar ganador
            winner = self._winner(x)
            
            # Actualizar pesos
            self._update(x, winner, t, self.n_iterations)
            
        return self
        
    def predict(self, X):
        """Asigna cada muestra al cluster correspondiente (neurona ganadora linealizada)."""
        labels = []
        for x in X:
            winner = self._winner(x)
            # Convertir coordenadas 2D a índice plano 1D
            label = winner[0] * self.y_dim + winner[1]
            labels.append(label)
        return np.array(labels)
        
    @property
    def cluster_centers_(self):
        """Retorna los pesos como centroides aplanados (n_clusters, input_len)."""
        return self.weights.reshape(-1, self.input_len)


def compute_inv_covs_per_cluster(
    Z_base_scaled: np.ndarray,
    labels_base: np.ndarray,
    n_clusters: int,
    eps: float = 1e-6
) -> Dict[int, np.ndarray]:
    """
    Calcula matrices de covarianza inversa (Mahalanobis) por cluster.
    """
    Z = np.asarray(Z_base_scaled)
    inv_covs = {}
    
    # Manejar caso DBSCAN donde puede haber ruido (-1)
    unique_labels = np.unique(labels_base)
    unique_labels = unique_labels[unique_labels >= 0]
    
    # Si n_clusters es menor que el max label, ajustar (para DBSCAN dinámico)
    if len(unique_labels) > 0:
        max_label = int(unique_labels.max())
        if max_label >= n_clusters:
            n_clusters = max_label + 1

    for k in unique_labels:
        k = int(k)
        # Muestras del cluster k
        Zk = Z[labels_base == k]

        # Si el cluster es pequeño, usar identidad (métrica euclidiana)
        if len(Zk) < Z.shape[1] + 2:
            inv_covs[k] = np.eye(Z.shape[1])
            continue

        # Covarianza regularizada
        cov = np.cov(Zk.T)
        cov = cov + eps * np.eye(cov.shape[0])

        # Inversa (para métrica de Mahalanobis)
        inv_covs[k] = np.linalg.inv(cov)

    return inv_covs


def prepare_clustering_data(
    z_base: np.ndarray,
    z_T245: np.ndarray,
    z_T370: np.ndarray,
    z_T585: np.ndarray,
    n_per_scenario: int
) -> Tuple[StandardScaler, Dict[str, np.ndarray]]:
    """
    Prepara datos para clustering: estandarización y splits por escenario.
    """
    # Crear y ajustar scaler con datos BASE
    scaler = StandardScaler()
    scaler.fit(z_base)
    
    # Estandarizar todos los espacios latentes con los mismos parámetros
    z_base_scaled = scaler.transform(z_base)
    z_T245_scaled = scaler.transform(z_T245)
    z_T370_scaled = scaler.transform(z_T370)
    z_T585_scaled = scaler.transform(z_T585)
    
    # Dividir z_base_scaled por escenario
    z_B245_scaled = z_base_scaled[:n_per_scenario]
    z_B370_scaled = z_base_scaled[n_per_scenario:2*n_per_scenario]
    z_B585_scaled = z_base_scaled[2*n_per_scenario:]
    
    scaled_data = {
        'z_base_scaled': z_base_scaled,
        'z_T245_scaled': z_T245_scaled,
        'z_T370_scaled': z_T370_scaled,
        'z_T585_scaled': z_T585_scaled,
        'z_B245_scaled': z_B245_scaled,
        'z_B370_scaled': z_B370_scaled,
        'z_B585_scaled': z_B585_scaled,
    }
    
    return scaler, scaled_data


def compute_dunn_index(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Calcula el índice de Dunn simplificado.
    Dunn = min(inter_cluster_dist) / max(intra_cluster_diam)
    
    Args:
        X: Datos (N, D)
        labels: Etiquetas de cluster
        centroids: Centroides (K, D)
        
    Returns:
        Valor del índice de Dunn
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0] # Ignorar ruido
    
    if len(unique_labels) < 2:
        return 0.0
        
    # 1. Calcular distancias inter-cluster (entre centroides)
    # K x K matrix
    k = len(centroids)
    if k < 2: return 0.0
    
    # Solo considerar centroides de clusters que existen en labels
    # (DBSCAN puede tener centroides pre-calculados pero vacíos si filtramos ruido)
    active_centroids = []
    for lbl in unique_labels:
        if lbl < len(centroids):
            active_centroids.append(centroids[lbl])
    
    if len(active_centroids) < 2: return 0.0
    active_centroids = np.array(active_centroids)
    
    # Distancias entre pares de centroides
    dists = np.linalg.norm(active_centroids[:, np.newaxis] - active_centroids, axis=2)
    # Mask diagonal (distancia 0 consigo mismo)
    np.fill_diagonal(dists, np.inf)
    min_inter_dist = dists.min()
    
    # 2. Calcular diámetro intra-cluster (max distancia al centroide * 2 aprox)
    max_intra_diam = 0.0
    
    for lbl in unique_labels:
        mask = labels == lbl
        if not mask.any(): continue
        
        points = X[mask]
        if lbl < len(centroids):
            center = centroids[lbl]
            # Radio máximo (distancia al centroide)
            radii = np.linalg.norm(points - center, axis=1)
            diam = radii.max() * 2 # Aproximación del diámetro
            if diam > max_intra_diam:
                max_intra_diam = diam
                
    if max_intra_diam == 0:
        return 0.0
        
    return min_inter_dist / max_intra_diam


def compute_cluster_resilience_metrics(
    cluster_id: int,
    centroid: np.ndarray,
    labels_B245: np.ndarray,
    labels_B370: np.ndarray,
    labels_B585: np.ndarray,
    labels_T245: np.ndarray,
    labels_T370: np.ndarray,
    labels_T585: np.ndarray,
    z_B245_scaled: np.ndarray,
    z_B370_scaled: np.ndarray,
    z_B585_scaled: np.ndarray,
    z_T245_scaled: np.ndarray,
    z_T370_scaled: np.ndarray,
    z_T585_scaled: np.ndarray,
) -> Optional[Dict]:
    """
    Calcula métricas de resiliencia para un cluster individual.
    """
    # Máscaras de píxeles en este cluster para cada escenario BASE
    mask_B245 = (labels_B245 == cluster_id)
    mask_B370 = (labels_B370 == cluster_id)
    mask_B585 = (labels_B585 == cluster_id)
    
    # Calcular n_pixels promedio
    n_pixels_245 = mask_B245.sum()
    n_pixels_370 = mask_B370.sum()
    n_pixels_585 = mask_B585.sum()
    n_pixels = int((n_pixels_245 + n_pixels_370 + n_pixels_585) / 3)
    
    if n_pixels == 0:
        return None
        
    # Para DBSCAN, si no hay centroide definido, usar la media de los puntos base
    if centroid is None:
        points_base = []
        if mask_B245.any(): points_base.append(z_B245_scaled[mask_B245])
        if mask_B370.any(): points_base.append(z_B370_scaled[mask_B370])
        if mask_B585.any(): points_base.append(z_B585_scaled[mask_B585])
        
        if points_base:
            centroid = np.vstack(points_base).mean(axis=0)
        else:
            return None # No debería pasar si n_pixels > 0
    
    # --- Compacidad ---
    # Compacidad BASE (promedio de los 3 escenarios base)
    comp_base_245 = np.linalg.norm(z_B245_scaled[mask_B245] - centroid, axis=1).mean() if mask_B245.any() else 0
    comp_base_370 = np.linalg.norm(z_B370_scaled[mask_B370] - centroid, axis=1).mean() if mask_B370.any() else 0
    comp_base_585 = np.linalg.norm(z_B585_scaled[mask_B585] - centroid, axis=1).mean() if mask_B585.any() else 0
    compactness_base = (comp_base_245 + comp_base_370 + comp_base_585) / 3
    
    # Compacidad en FUTUROS
    compactness_T245 = np.linalg.norm(z_T245_scaled[mask_B245] - centroid, axis=1).mean() if mask_B245.any() else 0
    compactness_T370 = np.linalg.norm(z_T370_scaled[mask_B370] - centroid, axis=1).mean() if mask_B370.any() else 0
    compactness_T585 = np.linalg.norm(z_T585_scaled[mask_B585] - centroid, axis=1).mean() if mask_B585.any() else 0
    
    # Expansión relativa (% de cambio en compacidad)
    expansion_T245 = (compactness_T245 / compactness_base - 1) * 100 if compactness_base > 0 else 0
    expansion_T370 = (compactness_T370 / compactness_base - 1) * 100 if compactness_base > 0 else 0
    expansion_T585 = (compactness_T585 / compactness_base - 1) * 100 if compactness_base > 0 else 0
    
    # --- Estabilidad (Recall) ---
    # % de píxeles originales que permanecen en el mismo cluster
    stability_T245 = (labels_T245[mask_B245] == cluster_id).sum() / n_pixels_245 * 100 if n_pixels_245 > 0 else 0
    stability_T370 = (labels_T370[mask_B370] == cluster_id).sum() / n_pixels_370 * 100 if n_pixels_370 > 0 else 0
    stability_T585 = (labels_T585[mask_B585] == cluster_id).sum() / n_pixels_585 * 100 if n_pixels_585 > 0 else 0
    
    # --- Jaccard (IoU) ---
    # Intersection: Píxeles que eran k y siguen siendo k (TP)
    # Union: Píxeles que eran k (TP+FN) + Píxeles que ahora son k (TP+FP)
    # J = TP / (TP + FN + FP)
    
    def calc_jaccard(mask_base, labels_fut, cid):
        if not mask_base.any(): return 0.0
        
        # Set A: índices donde mask_base es True
        # Set B: índices donde labels_fut == cid
        
        # Intersección: (mask_base) AND (labels_fut == cid)
        intersection = (mask_base & (labels_fut == cid)).sum()
        
        # Unión: (mask_base) OR (labels_fut == cid)
        union = (mask_base | (labels_fut == cid)).sum()
        
        return (intersection / union * 100) if union > 0 else 0.0

    jaccard_T245 = calc_jaccard(mask_B245, labels_T245, cluster_id)
    jaccard_T370 = calc_jaccard(mask_B370, labels_T370, cluster_id)
    jaccard_T585 = calc_jaccard(mask_B585, labels_T585, cluster_id)
    
    # --- Latent Centroid Drift ---
    # Distancia entre el centroide original y el centroide de los puntos proyectados en el futuro
    # Centroid T585: media de los puntos en T585 que fueron asignados al cluster k
    
    def calc_drift(z_fut, labels_fut, cid, base_cent):
        mask_fut = (labels_fut == cid)
        if not mask_fut.any(): return 0.0
        
        cent_fut = z_fut[mask_fut].mean(axis=0)
        return np.linalg.norm(base_cent - cent_fut)

    drift_T245 = calc_drift(z_T245_scaled, labels_T245, cluster_id, centroid)
    drift_T370 = calc_drift(z_T370_scaled, labels_T370, cluster_id, centroid)
    drift_T585 = calc_drift(z_T585_scaled, labels_T585, cluster_id, centroid)
    
    return {
        "cluster_id": cluster_id,
        "n_pixels": n_pixels,
        # Compacidad
        "compactness_base": compactness_base,
        "compactness_T585": compactness_T585,
        "expansion_T585": expansion_T585,
        # Estabilidad (Recall)
        "stability_T245": stability_T245,
        "stability_T370": stability_T370,
        "stability_T585": stability_T585,
        # Jaccard (IoU)
        "jaccard_T245": jaccard_T245,
        "jaccard_T370": jaccard_T370,
        "jaccard_T585": jaccard_T585,
        # Drift
        "drift_T245": drift_T245,
        "drift_T370": drift_T370,
        "drift_T585": drift_T585,
    }


def calculate_clustering_quality(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de calidad de clustering intrínsecas."""
    # Filtrar ruido (-1) para métricas si es DBSCAN
    mask = labels != -1
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
            "dunn_index": np.nan
        }
        
    X_clean = X[mask]
    labels_clean = labels[mask]
    
    # Subsamplear si es muy grande para silhouette (es O(N^2))
    if len(X_clean) > 5000:
        idx = np.random.choice(len(X_clean), 5000, replace=False)
        X_sil = X_clean[idx]
        labels_sil = labels_clean[idx]
    else:
        X_sil = X_clean
        labels_sil = labels_clean
        
    return {
        "silhouette": silhouette_score(X_sil, labels_sil),
        "calinski_harabasz": calinski_harabasz_score(X_clean, labels_clean),
        "davies_bouldin": davies_bouldin_score(X_clean, labels_clean),
        "dunn_index": compute_dunn_index(X_clean, labels_clean, centroids)
    }


def cluster_and_measure_resilience(
    latents: Dict[str, np.ndarray],
    n_per_scenario: int,
    method: str = 'kmeans',
    n_clusters: int = 10,
    random_state: int = 42,
    n_init: int = 50,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    som_x: int = 10,
    som_y: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Pipeline completo de clustering y cálculo de métricas de resiliencia.
    """
    # Extraer latentes
    z_base = latents["base"]
    z_T245 = latents["T245"]
    z_T370 = latents["T370"]
    z_T585 = latents["T585"]
    
    if verbose:
        print(f"\nPreparando datos para clustering ({method.upper()})")
        print(f"  z_base: {z_base.shape}")
        print(f"  z_T245: {z_T245.shape}")
    
    # 1. Preparar datos (estandarización + splits)
    scaler, scaled_data = prepare_clustering_data(
        z_base, z_T245, z_T370, z_T585, n_per_scenario
    )
    
    z_base_scaled = scaled_data['z_base_scaled']
    z_T245_scaled = scaled_data['z_T245_scaled']
    z_T370_scaled = scaled_data['z_T370_scaled']
    z_T585_scaled = scaled_data['z_T585_scaled']
    z_B245_scaled = scaled_data['z_B245_scaled']
    z_B370_scaled = scaled_data['z_B370_scaled']
    z_B585_scaled = scaled_data['z_B585_scaled']
    
    # 2. Clustering en BASE estandarizado
    model = None
    labels_base = None
    centroids_base = None
    
    if method.lower() == 'kmeans':
        if verbose: print(f"\nRealizando KMeans con K={n_clusters}")
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        labels_base = model.fit_predict(z_base_scaled)
        centroids_base = model.cluster_centers_
        
    elif method.lower() == 'dbscan':
        if verbose: print(f"\nRealizando DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples})")
        model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels_base = model.fit_predict(z_base_scaled)
        # DBSCAN no tiene centroides explícitos, los calculamos a posteriori
        unique_labels = set(labels_base)
        if -1 in unique_labels: unique_labels.remove(-1)
        n_clusters = len(unique_labels)
        
        # Calcular centroides como media de puntos
        centroids_dict = {}
        for k in unique_labels:
            centroids_dict[k] = z_base_scaled[labels_base == k].mean(axis=0)
        
        # Convertir a array si los labels son secuenciales 0..K-1 (DBSCAN suele serlo)
        if n_clusters > 0:
            max_label = max(unique_labels)
            centroids_base = np.zeros((max_label + 1, z_base_scaled.shape[1]))
            for k, center in centroids_dict.items():
                centroids_base[k] = center
        else:
            centroids_base = np.zeros((0, z_base_scaled.shape[1]))
            
    elif method.lower() == 'som':
        if verbose: print(f"\nRealizando SOM ({som_x}x{som_y})")
        model = SimpleSOM(
            x_dim=som_x, 
            y_dim=som_y, 
            input_len=z_base_scaled.shape[1],
            random_state=random_state
        )
        model.fit(z_base_scaled)
        labels_base = model.predict(z_base_scaled)
        centroids_base = model.cluster_centers_
        n_clusters = som_x * som_y
        
    else:
        raise ValueError(f"Método desconocido: {method}")
    
    # Calcular métricas de calidad intrínseca
    quality_metrics = calculate_clustering_quality(z_base_scaled, labels_base, centroids_base)
    if verbose:
        print(f"\nCalidad del clustering:")
        for k, v in quality_metrics.items():
            print(f"  {k}: {v:.4f}")

    # Calcular matrices de covarianza inversa (si aplica)
    inv_covs = compute_inv_covs_per_cluster(z_base_scaled, labels_base, n_clusters)
    
    if verbose:
        print(f"\nDistribución de píxeles en clusters BASE:")
        unique, counts = np.unique(labels_base, return_counts=True)
        for k, count in zip(unique, counts):
            pct = count / len(labels_base) * 100
            label_name = f"Cluster {k}" if k != -1 else "Ruido (-1)"
            print(f"  {label_name}: {count} píxeles ({pct:.1f}%)")
    
    # 3. Proyectar futuros a clusters BASE
    if method.lower() == 'kmeans':
        labels_T245 = model.predict(z_T245_scaled)
        labels_T370 = model.predict(z_T370_scaled)
        labels_T585 = model.predict(z_T585_scaled)
    elif method.lower() == 'som':
        labels_T245 = model.predict(z_T245_scaled)
        labels_T370 = model.predict(z_T370_scaled)
        labels_T585 = model.predict(z_T585_scaled)
    elif method.lower() == 'dbscan':
        # DBSCAN no tiene predict(). Asignamos al centroide más cercano.
        def assign_nearest(X, centroids):
            if len(centroids) == 0:
                return np.full(X.shape[0], -1)
            dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            return np.argmin(dists, axis=1)
            
        labels_T245 = assign_nearest(z_T245_scaled, centroids_base)
        labels_T370 = assign_nearest(z_T370_scaled, centroids_base)
        labels_T585 = assign_nearest(z_T585_scaled, centroids_base)
    
    # 4. Calcular ARI (Adjusted Rand Index) entre BASE y FUTUROS
    # Nota: Comparamos labels_base (ground truth temporal) vs labels_futuro
    # Pero labels_base tiene 3x puntos. Necesitamos alinear.
    # Opción: Calcular ARI por escenario.
    
    # Dividir labels_base por escenario
    labels_B245 = labels_base[:n_per_scenario]
    labels_B370 = labels_base[n_per_scenario:2*n_per_scenario]
    labels_B585 = labels_base[2*n_per_scenario:]
    
    ari_T245 = adjusted_rand_score(labels_B245, labels_T245)
    ari_T370 = adjusted_rand_score(labels_B370, labels_T370)
    ari_T585 = adjusted_rand_score(labels_B585, labels_T585)
    
    quality_metrics["ARI_T245"] = ari_T245
    quality_metrics["ARI_T370"] = ari_T370
    quality_metrics["ARI_T585"] = ari_T585
    
    if verbose:
        print(f"  ARI (Estabilidad Global) T585: {ari_T585:.4f}")
    
    # 5. Calcular métricas de resiliencia por cluster
    if verbose:
        print(f"\nCalculando métricas de resiliencia por cluster...")
    
    cluster_resilience = []
    
    # Iterar sobre clusters válidos (excluyendo ruido -1)
    unique_labels = np.unique(labels_base)
    unique_labels = unique_labels[unique_labels >= 0]
    
    for cluster_id in unique_labels:
        # Obtener centroide (si existe)
        centroid = None
        if centroids_base is not None and 0 <= cluster_id < len(centroids_base):
            centroid = centroids_base[cluster_id]
            
        metrics = compute_cluster_resilience_metrics(
            cluster_id=int(cluster_id),
            centroid=centroid,
            labels_B245=labels_B245,
            labels_B370=labels_B370,
            labels_B585=labels_B585,
            labels_T245=labels_T245,
            labels_T370=labels_T370,
            labels_T585=labels_T585,
            z_B245_scaled=z_B245_scaled,
            z_B370_scaled=z_B370_scaled,
            z_B585_scaled=z_B585_scaled,
            z_T245_scaled=z_T245_scaled,
            z_T370_scaled=z_T370_scaled,
            z_T585_scaled=z_T585_scaled,
        )
        
        if metrics is not None:
            cluster_resilience.append(metrics)
    
    df_cluster_resilience = pd.DataFrame(cluster_resilience)
    
    # Resultados consolidados
    results = {
        "model": model,
        "method": method,
        "labels_base": labels_base,
        "labels_T245": labels_T245,
        "labels_T370": labels_T370,
        "labels_T585": labels_T585,
        "labels_B245": labels_B245,
        "labels_B370": labels_B370,
        "labels_B585": labels_B585,
        "resilience_df": df_cluster_resilience,
        "quality_metrics": quality_metrics,
        "scaler": scaler,
        "centroids": centroids_base,
        "z_B245_scaled": z_B245_scaled,
        "z_B370_scaled": z_B370_scaled,
        "z_B585_scaled": z_B585_scaled,
        "z_T245_scaled": z_T245_scaled,
        "z_T370_scaled": z_T370_scaled,
        "z_T585_scaled": z_T585_scaled,
        "inv_covs": inv_covs,
    }
    
    if verbose and len(df_cluster_resilience) > 0:
        print(f"\nMétricas de resiliencia por cluster (Top 5):")
        cols = ["cluster_id", "n_pixels", "jaccard_T585", "drift_T585", "stability_T585"]
        print(df_cluster_resilience[cols].head().to_string(index=False))
        
        print(f"\nInterpretación:")
        print(f"  Jaccard T585: {df_cluster_resilience['jaccard_T585'].mean():.1f}% promedio")
        print(f"  Drift T585: {df_cluster_resilience['drift_T585'].mean():.4f} promedio")
        
    return results


def get_cluster_summary(results: Dict) -> pd.DataFrame:
    """
    Genera un resumen consolidado de las métricas de clustering.
    """
    df = results['resilience_df'].copy()
    if len(df) == 0:
        return pd.DataFrame()
        
    # Calcular métricas agregadas
    df['avg_jaccard'] = df[['jaccard_T245', 'jaccard_T370', 'jaccard_T585']].mean(axis=1)
    df['avg_drift'] = df[['drift_T245', 'drift_T370', 'drift_T585']].mean(axis=1)
    df['avg_stability'] = df[['stability_T245', 'stability_T370', 'stability_T585']].mean(axis=1)
    
    # Score de resiliencia compuesto
    # Mayor Jaccard (overlap) y Menor Drift = Mejor
    # Score = Jaccard - (Drift * factor)
    # Drift suele ser < 1.0 en espacio escalado si es pequeño, o > 1.0 si es grande.
    # Jaccard es 0-100.
    # Normalicemos drift? Simplemente usaremos Jaccard como proxy principal de "identidad territorial".
    df['resilience_score'] = df['avg_jaccard']
    
    return df[['cluster_id', 'n_pixels', 'avg_jaccard', 'avg_drift', 'resilience_score']].sort_values(
        'resilience_score', ascending=False
    )
