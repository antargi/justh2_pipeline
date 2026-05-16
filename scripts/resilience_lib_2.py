"""
Librería de Resiliencia Climática - IRCT y Validaciones

Módulo para cálculo del Índice de Resiliencia Climática-Territorial (IRCT) 
y validación multidimensional en diferentes contextos de energía y clima.

Componentes principales:
- Funciones de componentes IRCT (reconstrucción, desplazamiento, estabilidad, expansión, H2)
- Funciones de validación (escenario, espacial, estructural, robustez)
- Utilidades de análisis (Moran's I, correlaciones, perturbaciones)

Autor: Análisis de resiliencia climática en valle de hidrógeno
"""

import numpy as np
import pandas as pd
from scipy.stats import (
    wilcoxon, ks_2samp, spearmanr, mannwhitneyu, norm, 
    kendalltau
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)


# =============================================================================
# 1. FUNCIONES DE COMPONENTES IRCT
# =============================================================================

def percentile_rank(arr):
    """
    Calcula el percentil rank (0-1) de un array, manejando empates 
    promediando sus ranks.
    
    Retorna valores en [0, 1] con extremos exactos:
    - rank: 1..n (average ties)
    - mapeo: (rank-1)/(n-1) ∈ [0,1]
    """
    arr = np.asarray(arr)
    n = len(arr)
    if n <= 1:
        return np.zeros_like(arr, dtype=float)
    
    import pandas as _pd
    ranks = _pd.Series(arr).rank(method="average").to_numpy()
    return (ranks - 1.0) / (n - 1.0)


def compute_cluster_stability_softmax(z_scaled, centroids_base, labels_base, tau=1.0):
    """
    Estabilidad de pertenencia al clúster (Componente 3: S_C).
    
    Calcula softmax de distancias negativas con temperatura para medir
    la probabilidad de que un píxel mantiene identidad de cluster.
    
    Args:
        z_scaled: datos en espacio latente (n_samples, latent_dim)
        centroids_base: centroides de clusters en espacio latente (k_clusters, latent_dim)
        labels_base: asignaciones cluster para cada muestra (n_samples,)
        tau: temperatura para softmax (default=1.0)
    
    Returns:
        S_C: estabilidad (0-1, 1=más resiliente)
        all_distances: matriz de distancias a todos los centroides
    """
    z = np.asarray(z_scaled)
    C = np.asarray(centroids_base)

    # Distancias euclidianas vectorizado
    zz = np.sum(z*z, axis=1, keepdims=True)
    cc = np.sum(C*C, axis=1, keepdims=True).T
    all_distances = np.sqrt(np.maximum(zz + cc - 2*np.dot(z, C.T), 0.0))

    # Normalizar distancias para comparabilidad entre modelos
    d_std = np.std(all_distances)
    d_std = d_std if d_std > 0 else 1.0
    norm_d = all_distances / d_std

    # Softmax estable con temperatura
    logits = -norm_d / max(tau, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    softmax_probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)

    idx = np.arange(len(z))
    S_C = softmax_probs[idx, labels_base]
    
    return S_C, all_distances


def compute_latent_displacement(z_scaled, centroids_base, labels_base, *, inv_covs_base=None):
    """
    Desplazamiento latente (Componente 2: S_D).
    
    Mide la distancia del píxel a su centroide de cluster, con opciones
    de métrica euclídea o Mahalanobis.
    
    Args:
        z_scaled: datos en espacio latente (n_samples, latent_dim)
        centroids_base: centroides (k_clusters, latent_dim)
        labels_base: asignaciones cluster (n_samples,)
        inv_covs_base: matrices covarianza inversas por cluster (opcional)
    
    Returns:
        S_D: estabilidad por desplazamiento (0-1, 1=más resiliente)
        distances: distancias a centroide asignado
    """
    z = np.asarray(z_scaled)
    C = np.asarray(centroids_base)
    n = len(z)
    distances = np.empty(n)

    if inv_covs_base is None:
        # Métrica euclídea
        for i in range(n):
            k = labels_base[i]
            distances[i] = np.linalg.norm(z[i] - C[k])
    else:
        # Métrica Mahalanobis por clúster
        for i in range(n):
            k = labels_base[i]
            d = z[i] - C[k]
            distances[i] = np.sqrt(np.dot(d, inv_covs_base[k].dot(d)))

    S_D = 1 - percentile_rank(distances)
    return S_D, distances


def compute_cluster_expansion(z_base_scaled, z_future_scaled, centroids_base, 
                             labels_base, epsilon=1e-8, p99_clip=False):
    """
    Expansión del clúster (Componente 4: S_E).
    
    Mide la estabilidad del radio del clúster comparando distancias 
    en periodo base vs futuro.
    
    Args:
        z_base_scaled: datos BASE en espacio latente
        z_future_scaled: datos FUTUROS en espacio latente
        centroids_base: centroides (invariante)
        labels_base: asignaciones cluster en BASE
        epsilon: valor mínimo para evitar div/0
        p99_clip: recortar ratio a percentil 99
    
    Returns:
        S_E: estabilidad expansión (0-1, 1=poco cambio)
        ratio: ratios F/B de distancias futuro/base
    """
    zB, zF, C = map(np.asarray, (z_base_scaled, z_future_scaled, centroids_base))
    n = len(zB)
    B = np.empty(n)
    F = np.empty(n)

    for i in range(n):
        k = labels_base[i]
        B[i] = np.linalg.norm(zB[i] - C[k])
        F[i] = np.linalg.norm(zF[i] - C[k])

    ratio = F / (B + epsilon)
    
    # Cambio simétrico: magnitud del cambio en log-space
    delta = np.abs(np.log(np.clip(ratio, 1e-12, None)))

    if p99_clip:
        p99 = np.percentile(delta, 99)
        delta = np.clip(delta, 0, p99)

    S_E = 1 - percentile_rank(delta)
    return S_E, ratio


def compute_h2_stability(h2_base, h2_future, epsilon=1e-8, p99_clip=True):
    """
    Estabilidad energética de H₂ (Componente 5: S_H2).
    
    Mide la estabilidad del potencial de producción de hidrógeno
    comparando periodo base vs futuro.
    
    Args:
        h2_base: producción H2 en BASE (n_samples,)
        h2_future: producción H2 en FUTURO (n_samples,)
        epsilon: valor mínimo para evitar div/0
        p99_clip: recortar ratio a percentil 99
    
    Returns:
        S_H2: estabilidad energética (0-1, 1=más resiliente)
        h2_ratios: ratios futuro/base
    """
    h2_ratios = h2_future / (h2_base + epsilon)
    
    if p99_clip:
        p99 = np.percentile(h2_ratios, 99)
        h2_ratios = np.clip(h2_ratios, 0, p99)
    else:
        h2_ratios = np.clip(h2_ratios, 0, None)
    
    perc_ranks = percentile_rank(h2_ratios)
    S_H2 = perc_ranks
    
    return S_H2, h2_ratios


def compute_reconstruction_anomaly(
    model,
    X_orig,
    X_normalized,
    device='cpu',
    *,
    use_normalized=True,
    inverse_transform=None,
    reduce='mse',
    output='stability'
):
    """
    Anomalía de reconstrucción (Componente 1: A).
    
    Calcula error de reconstrucción por muestra normalizado a percentil rank.
    
    Args:
        model: modelo AE/VAE entrenado
        X_orig: datos originales (sin normalizar)
        X_normalized: datos normalizados
        device: 'cpu' o 'cuda'
        use_normalized: usar X_normalized como entrada
        inverse_transform: función para des-normalizar (si use_normalized=False)
        reduce: 'mse' (default) o 'mae' para métrica de error
        output: 'stability' (1=mejor) o 'anomaly' (1=peor)
    
    Returns:
        A: anomalía de reconstrucción (0-1)
        reconstruction_errors: errores crudos por muestra
    """
    model.eval()
    with torch.no_grad():
        X_in = X_normalized if use_normalized else X_orig
        X_tensor = torch.as_tensor(X_in, dtype=torch.float32, device=device)
        
        if isinstance(model, torch.nn.Module):
            # Detectar si es VAE o AE
            try:
                output_tuple = model(X_tensor)
                if isinstance(output_tuple, tuple) and len(output_tuple) == 3:
                    x_hat, _, _ = output_tuple  # VAE
                else:
                    x_hat = output_tuple
            except:
                x_hat, _ = model(X_tensor)  # AE
        
        x_hat_np = x_hat.cpu().numpy() if isinstance(x_hat, torch.Tensor) else x_hat

    if use_normalized:
        target = np.asarray(X_normalized)
    else:
        if inverse_transform is None:
            raise ValueError("Proporciona inverse_transform para des-normalizar.")
        x_hat_np = inverse_transform(x_hat_np)
        target = np.asarray(X_orig)

    if reduce == 'mae':
        reconstruction_errors = np.mean(np.abs(target - x_hat_np), axis=1)
    else:
        reconstruction_errors = np.mean((target - x_hat_np) ** 2, axis=1)

    try:
        import pandas as _pd
        ranks = _pd.Series(reconstruction_errors).rank(method="average").to_numpy()
        n = len(reconstruction_errors)
        pr = (ranks - 1.0) / (n - 1.0) if n > 1 else np.zeros_like(reconstruction_errors, dtype=float)
    except:
        order = np.argsort(reconstruction_errors)
        pr = np.empty_like(reconstruction_errors, dtype=float)
        pr[order] = np.linspace(0.0, 1.0, len(reconstruction_errors))

    if output == 'stability':
        A = 1.0 - pr
    elif output == 'anomaly':
        A = pr
    else:
        raise ValueError("output debe ser 'stability' o 'anomaly'.")

    return A, reconstruction_errors


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
    recon_use_normalized=True,
    inverse_transform=None,
    softmax_tau=1.0,
    expansion_p99_clip=False,
    h2_p99_clip=True,
    eps=1e-8
):
    """
    Calcula el IRCT pixel-wise como media geométrica ponderada.
    
    IRCT_i^(s) = A_i^{w_a} * S_D,i^{w_d} * S_C,i^{w_c} * S_E,i^{w_e} * S_H2,i^{w_h}
    
    Args:
        model: modelo AE/VAE
        X_base_orig/norm: datos BASE originales y normalizados
        X_future_orig/norm: datos FUTUROS originales y normalizados
        z_base_scaled: datos BASE en espacio latente escalado
        z_future_scaled: datos FUTUROS en espacio latente escalado
        centroids_base: centroides de clusters
        labels_base: asignaciones de cluster (BASE)
        h2_base/future: producción de H2 (opcional)
        weights: dict con w_a, w_d, w_c, w_e, w_h (default: pesos iguales)
        device: 'cpu' o 'cuda'
        recon_use_normalized: usar normalizado en reconstrucción
        softmax_tau: temperatura para softmax
        expansion_p99_clip: recortar expansión a p99
        h2_p99_clip: recortar H2 a p99
        eps: piso numérico
    
    Returns:
        dict con IRCT y componentes intermedios
    """
    if weights is None:
        weights = {'w_a': 0.15, 'w_d': 0.30, 'w_c': 0.25, 'w_e': 0.20, 'w_h': 0.10}
    
    # Normalizar pesos
    wsum = sum(weights.values())
    if not np.isclose(wsum, 1.0):
        weights = {k: v / (wsum if wsum else 1.0) for k, v in weights.items()}

    # 1. Anomalía de reconstrucción
    A, recon_errors = compute_reconstruction_anomaly(
        model,
        X_future_orig,
        X_future_norm,
        device=device,
        use_normalized=recon_use_normalized,
        inverse_transform=inverse_transform
    )

    # 2. Desplazamiento latente
    S_D, latent_distances = compute_latent_displacement(
        z_future_scaled, centroids_base, labels_base
    )

    # 3. Estabilidad de pertenencia
    S_C, all_distances = compute_cluster_stability_softmax(
        z_future_scaled, centroids_base, labels_base, tau=softmax_tau
    )

    # 4. Expansión del clúster
    S_E, expansion_ratios = compute_cluster_expansion(
        z_base_scaled, z_future_scaled, centroids_base, labels_base,
        p99_clip=expansion_p99_clip
    )

    # 5. Estabilidad H2
    if h2_base is not None and h2_future is not None:
        S_H2, h2_ratios = compute_h2_stability(h2_base, h2_future, p99_clip=h2_p99_clip)
        use_h2 = True
    else:
        n = len(A)
        S_H2 = np.ones(n)
        h2_ratios = np.ones(n)
        use_h2 = False

    # 6. Chequeos de integridad
    n = len(A)
    assert all(len(x) == n for x in [S_D, S_C, S_E, S_H2]), "Longitudes inconsistentes"
    
    for arr in (A, S_D, S_C, S_E, S_H2):
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(arr, 0.0, 1.0, out=arr)

    # 7. IRCT como media geométrica ponderada
    log_IRCT = (
        weights['w_a'] * np.log(np.maximum(A, eps)) +
        weights['w_d'] * np.log(np.maximum(S_D, eps)) +
        weights['w_c'] * np.log(np.maximum(S_C, eps)) +
        weights['w_e'] * np.log(np.maximum(S_E, eps)) +
        weights['w_h'] * np.log(np.maximum(S_H2, eps))
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


# =============================================================================
# 2. FUNCIONES DE VALIDACIÓN
# =============================================================================

def _paired(a, b):
    """Emparejar arrays y filtrar valores no finitos."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    n = min(len(a), len(b))
    return a[:n], b[:n]


def validate_scenario_coherence(irct_results_dict, model_order):
    """
    VALIDACIÓN 1: Coherencia por escenario.
    
    Verifica que IRCT(SSP245) > IRCT(SSP370) > IRCT(SSP585)
    usando tests de Wilcoxon y Kolmogorov-Smirnov.
    
    Args:
        irct_results_dict: dict {model_key: {scenario: IRCT_array}}
        model_order: lista de claves de modelos
    
    Returns:
        dict con resultados de validación por modelo
    """
    results = {}
    
    for model_key in model_order:
        irct_245 = irct_results_dict[model_key]['SSP245']['IRCT']
        irct_370 = irct_results_dict[model_key]['SSP370']['IRCT']
        irct_585 = irct_results_dict[model_key]['SSP585']['IRCT']
        
        median_245 = np.median(irct_245[np.isfinite(irct_245)])
        median_370 = np.median(irct_370[np.isfinite(irct_370)])
        median_585 = np.median(irct_585[np.isfinite(irct_585)])
        
        # Verificar orden esperado
        order_check = (median_245 > median_370) and (median_370 > median_585)
        
        # Tests estadísticos
        x_245_370, y_245_370 = _paired(irct_245, irct_370)
        x_370_585, y_370_585 = _paired(irct_370, irct_585)
        x_245_585, y_245_585 = _paired(irct_245, irct_585)
        
        stat_245_370, p_245_370 = wilcoxon(x_245_370, y_245_370, alternative='greater')
        stat_370_585, p_370_585 = wilcoxon(x_370_585, y_370_585, alternative='greater')
        stat_245_585, p_245_585 = wilcoxon(x_245_585, y_245_585, alternative='greater')
        
        ks_stat_245_370, ks_p_245_370 = ks_2samp(x_245_370, y_245_370, alternative='greater')
        ks_stat_370_585, ks_p_370_585 = ks_2samp(x_370_585, y_370_585, alternative='greater')
        ks_stat_245_585, ks_p_245_585 = ks_2samp(x_245_585, y_245_585, alternative='greater')
        
        results[model_key] = {
            'median_245': median_245,
            'median_370': median_370,
            'median_585': median_585,
            'order_correct': order_check,
            'wilcoxon_245_370_p': p_245_370,
            'wilcoxon_370_585_p': p_370_585,
            'wilcoxon_245_585_p': p_245_585,
            'ks_245_370_p': ks_p_245_370,
            'ks_370_585_p': ks_p_370_585,
            'ks_245_585_p': ks_p_245_585,
        }
    
    return results


def _haversine_matrix(coords):
    """Matriz de distancias haversine (km) para coords lat/lon en grados."""
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
    Calcula Moran's I con kNN simétrico y p-valores por permutación.
    
    Args:
        values: array de valores IRCT
        coords: array de coordenadas (lat, lon)
        k_neighbors: número de vecinos para kNN
        n_perm: número de permutaciones
        use_normal_approx: calcular también p-valor normal (aproximado)
    
    Returns:
        I, expected_I, z_score, p_perm, p_norm
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 4:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    dist_matrix = _haversine_matrix(coords)

    # kNN binario
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        nn = np.argsort(dist_matrix[i])[1:k_neighbors+1]
        W[i, nn] = 1.0

    # Simetrizar
    W = ((W + W.T) > 0).astype(float)

    # Row-standardize
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

    expected_I = -1.0 / (n - 1)

    # Permutaciones
    count_ge = 1
    for _ in range(n_perm):
        xp = np.random.permutation(xc)
        num_p = np.sum(W * np.outer(xp, xp))
        I_p = (n / W_sum) * (num_p / denom)
        if I_p >= I:
            count_ge += 1
    p_perm = count_ge / (n_perm + 1)

    # Aproximación normal (opcional)
    if use_normal_approx:
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


def validate_spatial_stability(irct_results_dict, coords_df, model_order):
    """
    VALIDACIÓN 2: Estabilidad espacial (Moran's I).
    
    Verifica que IRCT presenta patrones geográficos coherentes.
    
    Args:
        irct_results_dict: dict con IRCT por modelo/escenario
        coords_df: DataFrame con columnas 'lat', 'lon'
        model_order: lista de claves de modelos
    
    Returns:
        dict con resultados de Moran's I por modelo/escenario
    """
    results = {}
    coords_array = coords_df[['lat', 'lon']].values
    
    for model_key in model_order:
        results[model_key] = {}
        
        for scenario_name in ['SSP245', 'SSP370', 'SSP585']:
            irct_vals = irct_results_dict[model_key][scenario_name]['IRCT']
            
            I, expected_I, z_norm, p_perm, p_norm = compute_morans_i(
                irct_vals, coords_array, k_neighbors=8, n_perm=999, use_normal_approx=True
            )
            
            is_significant = (p_perm < 0.05)
            is_positive = I > 0
            
            results[model_key][scenario_name] = {
                'morans_I': I,
                'expected_I': expected_I,
                'z_score_normal': z_norm,
                'p_value_perm_one_sided_pos': p_perm,
                'p_value_normal_two_sided': p_norm,
                'is_significant_perm': is_significant,
                'is_positive': is_positive
            }
    
    return results


def cliffs_delta(x, y):
    """Tamaño de efecto de Cliff para dos muestras."""
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    count = 0
    for xi in x:
        count += np.sum(xi > y) - np.sum(xi < y)
    return count / (nx * ny)


def holm_bonferroni(pvals):
    """
    Ajuste Holm-Bonferroni para múltiples comparaciones.
    
    Returns:
        p-valores ajustados (monótonos)
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    for k, idx in enumerate(order):
        adj[idx] = max(pvals[order[:k+1]]) * (m - k)
    adj = np.minimum.accumulate(adj[np.argsort(order)[::-1]])[::-1]
    adj = np.clip(adj, 0, 1)
    return adj


def validate_structural_stability(irct_results_dict, clustering_results_dict, model_order):
    """
    VALIDACIÓN 3: Estabilidad estructural (IRCT vs retención de cluster).
    
    Verifica correlación positiva entre IRCT y tasa de retención.
    
    Args:
        irct_results_dict: dict con componentes y IRCT
        clustering_results_dict: dict con labels y resultados clustering
        model_order: lista de claves de modelos
    
    Returns:
        dict con resultados estructurales por modelo/escenario
    """
    results = {}
    
    for model_key in model_order:
        results[model_key] = {}
        
        pvals_spearman = []
        pvals_u = []
        scen_store = {}
        
        for scenario_name in ['SSP245', 'SSP370', 'SSP585']:
            irct_vals = irct_results_dict[model_key][scenario_name]['IRCT']
            
            # Obtener labels según escenario
            if scenario_name == 'SSP245':
                labels_b = clustering_results_dict[model_key]['labels_B245']
                labels_t = clustering_results_dict[model_key]['labels_T245']
            elif scenario_name == 'SSP370':
                labels_b = clustering_results_dict[model_key]['labels_B370']
                labels_t = clustering_results_dict[model_key]['labels_T370']
            else:
                labels_b = clustering_results_dict[model_key]['labels_B585']
                labels_t = clustering_results_dict[model_key]['labels_T585']
            
            # Estabilidad de partición
            ari = adjusted_rand_score(labels_b, labels_t)
            nmi = normalized_mutual_info_score(labels_b, labels_t)
            
            # Análisis de píxeles retenidos vs migrados
            retained_mask = (labels_b == labels_t)
            irct_retained = irct_vals[retained_mask]
            irct_migrated = irct_vals[~retained_mask]
            
            # Mann-Whitney U
            if (len(irct_retained) > 0) and (len(irct_migrated) > 0):
                stat_u, p_diff = mannwhitneyu(irct_retained, irct_migrated, alternative='greater')
                delta = cliffs_delta(irct_retained, irct_migrated)
            else:
                p_diff, delta = np.nan, np.nan
            
            pvals_spearman.append(p_diff if np.isfinite(p_diff) else 1.0)
            pvals_u.append(p_diff if np.isfinite(p_diff) else 1.0)
            
            scen_store[scenario_name] = {
                'ari': ari,
                'nmi': nmi,
                'irct_retained_mean': float(np.nanmean(irct_retained)) if irct_retained.size else np.nan,
                'irct_migrated_mean': float(np.nanmean(irct_migrated)) if irct_migrated.size else np.nan,
                'retention_rate': float(retained_mask.mean()),
                'mannwhitneyu_p': p_diff,
                'cliffs_delta': delta
            }
        
        results[model_key] = scen_store
    
    return results


def sample_dirichlet_around(weights_original, perturbation=0.2, rng=None):
    """
    Muestrea pesos alrededor del vector original usando Dirichlet.
    
    Args:
        weights_original: dict con w_a, w_d, w_c, w_e, w_h
        perturbation: magnitud de variación (mayor = más disperso)
        rng: numpy random generator
    
    Returns:
        dict con nuevos pesos muestreados
    """
    if rng is None:
        rng = np.random.default_rng()
    
    keys = ['w_a', 'w_d', 'w_c', 'w_e', 'w_h']
    w0 = np.array([weights_original[k] for k in keys], dtype=float)
    
    conc = max(1e-3, 1.0 / (perturbation ** 2))
    alpha = np.clip(w0 * conc, 1e-3, None)
    w = rng.dirichlet(alpha)
    
    return {k: v for k, v in zip(keys, w)}


def topk_jaccard(a, b, k_ratio=0.10):
    """
    Similitud Jaccard entre top-k elementos de dos arrays.
    
    k_ratio: fracción de top elementos a considerar
    """
    k = max(1, int(len(a) * k_ratio))
    idx_a = np.argpartition(-a, k-1)[:k]
    idx_b = np.argpartition(-b, k-1)[:k]
    set_a, set_b = set(idx_a.tolist()), set(idx_b.tolist())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else np.nan


def component_alignment(irct_vals, comps, k_ratio=0.10):
    """
    Calcula correlaciones Spearman y Jaccard Top-k entre IRCT y componentes.
    
    Args:
        irct_vals: valores IRCT
        comps: dict con componentes {A, S_D, S_C, S_E, S_H2}
        k_ratio: fracción de top elementos
    
    Returns:
        dict con estadísticas de alineación por componente
    """
    stats = {}
    k = max(1, int(len(irct_vals) * k_ratio))
    irct_top = set(np.argpartition(-irct_vals, k-1)[:k])
    
    for key in ['A', 'S_D', 'S_C', 'S_E', 'S_H2']:
        rho, _ = spearmanr(irct_vals, comps[key])
        comp_top = set(np.argpartition(-comps[key], k-1)[:k])
        jac = len(irct_top & comp_top) / len(irct_top | comp_top) if len(irct_top | comp_top) > 0 else np.nan
        stats[key] = {'spearman': rho, 'topk_jaccard': jac}
    
    return stats


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
    """
    Precomputa componentes IRCT sin recalcular forward del modelo.
    
    Retorna dict {A, S_D, S_C, S_E, S_H2}
    """
    A, _ = compute_reconstruction_anomaly(model, X_future_orig, X_future_norm, device)
    S_D, _ = compute_latent_displacement(z_future_scaled, centroids_base, labels_base_subset)
    S_C, _ = compute_cluster_stability_softmax(z_future_scaled, centroids_base, labels_base_subset)
    S_E, _ = compute_cluster_expansion(z_base_scaled_subset, z_future_scaled, centroids_base, labels_base_subset)
    
    if (h2_base is not None) and (h2_future is not None):
        S_H2, _ = compute_h2_stability(h2_base, h2_future)
    else:
        S_H2 = np.ones_like(A)
    
    return {'A': A, 'S_D': S_D, 'S_C': S_C, 'S_E': S_E, 'S_H2': S_H2}


def validate_robustness(irct_results_dict, comps_dict, clustering_results_dict, 
                       model_order, n_perturbations=50, perturbation_magnitude=0.20, seed=42):
    """
    VALIDACIÓN 4: Robustez del índice (sensibilidad a perturbaciones de pesos).
    
    Args:
        irct_results_dict: dict con IRCT por modelo/escenario
        comps_dict: dict con componentes precomputados
        clustering_results_dict: dict con resultados clustering
        model_order: lista de claves de modelos
        n_perturbations: número de perturbaciones
        perturbation_magnitude: magnitud de variación (±%)
        seed: seed para reproducibilidad
    
    Returns:
        dict con estadísticas de robustez por modelo/escenario
    """
    results = {}
    rng_master = np.random.default_rng(seed)
    
    for model_key in model_order:
        results[model_key] = {}
        
        for scenario_name in ['SSP245', 'SSP370', 'SSP585']:
            irct_original = irct_results_dict[model_key][scenario_name]['IRCT']
            weights_original = irct_results_dict[model_key][scenario_name]['weights']
            
            # Obtener componentes precomputados
            if comps_dict and model_key in comps_dict and scenario_name in comps_dict[model_key]:
                comps = comps_dict[model_key][scenario_name]
            else:
                # Si no hay precomputo, recombinar desde resultados
                comps = {
                    'A': irct_results_dict[model_key][scenario_name]['A'],
                    'S_D': irct_results_dict[model_key][scenario_name]['S_D'],
                    'S_C': irct_results_dict[model_key][scenario_name]['S_C'],
                    'S_E': irct_results_dict[model_key][scenario_name]['S_E'],
                    'S_H2': irct_results_dict[model_key][scenario_name]['S_H2']
                }
            
            # Diagnóstico de alineación
            align_stats = component_alignment(irct_original, comps, k_ratio=0.10)
            
            # Perturbaciones
            rhos, taus, jaccs, weights_used = [], [], [], []
            
            for i in range(n_perturbations):
                rng = np.random.default_rng(rng_master.integers(0, 2**32-1))
                weights_perturbed = sample_dirichlet_around(
                    weights_original,
                    perturbation=perturbation_magnitude,
                    rng=rng
                )
                
                # Recombinar con nueva ponderación
                IRCT_p = (
                    (comps['A'] ** weights_perturbed['w_a']) *
                    (comps['S_D'] ** weights_perturbed['w_d']) *
                    (comps['S_C'] ** weights_perturbed['w_c']) *
                    (comps['S_E'] ** weights_perturbed['w_e']) *
                    (comps['S_H2'] ** weights_perturbed['w_h'])
                )
                
                rho, _ = spearmanr(irct_original, IRCT_p)
                tau, _ = kendalltau(irct_original, IRCT_p)
                jac = topk_jaccard(irct_original, IRCT_p, k_ratio=0.10)
                
                rhos.append(rho)
                taus.append(tau)
                jaccs.append(jac)
                weights_used.append(weights_perturbed)
            
            rhos = np.array(rhos)
            taus = np.array(taus)
            jaccs = np.array(jaccs)
            
            frac_all = np.nanmean((rhos > 0.90) & (taus > 0.80) & (jaccs > 0.70))
            q_rho = np.nanpercentile(rhos, [10, 50, 90])
            q_tau = np.nanpercentile(taus, [10, 50, 90])
            q_jac = np.nanpercentile(jaccs, [10, 50, 90])
            
            worst_idx = int(np.nanargmin(rhos))
            worst_w = weights_used[worst_idx]
            
            is_robust = (np.nanmean(rhos) > 0.90) and (np.nanmean(taus) > 0.80) and (np.nanmean(jaccs) > 0.70)
            
            results[model_key][scenario_name] = {
                'n_perturbations': n_perturbations,
                'perturbation_magnitude': perturbation_magnitude,
                'component_alignment': align_stats,
                'spearman_mean': float(np.nanmean(rhos)),
                'spearman_std': float(np.nanstd(rhos)),
                'spearman_min': float(np.nanmin(rhos)),
                'spearman_max': float(np.nanmax(rhos)),
                'spearman_p10': float(q_rho[0]),
                'spearman_p50': float(q_rho[1]),
                'spearman_p90': float(q_rho[2]),
                'kendall_tau_mean': float(np.nanmean(taus)),
                'kendall_tau_std': float(np.nanstd(taus)),
                'kendall_tau_p10': float(q_tau[0]),
                'kendall_tau_p50': float(q_tau[1]),
                'kendall_tau_p90': float(q_tau[2]),
                'top10_jaccard_mean': float(np.nanmean(jaccs)),
                'top10_jaccard_std': float(np.nanstd(jaccs)),
                'top10_jaccard_p10': float(q_jac[0]),
                'top10_jaccard_p50': float(q_jac[1]),
                'top10_jaccard_p90': float(q_jac[2]),
                'robust_share_all_thresholds': float(frac_all),
                'is_robust': bool(is_robust),
                'weights_baseline': weights_original,
                'worst_case_rho': float(rhos[worst_idx]),
                'worst_case_tau': float(taus[worst_idx]),
                'worst_case_jaccard': float(jaccs[worst_idx]),
                'worst_case_weights': worst_w,
            }
    
    return results
