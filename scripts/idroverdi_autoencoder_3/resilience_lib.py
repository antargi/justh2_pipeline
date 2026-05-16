"""
Módulo de funciones reutilizables para cálculo del Índice de Resiliencia Climático-Territorial (IRCT)

Este módulo contiene todas las funciones necesarias para:
1. Calcular componentes individuales del IRCT
2. Calcular el IRCT pixel-wise
3. Agregar IRCT por clúster
4. Visualizar resultados espaciales
5. Validar el índice
6. Análisis de sensibilidad de pesos

Uso típico:
    from resilience_lib import compute_IRCT_pixel_wise, aggregate_IRCT_by_cluster
    from resilience_lib import WEIGHT_SCHEMES, sensitivity_analysis_univariate
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import softmax
from scipy.stats import percentileofscore, rankdata, spearmanr
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# ESQUEMAS DE PONDERACIÓN DEL IRCT CON JUSTIFICACIÓN TEÓRICA
# ============================================================================

WEIGHT_SCHEMES = {
    'theoretical': {
        'name': 'Teórico (basado en literatura de resiliencia)',
        'weights': {'w_a': 0.15, 'w_d': 0.30, 'w_c': 0.25, 'w_e': 0.20, 'w_h': 0.10},
        'rationale': """
        Basado en marco conceptual de resiliencia socio-ecológica:
        
        - S_D (0.30): PERSISTENCIA - Capacidad de mantener estado característico
                      ante perturbación (Holling 1973, "Engineering vs Ecological Resilience")
                      Métrica: distancia en espacio latente z_future → z_base
        
        - S_C (0.25): IDENTIDAD DE RÉGIMEN - Retención de pertenencia a régimen territorial
                      (Walker et al. 2004, "Resilience, Adaptability and Transformability")
                      Métrica: probabilidad softmax de mantener cluster original
        
        - S_E (0.20): COHESIÓN INTERNA - Variabilidad/dispersión del sistema
                      Contracción = mayor cohesión (Scheffer 2009, "Critical Transitions")
                      Métrica: ratio de expansión σ_future / σ_base
        
        - A (0.15):   REPRESENTABILIDAD - Patrón climático dentro del dominio conocido
                      Métrica: error de reconstrucción del autoencoder
        
        - S_H₂ (0.10): FUNCIÓN TERRITORIAL - Provisión de servicios energéticos
                      (Folke 2006, "Resilience: The emergence of a perspective")
                      Métrica: ratio de producción H₂_future / H₂_base
        
        Criterio de asignación:
        1. S_D + S_C (0.55) capturan ~55% porque son métricas DIRECTAS de 
           "mantener estructura ante perturbación" (definición core de resiliencia)
        2. S_E + A (0.35) son métricas INDIRECTAS de estabilidad interna
        3. S_H₂ (0.10) es contextual (solo válido para valle de hidrógeno)
        
        Referencias:
        - Holling, C.S. (1973). Resilience and stability of ecological systems. 
          Annual Review of Ecology and Systematics, 4, 1-23.
        - Walker, B., et al. (2004). Resilience, adaptability and transformability 
          in social-ecological systems. Ecology and Society, 9(2), 5.
        - Folke, C. (2006). Resilience: The emergence of a perspective for 
          social-ecological systems analyses. Global Environmental Change, 16, 253-267.
        - Scheffer, M., et al. (2009). Early-warning signals for critical transitions. 
          Nature, 461, 53-59.
        """,
        'color': '#2E86AB'  # Azul oscuro para gráficos
    },
    
    'uniform': {
        'name': 'Uniforme (sin sesgo a priori)',
        'weights': {'w_a': 0.20, 'w_d': 0.20, 'w_c': 0.20, 'w_e': 0.20, 'w_h': 0.20},
        'rationale': """
        Ponderación igual para todos los componentes.
        
        Útil como baseline para comparación y validación de sensibilidad.
        Asume que no tenemos conocimiento previo sobre la importancia relativa
        de cada dimensión de resiliencia.
        """,
        'color': '#A23B72'  # Morado para gráficos
    },
    
    'latent_focused': {
        'name': 'Focalizado en espacio latente',
        'weights': {'w_a': 0.05, 'w_d': 0.35, 'w_c': 0.30, 'w_e': 0.25, 'w_h': 0.05},
        'rationale': """
        Prioriza métricas emergentes del espacio latente sobre otras.
        
        - S_D + S_C + S_E (0.90): métricas derivadas del espacio latente z
        - A + S_H₂ (0.10): métricas auxiliares
        
        Justificación: El espacio latente captura la estructura multivariada
        no-lineal del sistema climático de forma más completa que métricas
        individuales como error de reconstrucción o producción de H₂.
        """,
        'color': '#F18F01'  # Naranja para gráficos
    },
    
    'energy_focused': {
        'name': 'Focalizado en función energética',
        'weights': {'w_a': 0.10, 'w_d': 0.20, 'w_c': 0.20, 'w_e': 0.15, 'w_h': 0.35},
        'rationale': """
        Prioriza la provisión de servicios energéticos (H₂).
        
        Útil para análisis orientados a planificación energética donde
        la capacidad de mantener producción de hidrógeno es crítica.
        
        Solo válido cuando se dispone de datos de producción H₂.
        """,
        'color': '#C73E1D'  # Rojo para gráficos
    }
}


def get_weight_scheme(scheme='theoretical', verbose=True):
    """
    Retorna esquema de pesos del IRCT con justificación teórica.
    
    Parameters
    ----------
    scheme : str, default='theoretical'
        Nombre del esquema: 'theoretical', 'uniform', 'latent_focused', 'energy_focused'
    verbose : bool, default=True
        Si True, imprime tabla justificativa
        
    Returns
    -------
    dict
        Diccionario con pesos {'w_a': float, 'w_d': float, ...}
        
    Example
    -------
    >>> weights = get_weight_scheme('theoretical', verbose=True)
    >>> print(weights)
    {'w_a': 0.15, 'w_d': 0.30, 'w_c': 0.25, 'w_e': 0.20, 'w_h': 0.10}
    """
    if scheme not in WEIGHT_SCHEMES:
        available = ', '.join(WEIGHT_SCHEMES.keys())
        raise ValueError(f"Esquema '{scheme}' no válido. Opciones: {available}")
    
    scheme_info = WEIGHT_SCHEMES[scheme]
    
    if verbose:
        print(f"\nESQUEMA DE PESOS: {scheme_info['name']}")
        print("="*80)
        print(f"\nPesos:")
        for comp, val in scheme_info['weights'].items():
            comp_names = {
                'w_a': 'Anomalía Reconstrucción',
                'w_d': 'Desplazamiento Latente',
                'w_c': 'Estabilidad de Cluster',
                'w_e': 'Expansión de Cluster',
                'w_h': 'Estabilidad H₂'
            }
            print(f"  {comp_names.get(comp, comp):30s} ({comp}): {val:.2f}")
        
        print(f"\nJustificación:")
        print(scheme_info['rationale'])
        print("="*80)
    
    return scheme_info['weights'].copy()


def print_theoretical_framework_table():
    """
    Imprime tabla de traducción entre conceptos teóricos de resiliencia
    y métricas operacionales del espacio latente.
    
    Útil para documentación de tesis y artículos.
    """
    print("\n" + "="*100)
    print("MARCO TEÓRICO: Traducción Conceptual entre Resiliencia Clásica y Espacio Latente")
    print("="*100)
    
    framework = [
        {
            'concepto': 'PERSISTENCIA',
            'definicion': 'Mantener estado característico\nante perturbación',
            'metrica': 'S_D = Desplazamiento Latente',
            'ecuacion': '1 / (1 + ||z_future - c_k||)',
            'interpretacion': 'Baja deriva en espacio latente\n= sistema mantiene estado base',
            'ref': 'Holling 1973'
        },
        {
            'concepto': 'IDENTIDAD DE RÉGIMEN',
            'definicion': 'Permanecer en el mismo\nrégimen territorial',
            'metrica': 'S_C = Estabilidad de Cluster',
            'ecuacion': 'P_softmax(cluster_k | z_future)',
            'interpretacion': 'Alta probabilidad de retener\ncluster = régimen estable',
            'ref': 'Walker 2004'
        },
        {
            'concepto': 'COHESIÓN INTERNA',
            'definicion': 'Dispersión/variabilidad\ndel sistema',
            'metrica': 'S_E = Estabilidad de Expansión',
            'ecuacion': '1 / (σ_future/σ_base + ε)',
            'interpretacion': 'Contracción del cluster\n= mayor cohesión interna',
            'ref': 'Scheffer 2009'
        },
        {
            'concepto': 'REPRESENTABILIDAD',
            'definicion': 'Patrón climático dentro\nde dominio conocido',
            'metrica': 'A = Anomalía de Reconstrucción',
            'ecuacion': '1 - percentil(||X - X_hat||²)',
            'interpretacion': 'Bajo error de reconstrucción\n= patrón climático válido',
            'ref': '-'
        },
        {
            'concepto': 'PROVISIÓN DE SERVICIOS',
            'definicion': 'Función territorial\n(energética)',
            'metrica': 'S_H₂ = Estabilidad Energética',
            'ecuacion': 'percentil(H₂_future / H₂_base)',
            'interpretacion': 'Mantener/aumentar producción\n= función resiliente',
            'ref': 'Folke 2006'
        }
    ]
    
    # Header
    print(f"\n{'CONCEPTO TEÓRICO':<25} | {'OPERACIONALIZACIÓN':<30} | {'ECUACIÓN':<35} | {'INTERPRETACIÓN':<40} | {'REF':<15}")
    print("-"*100)
    
    # Rows
    for item in framework:
        concepto_lines = item['concepto'].split('\n')
        metrica_lines = item['metrica'].split('\n')
        ecuacion_lines = item['ecuacion'].split('\n')
        interp_lines = item['interpretacion'].split('\n')
        
        max_lines = max(len(concepto_lines), len(metrica_lines), len(ecuacion_lines), len(interp_lines))
        
        for i in range(max_lines):
            concepto = concepto_lines[i] if i < len(concepto_lines) else ''
            metrica = metrica_lines[i] if i < len(metrica_lines) else ''
            ecuacion = ecuacion_lines[i] if i < len(ecuacion_lines) else ''
            interp = interp_lines[i] if i < len(interp_lines) else ''
            ref = item['ref'] if i == 0 else ''
            
            print(f"{concepto:<25} | {metrica:<30} | {ecuacion:<35} | {interp:<40} | {ref:<15}")
        
        print("-"*100)
    
    print("\nVENTAJAS DEL ESPACIO LATENTE COMO PROXY DE RESILIENCIA:")
    print("  1. Reducción de dimensionalidad semántica: z comprime 20+ variables climáticas en 5-10 dimensiones")
    print("  2. Captura relaciones no-lineales: aprende patrones multivariados, no valores absolutos")
    print("  3. Invarianzas aprendidas: z codifica 'modos dominantes' del sistema climático")
    print("  4. Interpretación como espacio de estados: clusters = regímenes alternativos (sensu Scheffer 2001)")
    print("  5. Ventaja sobre índices univariados: captura interacciones entre múltiples drivers climáticos")
    print("="*100 + "\n")


# ============================================================================
# FUNCIONES DE COMPONENTES INDIVIDUALES DEL IRCT
# ============================================================================

def percentile_rank(arr):
    """
    Convierte un array de valores a percentiles [0, 100].
    
    Vectorizado para mayor eficiencia.
    
    Parameters
    ----------
    arr : np.ndarray
        Array de valores a convertir
        
    Returns
    -------
    np.ndarray
        Percentiles correspondientes a cada valor
    """
    arr = np.asarray(arr)
    sorted_arr = np.sort(arr)
    ranks = np.searchsorted(sorted_arr, arr, side='right')
    percentiles = 100.0 * ranks / len(arr)
    return percentiles


def compute_cluster_stability_softmax(z_scaled, centroids_base, labels_base, tau=1.0):
    """
    Calcula estabilidad de pertenencia a clúster usando softmax sobre distancias.
    
    S_C = probabilidad_softmax del clúster asignado
    
    Parameters
    ----------
    z_scaled : np.ndarray (n_samples, latent_dim)
        Espacio latente estandarizado
    centroids_base : np.ndarray (n_clusters, latent_dim)
        Centroides de los clústers BASE
    labels_base : np.ndarray (n_samples,)
        Etiquetas de clúster asignadas
    tau : float, default=1.0
        Temperatura para softmax (menor = más peaky)
        
    Returns
    -------
    np.ndarray (n_samples,)
        Estabilidad de pertenencia [0, 1] para cada píxel
    """
    z_scaled = np.asarray(z_scaled)
    centroids_base = np.asarray(centroids_base)
    labels_base = np.asarray(labels_base)
    
    # Calcular distancias euclidianas a todos los centroides
    dists = np.linalg.norm(z_scaled[:, None, :] - centroids_base[None, :, :], axis=2)
    
    # Convertir a similitudes (negativo de distancia escalado por tau)
    logits = -dists / tau
    
    # Aplicar softmax
    probs = softmax(logits, axis=1)
    
    # Extraer probabilidad del clúster asignado
    stability = probs[np.arange(len(labels_base)), labels_base]
    
    return stability


def compute_latent_displacement(z_scaled, centroids_base, labels_base, inv_covs_base=None, *, return_details=False):
    """
    Calcula desplazamiento latente normalizado desde el centroide BASE.
    
    S_D = 1 / (1 + d_normalized)
    
    Si inv_covs_base está disponible, usa distancia de Mahalanobis.
    Si no, usa distancia euclidiana normalizada.
    
    Parameters
    ----------
    z_scaled : np.ndarray (n_samples, latent_dim)
        Espacio latente estandarizado
    centroids_base : np.ndarray (n_clusters, latent_dim)
        Centroides de los clústers BASE
    labels_base : np.ndarray (n_samples,)
        Etiquetas de clúster asignadas
    inv_covs_base : dict or None
        Diccionario {cluster_id: inv_cov_matrix} para Mahalanobis
        
    Returns
    -------
    np.ndarray (n_samples,)
        Estabilidad de desplazamiento [0, 1] para cada píxel
    """
    z_scaled = np.asarray(z_scaled)
    centroids_base = np.asarray(centroids_base)
    labels_base = np.asarray(labels_base)
    
    displacements = np.zeros(len(z_scaled))
    
    if inv_covs_base is not None:
        # Distancia de Mahalanobis por clúster
        for i, (point, label) in enumerate(zip(z_scaled, labels_base)):
            centroid = centroids_base[label]
            diff = point - centroid
            inv_cov = inv_covs_base.get(label, np.eye(len(diff)))
            dist = np.sqrt(diff @ inv_cov @ diff)
            displacements[i] = dist
    else:
        # Distancia euclidiana simple
        for i, (point, label) in enumerate(zip(z_scaled, labels_base)):
            centroid = centroids_base[label]
            dist = np.linalg.norm(point - centroid)
            displacements[i] = dist
    
    # Convertir a estabilidad: S_D = 1 / (1 + d)
    stability = 1.0 / (1.0 + displacements)
    
    if return_details:
        return stability, displacements
    return stability


def compute_cluster_expansion(z_base_scaled, z_future_scaled, centroids_base, labels_base, 
                              epsilon=1e-8, p99_clip=False):
    """
    Calcula expansión del clúster PÍXEL-A-PÍXEL usando cambio en distancia.
    
    Para cada píxel i en clúster k:
        dist_base_i = ||z_base_i - c_k||
        dist_future_i = ||z_future_i - c_k||
        ratio_i = dist_future_i / (dist_base_i + epsilon)
    
    Esta aproximación permite capturar variación píxel-individual en lugar de
    solo agregar a nivel clúster.
    
    Parameters
    ----------
    z_base_scaled : np.ndarray (n_samples, latent_dim)
        Espacio latente BASE estandarizado
    z_future_scaled : np.ndarray (n_samples, latent_dim)
        Espacio latente FUTURE estandarizado
    centroids_base : np.ndarray (n_clusters, latent_dim)
        Centroides de los clústers BASE
    labels_base : np.ndarray (n_samples,)
        Etiquetas de clúster BASE
    epsilon : float, default=1e-8
        Pequeño valor para evitar división por cero
    p99_clip : bool, default=False
        Si True, clipea outliers al percentil 99
        
    Returns
    -------
    np.ndarray (n_samples,)
        Razón de expansión para cada píxel individualmente
    """
    z_base_scaled = np.asarray(z_base_scaled)
    z_future_scaled = np.asarray(z_future_scaled)
    centroids_base = np.asarray(centroids_base)
    labels_base = np.asarray(labels_base)
    
    n_samples = len(z_base_scaled)
    expansion_ratios = np.zeros(n_samples)
    
    # Calcular ratio PÍXEL-A-PÍXEL
    for i in range(n_samples):
        k = labels_base[i]
        centroid = centroids_base[k]
        
        # Distancia individual de cada píxel al centroide del cluster
        dist_base_i = np.linalg.norm(z_base_scaled[i] - centroid)
        dist_future_i = np.linalg.norm(z_future_scaled[i] - centroid)
        
        # Ratio de cambio individual
        expansion_ratios[i] = dist_future_i / (dist_base_i + epsilon)
    
    if p99_clip:
        p99 = np.percentile(expansion_ratios, 99)
        expansion_ratios = np.clip(expansion_ratios, 0, p99)
    
    # DEBUG
    unique_vals = len(np.unique(expansion_ratios))
    print(f"[DEBUG compute_cluster_expansion] Retornando {unique_vals} valores únicos de {n_samples} píxeles")
    
    return expansion_ratios


def compute_h2_stability(h2_base, h2_future, epsilon=1e-8, p99_clip=True):
    """
    Calcula la estabilidad de producción de H₂ como la posición percentil
    del ratio FUTURE / BASE.

    Mayor producción futura relativa implica mayor resiliencia.

    S_H2 = percentile_rank(h2_future / h2_base)

    Parameters
    ----------
    h2_base : np.ndarray (n_samples,)
        Producción de H₂ en escenario BASE
    h2_future : np.ndarray (n_samples,)
        Producción de H₂ en escenario FUTURE
    epsilon : float, default=1e-8
        Valor pequeño para evitar división por cero
    p99_clip : bool, default=True
        Si True, clipea outliers al percentil 99

    Returns
    -------
    tuple
        (S_H2, h2_ratios)
        S_H2 : np.ndarray (n_samples,)
            Estabilidad normalizada en [0, 1]
        h2_ratios : np.ndarray (n_samples,)
            Ratios FUTURE / BASE
    """
    h2_base = np.asarray(h2_base)
    h2_future = np.asarray(h2_future)

    # Ratio de cambio FUTURE / BASE
    h2_ratios = h2_future / (h2_base + epsilon)

    if p99_clip:
        p99 = np.percentile(h2_ratios, 99)
        h2_ratios = np.clip(h2_ratios, 0, p99)
    else:
        h2_ratios = np.clip(h2_ratios, 0, None)

    # Percentile rank normalizado [0,1]
    perc_ranks = percentile_rank(h2_ratios) / 100.0
    S_H2 = perc_ranks

    return S_H2, h2_ratios


def compute_reconstruction_anomaly(
    model,
    X_orig,
    X_normalized,
    device='cpu',
    use_normalized=True,
    inverse_transform=None,
    reduce='mse',
    output='stability'
):
    """
    Calcula anomalía de reconstrucción del modelo AE/VAE.
    
    A = ||X - X_recon||² (por píxel o agregado)
    
    Parameters
    ----------
    model : nn.Module
        Modelo AE o VAE entrenado
    X_orig : np.ndarray (n_samples, n_features)
        Datos originales sin normalizar
    X_normalized : np.ndarray (n_samples, n_features)
        Datos normalizados (usados si use_normalized=True)
    device : str, default='cpu'
        Device para PyTorch
    use_normalized : bool, default=True
        Si True, reconstruye desde datos normalizados
    inverse_transform : callable or None
        Función para desnormalizar (scaler.inverse_transform)
    reduce : str, default='mse'
        Métrica de reducción: 'mse', 'mae', 'rmse'
    output : str, default='stability'
        'stability' retorna estabilidad (1 - percentil(error))
        'error' retorna error raw
        'percentile' retorna percentil del error
        'both' retorna (estabilidad, errores)
        
    Returns
    -------
    np.ndarray (n_samples,)
        Anomalía o estabilidad de reconstrucción por píxel
    """
    # DEBUG: Log de parámetros recibidos
    print(f"\n[DEBUG compute_reconstruction_anomaly]")
    print(f"  model type: {type(model).__name__}")
    print(f"  X_orig shape: {X_orig.shape if hasattr(X_orig, 'shape') else type(X_orig)}")
    print(f"  X_normalized shape: {X_normalized.shape if hasattr(X_normalized, 'shape') else type(X_normalized)}")
    print(f"  device: {device}")
    print(f"  use_normalized: {use_normalized}")
    print(f"  reduce: {reduce}")
    print(f"  output: {output} (type: {type(output)})")
    
    model.eval()
    
    # Preparar datos
    X_input = X_normalized if use_normalized else X_orig
    X_tensor = torch.FloatTensor(X_input).to(device)
    
    with torch.no_grad():
        # Forward pass
        if hasattr(model, 'reparam'):  # VAE
            mu, logvar = model.encode(X_tensor)
            z = model.reparam(mu, logvar)
            X_recon = model.dec(z)
        else:  # AE
            # AE retorna (x_hat, z), necesitamos solo x_hat
            model_output = model(X_tensor)
            if isinstance(model_output, tuple):
                X_recon = model_output[0]
            else:
                X_recon = model_output
        
        X_recon_np = X_recon.cpu().numpy()
    
    # Desnormalizar si es necesario
    if use_normalized and inverse_transform is not None:
        X_recon_np = inverse_transform(X_recon_np)
        X_compare = X_orig
    else:
        X_compare = X_input
    
    # Calcular error por píxel
    if reduce == 'mse':
        errors = np.mean((X_compare - X_recon_np) ** 2, axis=1)
    elif reduce == 'mae':
        errors = np.mean(np.abs(X_compare - X_recon_np), axis=1)
    elif reduce == 'rmse':
        errors = np.sqrt(np.mean((X_compare - X_recon_np) ** 2, axis=1))
    else:
        raise ValueError(f"reduce='{reduce}' no válido. Opciones: 'mse', 'mae', 'rmse'")
    
    # Retornar según output solicitado
    if output == 'error':
        return errors
    elif output == 'percentile':
        return percentile_rank(errors)
    elif output == 'stability':
        percentiles = percentile_rank(errors)
        return 1.0 - (percentiles / 100.0)
    elif output == 'both':
        percentiles = percentile_rank(errors)
        stability = 1.0 - (percentiles / 100.0)
        return stability, errors
    else:
        raise ValueError(f"output='{output}' no válido. Opciones: 'error', 'percentile', 'stability', 'both'")


# ============================================================================
# FUNCIÓN PRINCIPAL: CÁLCULO DEL IRCT PIXEL-WISE
# ============================================================================

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
    inv_covs_base=None,
    recon_use_normalized=True,
    inverse_transform=None,
    softmax_tau=1.0,
    expansion_p99_clip=False,
    h2_p99_clip=True,
    eps=1e-8
):
    """
    Calcula el Índice de Resiliencia Climático-Territorial (IRCT) píxel a píxel.
    
    Ecuación principal:
        IRCT_i = (A_i)^{w_a} * (S_D,i)^{w_d} * (S_C,i)^{w_c} * (S_E,i)^{w_e} * (S_H2,i)^{w_h}
    
    Donde S_E se calcula como:
        ratio_i = ||z_future_i - c_k|| / (||z_base_i - c_k|| + ε)
        S_{E,i} = 1 / (ratio_i + ε)    [luego normalizado a [0,1]]
    
    Componentes:
    - A: Anomalía de reconstrucción (estabilidad de codificación)
    - S_D: Estabilidad de desplazamiento latente (píxel-a-píxel)
    - S_C: Estabilidad de pertenencia a clúster (softmax, píxel-a-píxel)
    - S_E: Estabilidad de expansión del clúster (píxel-a-píxel, distancia individual)
    - S_H2: Estabilidad de producción H₂ (opcional, píxel-a-píxel)
    
    Parameters
    ----------
    model : nn.Module
        Modelo AE o VAE entrenado
    X_base_orig : np.ndarray (n_samples, n_features)
        Datos BASE originales sin normalizar
    X_base_norm : np.ndarray (n_samples, n_features)
        Datos BASE normalizados
    X_future_orig : np.ndarray (n_samples, n_features)
        Datos FUTURE originales sin normalizar
    X_future_norm : np.ndarray (n_samples, n_features)
        Datos FUTURE normalizados
    z_base_scaled : np.ndarray (n_samples, latent_dim)
        Espacio latente BASE estandarizado
    z_future_scaled : np.ndarray (n_samples, latent_dim)
        Espacio latente FUTURE estandarizado
    centroids_base : np.ndarray (n_clusters, latent_dim)
        Centroides de los clústers BASE
    labels_base : np.ndarray (n_samples,)
        Etiquetas de clúster BASE
    h2_base : np.ndarray (n_samples,) or None
        Producción de H₂ en BASE
    h2_future : np.ndarray (n_samples,) or None
        Producción de H₂ en FUTURE
    weights : dict or None
        Pesos de los componentes: {'w_a', 'w_d', 'w_c', 'w_e', 'w_h'}
        Default: w_d=0.30, w_c=0.25, w_e=0.20, w_a=0.15, w_h=0.10
    device : str, default='cpu'
        Device para PyTorch
    inv_covs_base : dict or None
        Matrices de covarianza inversa por clúster para Mahalanobis
    recon_use_normalized : bool, default=True
        Si True, calcula reconstrucción desde datos normalizados
    inverse_transform : callable or None
        Función para desnormalizar (scaler.inverse_transform)
    softmax_tau : float, default=1.0
        Temperatura para softmax en S_C
    expansion_p99_clip : bool, default=False
        Si True, clipea outliers en S_E
    h2_p99_clip : bool, default=True
        Si True, clipea outliers en S_H2
    eps : float, default=1e-8
        Epsilon para evitar divisiones por cero
        
    Returns
    -------
    dict
        Diccionario con los componentes y el IRCT final:
        {
            'reconstruction_anomaly': np.ndarray,
            'latent_displacement': np.ndarray,
            'cluster_stability': np.ndarray,
            'cluster_expansion': np.ndarray,
            'h2_stability': np.ndarray or None,
            'IRCT': np.ndarray
        }
    """
    # Pesos por defecto
    if weights is None:
        weights = {
            'w_a': 0.15,  # Reconstrucción
            'w_d': 0.30,  # Desplazamiento latente
            'w_c': 0.25,  # Estabilidad de clúster
            'w_e': 0.20,  # Expansión
            'w_h': 0.10   # H₂
        }
    
    w_a = weights.get('w_a', 0.15)
    w_d = weights.get('w_d', 0.30)
    w_c = weights.get('w_c', 0.25)
    w_e = weights.get('w_e', 0.20)
    w_h = weights.get('w_h', 0.10)
    
    # Verificar si hay datos de H₂
    has_h2 = (h2_base is not None) and (h2_future is not None)
    
    # 1. Anomalía de reconstrucción (A)
    result_A = compute_reconstruction_anomaly(
        model=model,
        X_orig=X_future_orig,
        X_normalized=X_future_norm,
        device=device,
        use_normalized=recon_use_normalized,
        inverse_transform=inverse_transform,
        output='both'
    )
    # Manejo robusto: puede retornar tupla o valor simple
    if isinstance(result_A, tuple):
        A_future, recon_errors = result_A
    else:
        A_future = result_A
        recon_errors = np.zeros_like(A_future)
    
    # 2. Desplazamiento latente (S_D)
    result_S_D = compute_latent_displacement(
        z_future_scaled, centroids_base, labels_base,
        inv_covs_base=inv_covs_base,
        return_details=True
    )
    
    if isinstance(result_S_D, tuple):
        S_D, latent_distances = result_S_D
    else:
        S_D = result_S_D
        latent_distances = np.zeros_like(S_D)
    
    # 3. Estabilidad de clúster softmax (S_C)
    S_C = compute_cluster_stability_softmax(
        z_future_scaled, centroids_base, labels_base,
        tau=softmax_tau
    )
    
    # 4. Expansión del clúster (S_E)
    expansion_raw = compute_cluster_expansion(
        z_base_scaled, z_future_scaled, centroids_base, labels_base,
        epsilon=eps, p99_clip=expansion_p99_clip
    )
    # La función retorna ratios (σ_future / σ_base)
    # Aplicamos: S_E = 1 / (ratio + epsilon) luego normalizamos a [0,1]
    if isinstance(expansion_raw, tuple):
        expansion_ratios = np.asarray(expansion_raw[1]) if len(expansion_raw) > 1 else np.asarray(expansion_raw[0])
    else:
        expansion_ratios = np.asarray(expansion_raw)
    
    # Paso 1: Aplicar ecuación LaTeX: S_E_raw = 1 / (σ_future/σ_base + ε)
    S_E_raw = 1.0 / (expansion_ratios + eps)
    
    # Paso 2: Normalizar a [0, 1] usando percentile_rank
    # Así S_E es comparable con otros componentes (A, S_D, S_C) que también están en [0,1]
    # Interpretación: ratio=1 (sin cambio) → S_E ≈ 50% (neutral)
    #                ratio<1 (contracción) → S_E > 50% (resiliente)
    #                ratio>1 (expansión) → S_E < 50% (vulnerable)
    ranks = rankdata(S_E_raw)
    S_E = (ranks - 1.0) / (len(S_E_raw) - 1.0) if len(S_E_raw) > 1 else np.ones_like(S_E_raw)
    
    # 5. Estabilidad H₂ (S_H2) - opcional
    if has_h2:
        result_S_H2 = compute_h2_stability(h2_base, h2_future, epsilon=eps, p99_clip=h2_p99_clip)
        if isinstance(result_S_H2, tuple):
            S_H2, h2_ratios = result_S_H2
        else:
            S_H2 = result_S_H2
            h2_ratios = np.asarray(h2_future) / (np.asarray(h2_base) + eps)
    else:
        S_H2 = None
        h2_ratios = None
        w_h = 0.0  # Sin H₂, su peso es 0
    
    # Renormalizar pesos si no hay H₂
    if not has_h2:
        total_weight = w_a + w_d + w_c + w_e
        w_a /= total_weight
        w_d /= total_weight
        w_c /= total_weight
        w_e /= total_weight
    
    # 6. CRÍTICO: Normalizar TODOS los componentes a [0, 1] ANTES de calcular IRCT
    # Esto es esencial para garantizar que el producto ponderado esté acotado
    A_future = np.clip(A_future, 0.0, 1.0)
    S_D = np.clip(S_D, 0.0, 1.0)
    S_C = np.clip(S_C, 0.0, 1.0)
    S_E = np.clip(S_E, 0.0, 1.0)
    if S_H2 is not None:
        S_H2 = np.clip(S_H2, 0.0, 1.0)
    
    # 7. Calcular IRCT como producto geométrico ponderado
    IRCT = (A_future ** w_a) * (S_D ** w_d) * (S_C ** w_c) * (S_E ** w_e)
    
    if has_h2:
        IRCT *= (S_H2 ** w_h)
    
    # Normalizar IRCT a [0, 1] (redundante pero seguro)
    IRCT = np.clip(IRCT, 0.0, 1.0)

    # Guardar pesos efectivos usados (ya normalizados si aplica)
    weights_used = {
        'w_a': w_a,
        'w_d': w_d,
        'w_c': w_c,
        'w_e': w_e,
        'w_h': w_h
    }
    
    # Devolver con nombres detallados y alias cortos para compatibilidad
    return {
        'reconstruction_anomaly': A_future,
        'latent_displacement': S_D,
        'cluster_stability': S_C,
        'cluster_expansion': S_E,
        'h2_stability': S_H2,
        'IRCT': IRCT,
        'weights': weights_used,
        # Detalles adicionales
        'reconstruction_errors': recon_errors,
        'latent_distances': latent_distances,
        # Aliases usados en otros notebooks
        'A': A_future,
        'S_D': S_D,
        'S_C': S_C,
        'S_E': S_E,
        'S_H2': S_H2,
        # Extras para agregaciones/plots
        'expansion_ratios': expansion_ratios,
        'h2_ratios': h2_ratios
    }


# ============================================================================
# FUNCIÓN WRAPPER: CÁLCULO AUTOMÁTICO DEL IRCT DESDE CLUSTERING_RESULTS
# ============================================================================

def compute_IRCT_from_clustering_results(
    model,
    clustering_results,
    X_base_orig,
    X_base_norm,
    X_future_orig,
    X_future_norm,
    scenario='T585',
    h2_base=None,
    h2_future=None,
    weights=None,
    device='cpu',
    recon_use_normalized=True,
    inverse_transform=None,
    softmax_tau=1.0,
    expansion_p99_clip=False,
    h2_p99_clip=True,
    eps=1e-8
):
    """
    Calcula el IRCT automáticamente desde los resultados de clustering.
    
    Esta es una función wrapper de alto nivel que extrae automáticamente
    todos los datos necesarios de CLUSTERING_RESULTS y llama a 
    compute_IRCT_pixel_wise.
    
    Parameters
    ----------
    model : nn.Module
        Modelo AE o VAE entrenado
    clustering_results : dict
        Diccionario retornado por cluster_and_measure_resilience()
        Debe contener:
        - centroids: centroides de clusters BASE
        - labels_B245/B370/B585: etiquetas BASE por escenario
        - z_B245_scaled/B370/B585: espacios latentes BASE escalados
        - z_T245_scaled/T370/T585: espacios latentes FUTURO escalados
        - inv_covs: matrices de covarianza inversa por cluster
    X_base_orig : np.ndarray (n_samples, n_features)
        Datos BASE originales sin normalizar (B245+B370+B585 concatenados)
    X_base_norm : np.ndarray (n_samples, n_features)
        Datos BASE normalizados
    X_future_orig : np.ndarray (n_samples, n_features)
        Datos FUTURE originales sin normalizar para el escenario elegido
    X_future_norm : np.ndarray (n_samples, n_features)
        Datos FUTURE normalizados para el escenario elegido
    scenario : str, default='T585'
        Escenario futuro a analizar: 'T245', 'T370', 'T585'
    h2_base : np.ndarray (n_samples,) or None
        Producción de H₂ en BASE
    h2_future : np.ndarray (n_samples,) or None
        Producción de H₂ en FUTURE
    weights : dict or None
        Pesos de los componentes del IRCT
    device : str, default='cpu'
        Device para PyTorch
    recon_use_normalized : bool, default=True
        Si True, calcula reconstrucción desde datos normalizados
    inverse_transform : callable or None
        Función para desnormalizar (scaler.inverse_transform)
    softmax_tau : float, default=1.0
        Temperatura para softmax en estabilidad de cluster
    expansion_p99_clip : bool, default=False
        Si True, clipea outliers en expansión de cluster
    h2_p99_clip : bool, default=True
        Si True, clipea outliers en estabilidad de H₂
    eps : float, default=1e-8
        Epsilon para evitar divisiones por cero
        
    Returns
    -------
    dict
        Diccionario con los componentes del IRCT y el índice final
        
    Example
    -------
    >>> # Después de ejecutar clustering
    >>> results_ae = cluster_and_measure_resilience(LATENTS["AE"], ...)
    >>> 
    >>> # Calcular IRCT automáticamente para SSP585
    >>> irct_ae_585 = compute_IRCT_from_clustering_results(
    ...     model=MODELS["AE"],
    ...     clustering_results=results_ae,
    ...     X_base_orig=X_BASE,
    ...     X_base_norm=X_base_normalized,
    ...     X_future_orig=X585_orig,
    ...     X_future_norm=X585_norm,
    ...     scenario='T585'
    ... )
    >>> 
    >>> print(f"IRCT medio: {irct_ae_585['IRCT'].mean():.3f}")
    """
    # Validar escenario
    valid_scenarios = ['T245', 'T370', 'T585']
    if scenario not in valid_scenarios:
        raise ValueError(f"scenario debe ser uno de {valid_scenarios}, recibido: {scenario}")
    
    # Extraer datos de clustering_results según el escenario
    scenario_map = {
        'T245': ('z_B245_scaled', 'z_T245_scaled', 'labels_B245'),
        'T370': ('z_B370_scaled', 'z_T370_scaled', 'labels_B370'),
        'T585': ('z_B585_scaled', 'z_T585_scaled', 'labels_B585'),
    }
    
    z_base_key, z_future_key, labels_key = scenario_map[scenario]
    
    z_base_scaled = clustering_results[z_base_key]
    z_future_scaled = clustering_results[z_future_key]
    labels_base = clustering_results[labels_key]
    centroids_base = clustering_results['centroids']
    inv_covs_base = clustering_results['inv_covs']
    
    # Llamar a la función principal de cálculo del IRCT
    irct_results = compute_IRCT_pixel_wise(
        model=model,
        X_base_orig=X_base_orig,
        X_base_norm=X_base_norm,
        X_future_orig=X_future_orig,
        X_future_norm=X_future_norm,
        z_base_scaled=z_base_scaled,
        z_future_scaled=z_future_scaled,
        centroids_base=centroids_base,
        labels_base=labels_base,
        h2_base=h2_base,
        h2_future=h2_future,
        weights=weights,
        device=device,
        inv_covs_base=inv_covs_base,
        recon_use_normalized=recon_use_normalized,
        inverse_transform=inverse_transform,
        softmax_tau=softmax_tau,
        expansion_p99_clip=expansion_p99_clip,
        h2_p99_clip=h2_p99_clip,
        eps=eps
    )
    
    return irct_results


def compute_IRCT_all_scenarios(
    model,
    clustering_results,
    X_base_orig,
    X_base_norm,
    X245_orig,
    X245_norm,
    X370_orig,
    X370_norm,
    X585_orig,
    X585_norm,
    n_per_scenario,
    h2_base=None,
    h2_245=None,
    h2_370=None,
    h2_585=None,
    weights=None,
    device='cpu',
    recon_use_normalized=True,
    inverse_transform=None,
    softmax_tau=1.0,
    expansion_p99_clip=False,
    h2_p99_clip=True,
    eps=1e-8,
    verbose=True
):
    """
    Calcula el IRCT para los tres escenarios SSP (245, 370, 585) automáticamente.
    
    Esta función itera sobre los tres escenarios futuros y calcula el IRCT
    pixel-wise para cada uno, extrayendo automáticamente los segmentos
    correctos de los datos BASE.
    
    Parameters
    ----------
    model : nn.Module
        Modelo AE o VAE entrenado
    clustering_results : dict
        Diccionario retornado por cluster_and_measure_resilience()
    X_base_orig : np.ndarray (3*n_per_scenario, n_features)
        Datos BASE originales concatenados (B245+B370+B585)
    X_base_norm : np.ndarray (3*n_per_scenario, n_features)
        Datos BASE normalizados concatenados
    X245_orig : np.ndarray (n_per_scenario, n_features)
        Datos SSP245 originales
    X245_norm : np.ndarray (n_per_scenario, n_features)
        Datos SSP245 normalizados
    X370_orig : np.ndarray (n_per_scenario, n_features)
        Datos SSP370 originales
    X370_norm : np.ndarray (n_per_scenario, n_features)
        Datos SSP370 normalizados
    X585_orig : np.ndarray (n_per_scenario, n_features)
        Datos SSP585 originales
    X585_norm : np.ndarray (n_per_scenario, n_features)
        Datos SSP585 normalizados
    n_per_scenario : int
        Número de píxeles por escenario
    h2_base : np.ndarray (3*n_per_scenario,) or None
        Producción de H₂ en BASE concatenada
    h2_245 : np.ndarray (n_per_scenario,) or None
        Producción de H₂ en SSP245
    h2_370 : np.ndarray (n_per_scenario,) or None
        Producción de H₂ en SSP370
    h2_585 : np.ndarray (n_per_scenario,) or None
        Producción de H₂ en SSP585
    weights : dict or None
        Pesos de los componentes del IRCT
    device : str, default='cpu'
        Device para PyTorch
    recon_use_normalized : bool, default=True
        Si True, calcula reconstrucción desde datos normalizados
    inverse_transform : callable or None
        Función para desnormalizar
    softmax_tau : float, default=1.0
        Temperatura para softmax
    expansion_p99_clip : bool, default=False
        Si True, clipea outliers en expansión
    h2_p99_clip : bool, default=True
        Si True, clipea outliers en H₂
    eps : float, default=1e-8
        Epsilon para evitar divisiones por cero
    verbose : bool, default=True
        Mostrar progreso
        
    Returns
    -------
    dict
        Diccionario con resultados IRCT para cada escenario:
        {
            'T245': {...},
            'T370': {...},
            'T585': {...}
        }
        
    Example
    -------
    >>> irct_all = compute_IRCT_all_scenarios(
    ...     model=MODELS["VAE"],
    ...     clustering_results=CLUSTERING_RESULTS["VAE"],
    ...     X_base_orig=X_BASE,
    ...     X_base_norm=X_base_normalized,
    ...     X245_orig=X245_orig,
    ...     X245_norm=X245_norm,
    ...     X370_orig=X370_orig,
    ...     X370_norm=X370_norm,
    ...     X585_orig=X585_orig,
    ...     X585_norm=X585_norm,
    ...     n_per_scenario=N_PER_SCENARIO
    ... )
    >>> 
    >>> print(f"IRCT medio SSP585: {irct_all['T585']['IRCT'].mean():.3f}")
    """
    scenarios = ['T245', 'T370', 'T585']
    scenario_data = {
        'T245': (X245_orig, X245_norm, h2_245),
        'T370': (X370_orig, X370_norm, h2_370),
        'T585': (X585_orig, X585_norm, h2_585),
    }
    
    irct_results_all = {}
    
    for i, scenario in enumerate(scenarios):
        if verbose:
            print(f"\nCalculando IRCT para escenario {scenario}...")
        
        # Extraer segmento correcto de BASE
        start_idx = i * n_per_scenario
        end_idx = (i + 1) * n_per_scenario
        
        X_base_orig_segment = X_base_orig[start_idx:end_idx]
        X_base_norm_segment = X_base_norm[start_idx:end_idx]
        
        # Extraer H₂ base si está disponible
        h2_base_segment = None
        if h2_base is not None:
            h2_base_segment = h2_base[start_idx:end_idx]
        
        # Datos del escenario futuro
        X_future_orig, X_future_norm, h2_future = scenario_data[scenario]
        
        # Calcular IRCT
        irct_results = compute_IRCT_from_clustering_results(
            model=model,
            clustering_results=clustering_results,
            X_base_orig=X_base_orig_segment,
            X_base_norm=X_base_norm_segment,
            X_future_orig=X_future_orig,
            X_future_norm=X_future_norm,
            scenario=scenario,
            h2_base=h2_base_segment,
            h2_future=h2_future,
            weights=weights,
            device=device,
            recon_use_normalized=recon_use_normalized,
            inverse_transform=inverse_transform,
            softmax_tau=softmax_tau,
            expansion_p99_clip=expansion_p99_clip,
            h2_p99_clip=h2_p99_clip,
            eps=eps
        )
        
        irct_results_all[scenario] = irct_results
        
        if verbose:
            print(f"  ✓ IRCT calculado para {scenario}")
            print(f"    Media: {irct_results['IRCT'].mean():.3f}")
            print(f"    Std: {irct_results['IRCT'].std():.3f}")
            print(f"    Min: {irct_results['IRCT'].min():.3f}")
            print(f"    Max: {irct_results['IRCT'].max():.3f}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ IRCT calculado para todos los escenarios")
        print(f"{'='*60}")
    
    return irct_results_all


# ============================================================================
# AGREGACIÓN DEL IRCT POR CLUSTER
# ============================================================================

def aggregate_IRCT_by_cluster(
    irct_results,
    labels_base,
    n_clusters,
    percentiles=[10, 50, 90]
):
    """
    Agrega métricas del IRCT por cluster.
    
    Calcula estadísticos agregados (media, std, percentiles) del IRCT
    y sus componentes para cada cluster.
    
    Parameters
    ----------
    irct_results : dict
        Diccionario retornado por compute_IRCT_pixel_wise o 
        compute_IRCT_from_clustering_results
    labels_base : np.ndarray
        Etiquetas de cluster asignadas a cada píxel
    n_clusters : int
        Número total de clusters
    percentiles : list, default=[10, 50, 90]
        Percentiles a calcular
        
    Returns
    -------
    pd.DataFrame
        DataFrame con estadísticos agregados por cluster
        
    Example
    -------
    >>> irct_585 = compute_IRCT_from_clustering_results(...)
    >>> agg_df = aggregate_IRCT_by_cluster(
    ...     irct_585, 
    ...     CLUSTERING_RESULTS["VAE"]["labels_B585"], 
    ...     K_CLUSTERS
    ... )
    >>> print(agg_df[['cluster_id', 'IRCT_mean', 'IRCT_std']])
    """
    metrics = ['reconstruction_anomaly', 'latent_displacement', 
               'cluster_stability', 'cluster_expansion', 'IRCT']
    
    # Agregar h2_stability si está disponible
    if irct_results['h2_stability'] is not None:
        metrics.append('h2_stability')
    
    aggregated = []
    
    for cluster_id in range(n_clusters):
        mask = (labels_base == cluster_id)
        n_pixels = mask.sum()
        
        if n_pixels == 0:
            continue
        
        cluster_stats = {'cluster_id': cluster_id, 'n_pixels': n_pixels}
        
        for metric in metrics:
            values = irct_results[metric]
            if values is None:
                continue
            
            cluster_values = values[mask]
            
            # Estadísticos básicos
            cluster_stats[f'{metric}_mean'] = cluster_values.mean()
            cluster_stats[f'{metric}_std'] = cluster_values.std()
            
            # Percentiles
            for p in percentiles:
                cluster_stats[f'{metric}_p{p}'] = np.percentile(cluster_values, p)
        
        aggregated.append(cluster_stats)
    
    return pd.DataFrame(aggregated)


# ============================================================================
# VISUALIZACIÓN ESPACIAL DEL IRCT CON INTERPOLACIÓN Y BASEMAP
# ============================================================================

def plot_IRCT_spatial(
    irct_results,
    coords_df,
    metric='IRCT',
    scenario='T585',
    model_key='VAE',
    n_neighbors=15,
    grid_res=100,
    figsize=(12, 8),
    cmap='RdYlGn',
    vmin=None,
    vmax=None,
    alpha=0.8,
    use_basemap=True,
    title=None,
    save_path=None,
    dpi=150
):
    """
    Visualiza espacialmente el IRCT o cualquiera de sus componentes con interpolación KNN y basemap.
    
    Usa interpolación KNN para crear un heatmap suave y opcionalmente agrega un basemap 
    geográfico usando contextily.
    
    Parameters
    ----------
    irct_results : dict
        Diccionario retornado por compute_IRCT_pixel_wise
    coords_df : pd.DataFrame
        DataFrame con columnas 'lat' y 'lon' para cada píxel
    metric : str, default='IRCT'
        Métrica a visualizar: 'IRCT', 'reconstruction_anomaly', 'latent_displacement',
        'cluster_stability', 'cluster_expansion', 'h2_stability'
    scenario : str, default='T585'
        Nombre del escenario para el título
    model_key : str, default='VAE'
        Nombre del modelo para el título
    n_neighbors : int, default=15
        Número de vecinos para interpolación KNN
    grid_res : int, default=100
        Resolución de la grilla de interpolación
    figsize : tuple, default=(12, 8)
        Tamaño de la figura
    cmap : str, default='RdYlGn'
        Colormap (RdYlGn: verde=resiliente, rojo=vulnerable)
    vmin : float or None
        Valor mínimo para escala de color (default: percentil 2)
    vmax : float or None
        Valor máximo para escala de color (default: percentil 98)
    alpha : float, default=0.8
        Transparencia del heatmap (0=transparente, 1=opaco)
    use_basemap : bool, default=True
        Si True, agrega basemap de contextily
    title : str or None
        Título personalizado
    save_path : str or None
        Ruta para guardar
    dpi : int, default=150
        Resolución para guardar
        
    Returns
    -------
    fig, ax
        Figura y axes de matplotlib
        
    Example
    -------
    >>> plot_IRCT_spatial(
    ...     irct_results=IRCT_RESULTS['VAE']['T585'],
    ...     coords_df=coords_df,
    ...     metric='IRCT',
    ...     scenario='SSP585',
    ...     n_neighbors=15,
    ...     use_basemap=True
    ... )
    """
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsRegressor
    
    # Validar métrica
    valid_metrics = ['IRCT', 'reconstruction_anomaly', 'latent_displacement', 
                     'cluster_stability', 'cluster_expansion', 'h2_stability']
    if metric not in valid_metrics:
        raise ValueError(f"metric debe ser uno de {valid_metrics}")
    
    # Extraer valores
    values = irct_results[metric]
    if values is None:
        raise ValueError(f"La métrica '{metric}' no está disponible en irct_results")
    
    if len(values) != len(coords_df):
        raise ValueError(f"Dimensiones incompatibles: values={len(values)}, coords={len(coords_df)}")
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extraer coordenadas
    lats = coords_df['lat'].values
    lons = coords_df['lon'].values
    
    # Transformar a Web Mercator (EPSG:3857) para basemap
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xs, ys = transformer.transform(lons, lats)
    except ImportError:
        print("Warning: pyproj no disponible, usando lat/lon directo")
        xs, ys = lons, lats
        use_basemap = False
    
    # Filtrar valores válidos
    valid_mask = ~(np.isnan(values) | np.isinf(values))
    coords = np.column_stack([xs[valid_mask], ys[valid_mask]])
    vals = values[valid_mask]
    
    if len(vals) == 0:
        raise ValueError(f"No hay valores válidos para la métrica '{metric}'")
    
    # Crear grid de interpolación
    grid_x = np.linspace(xs.min(), xs.max(), grid_res)
    grid_y = np.linspace(ys.min(), ys.max(), grid_res)
    GX, GY = np.meshgrid(grid_x, grid_y)
    extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
    grid_points = np.column_stack([GX.ravel(), GY.ravel()])
    
    # Interpolación KNN
    knn = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(vals)), weights="distance")
    knn.fit(coords, vals)
    pred_vals = knn.predict(grid_points).reshape(GX.shape)
    
    # Configurar límites del eje
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    # Agregar basemap si está disponible
    if use_basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                          crs="EPSG:3857", alpha=1.0, attribution_size=6, zoom='auto')
        except ImportError:
            print("Warning: contextily no disponible, omitiendo basemap")
        except Exception as e:
            print(f"Warning: Error al cargar basemap: {e}")
    
    # Determinar rango de colores
    if vmin is None:
        vmin = np.percentile(vals, 2)
    if vmax is None:
        vmax = np.percentile(vals, 98)
    
    # Overlay del heatmap
    im = ax.imshow(pred_vals, extent=extent, origin="lower",
                  cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, zorder=3)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Labels para métricas
    metric_labels = {
        'IRCT': 'IRCT\n(0=vulnerable, 1=resiliente)',
        'reconstruction_anomaly': 'Anomalía Reconstrucción (A)\n(0=malo, 1=bueno)',
        'latent_displacement': 'Desplazamiento Latente (S_D)\n(0=malo, 1=bueno)',
        'cluster_stability': 'Estabilidad Cluster (S_C)\n(0=malo, 1=bueno)',
        'cluster_expansion': 'Estabilidad Expansión (S_E)\n(0=malo, 1=bueno)',
        'h2_stability': 'Estabilidad H₂ (S_H₂)\n(0=malo, 1=bueno)'
    }
    
    cbar.set_label(metric_labels[metric], fontsize=10)
    
    # Título con estadísticas
    mean_val = vals.mean()
    std_val = vals.std()
    median_val = np.median(vals)
    
    if title is None:
        metric_short = {
            'IRCT': 'IRCT',
            'reconstruction_anomaly': 'Anomalía de Reconstrucción',
            'latent_displacement': 'Desplazamiento Latente',
            'cluster_stability': 'Estabilidad de Cluster',
            'cluster_expansion': 'Estabilidad de Expansión',
            'h2_stability': 'Estabilidad de H₂'
        }
        title = f"{metric_short[metric]} — {model_key} — {scenario}\nμ={mean_val:.3f}, σ={std_val:.3f}, med={median_val:.3f}"
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_axis_off()
    
    plt.tight_layout()
    
    # Guardar si se especifica
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  → Figura guardada: {save_path}")
    
    plt.show()
    
    return fig, ax


def plot_IRCT_components_spatial(
    irct_results,
    coords_df,
    scenario='T585',
    model_key='VAE',
    n_neighbors=15,
    grid_res=100,
    figsize=(24, 14),
    alpha=0.8,
    use_basemap=True,
    save_path=None,
    dpi=300
):
    """
    Visualiza todos los componentes del IRCT en un panel espacial de 2×3 subplots 
    con interpolación KNN y basemap.
    
    Crea un grid mostrando los 5 componentes individuales del IRCT más el índice final.
    
    Parameters
    ----------
    irct_results : dict
        Diccionario retornado por compute_IRCT_pixel_wise
    coords_df : pd.DataFrame
        DataFrame con columnas 'lat' y 'lon'
    scenario : str, default='T585'
        Nombre del escenario para el título
    model_key : str, default='VAE'
        Nombre del modelo para el título
    n_neighbors : int, default=15
        Número de vecinos para interpolación KNN
    grid_res : int, default=100
        Resolución de la grilla de interpolación
    figsize : tuple, default=(24, 14)
        Tamaño de la figura completa
    alpha : float, default=0.8
        Transparencia del heatmap
    use_basemap : bool, default=True
        Si True, agrega basemap de contextily
    save_path : str or None
        Ruta para guardar la figura
    dpi : int, default=300
        Resolución para guardar
        
    Returns
    -------
    fig, axes
        Figura y array de axes de matplotlib
        
    Example
    -------
    >>> plot_IRCT_components_spatial(
    ...     irct_results=IRCT_RESULTS['VAE']['T585'],
    ...     coords_df=coords_df,
    ...     scenario='SSP585',
    ...     model_key='VAE',
    ...     n_neighbors=15,
    ...     use_basemap=True
    ... )
    """
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsRegressor
    
    # Coordenadas
    lats = coords_df['lat'].values
    lons = coords_df['lon'].values
    
    # Transformar a Web Mercator
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xs, ys = transformer.transform(lons, lats)
    except ImportError:
        print("Warning: pyproj no disponible, usando lat/lon directo")
        xs, ys = lons, lats
        use_basemap = False
    
    # Crear grid de interpolación
    grid_x = np.linspace(xs.min(), xs.max(), grid_res)
    grid_y = np.linspace(ys.min(), ys.max(), grid_res)
    GX, GY = np.meshgrid(grid_x, grid_y)
    extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
    grid_points = np.column_stack([GX.ravel(), GY.ravel()])
    
    # Definir componentes con sus configuraciones
    components = [
        ('reconstruction_anomaly', 'Anomalía Reconstrucción\n(error autoencoder)', 'RdYlGn_r'),
        ('latent_displacement', 'Desplazamiento Latente\n(deriva en espacio latente)', 'RdYlGn_r'),
        ('cluster_stability', 'Estabilidad Cluster\n(retención de cluster)', 'RdYlGn'),
        ('cluster_expansion', 'Expansión Cluster\n(compacidad espacial)', 'RdYlGn_r'),
        ('IRCT', 'IRCT FINAL (Agregado)', 'RdYlGn'),
    ]
    
    # Agregar H₂ si está disponible
    if irct_results.get('h2_stability') is not None:
        components.insert(4, ('h2_stability', 'Estabilidad Energética\n(producción H₂)', 'RdYlGn'))
    
    # Crear figura
    n_components = len(components)
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot cada componente
    for idx, (key, label, cmap) in enumerate(components):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        values = irct_results[key]
        
        if values is None:
            ax.text(0.5, 0.5, f'No data for {key}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
        
        # Filtrar valores válidos
        valid_mask = ~(np.isnan(values) | np.isinf(values))
        coords = np.column_stack([xs[valid_mask], ys[valid_mask]])
        vals = values[valid_mask]
        
        if len(vals) == 0:
            ax.text(0.5, 0.5, f'No valid data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
        
        # Interpolación KNN
        knn = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(vals)), weights="distance")
        knn.fit(coords, vals)
        pred_vals = knn.predict(grid_points).reshape(GX.shape)
        
        # Configurar límites
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
        # Basemap
        if use_basemap:
            try:
                import contextily as ctx
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                              crs="EPSG:3857", alpha=1.0, attribution_size=6, zoom='auto')
            except:
                pass
        
        # Overlay del componente
        vmin = np.percentile(vals, 2)
        vmax = np.percentile(vals, 98)
        im = ax.imshow(pred_vals, extent=extent, origin="lower",
                      cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, zorder=3)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("(0=malo, 1=bueno)" if key != 'IRCT' else "(0=vulnerable, 1=resiliente)", 
                      fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        
        # Título con estadísticas
        mean_val = vals.mean()
        std_val = vals.std()
        median_val = np.median(vals)
        
        if key == 'IRCT':
            title_text = f"{label}\nμ={mean_val:.3f}, σ={std_val:.3f}, med={median_val:.3f}"
        else:
            title_text = f"{label}\nμ={mean_val:.3f}, σ={std_val:.3f}"
        
        ax.set_title(title_text, fontsize=11, fontweight='bold', pad=8)
        ax.set_axis_off()
    
    # Ocultar subplots vacíos
    for idx in range(n_components, len(axes)):
        axes[idx].axis('off')
    
    # Título general
    fig.suptitle(
        f'Descomposición del IRCT — {model_key} — {scenario}\n' +
        f'IRCT = f(A, S_D, S_C, S_E) | Cada componente contribuye al índice final',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Guardar si se especifica
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  → Figura guardada: {save_path}")
    
    plt.show()
    
    return fig, axes


def plot_IRCT_vs_clusters_comparison(
    irct_results,
    clustering_results,
    coords_df,
    scenario='T585',
    model_key='VAE',
    n_clusters=10,
    n_neighbors=15,
    grid_res=100,
    figsize=(20, 8),
    alpha=0.8,
    use_basemap=True,
    save_path=None,
    dpi=300
):
    """
    Visualiza lado a lado el IRCT y la estructura de clusters para comparación.
    
    Panel izquierdo: Heatmap interpolado del IRCT
    Panel derecho: Estructura de clusters (FUTURO)
    
    Parameters
    ----------
    irct_results : dict
        Diccionario retornado por compute_IRCT_pixel_wise
    clustering_results : dict
        Diccionario retornado por cluster_and_measure_resilience
    coords_df : pd.DataFrame
        DataFrame con columnas 'lat' y 'lon'
    scenario : str, default='T585'
        Escenario: 'T245', 'T370', 'T585'
    model_key : str, default='VAE'
        Nombre del modelo
    n_clusters : int, default=10
        Número de clusters
    n_neighbors : int, default=15
        Vecinos para interpolación KNN
    grid_res : int, default=100
        Resolución de grilla
    figsize : tuple, default=(20, 8)
        Tamaño de figura
    alpha : float, default=0.8
        Transparencia del heatmap
    use_basemap : bool, default=True
        Si True, agrega basemap
    save_path : str or None
        Ruta para guardar
    dpi : int, default=300
        Resolución para guardar
        
    Returns
    -------
    fig, (ax1, ax2)
        Figura y tupla de axes
        
    Example
    -------
    >>> plot_IRCT_vs_clusters_comparison(
    ...     irct_results=IRCT_RESULTS['VAE']['T585'],
    ...     clustering_results=CLUSTERING_RESULTS['VAE'],
    ...     coords_df=coords_df,
    ...     scenario='T585',
    ...     model_key='VAE'
    ... )
    """
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from matplotlib.colors import ListedColormap
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Coordenadas
    lats = coords_df['lat'].values
    lons = coords_df['lon'].values
    
    # Transformar a Web Mercator
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xs, ys = transformer.transform(lons, lats)
    except ImportError:
        print("Warning: pyproj no disponible")
        xs, ys = lons, lats
        use_basemap = False
    
    # Crear grid
    grid_x = np.linspace(xs.min(), xs.max(), grid_res)
    grid_y = np.linspace(ys.min(), ys.max(), grid_res)
    GX, GY = np.meshgrid(grid_x, grid_y)
    extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
    grid_points = np.column_stack([GX.ravel(), GY.ravel()])
    
    # === PANEL IZQUIERDO: IRCT ===
    irct_vals = irct_results['IRCT']
    valid_mask = ~(np.isnan(irct_vals) | np.isinf(irct_vals))
    coords = np.column_stack([xs[valid_mask], ys[valid_mask]])
    vals = irct_vals[valid_mask]
    
    # Interpolación KNN para IRCT
    knn = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(vals)), weights="distance")
    knn.fit(coords, vals)
    pred_irct = knn.predict(grid_points).reshape(GX.shape)
    
    # Configurar ax1
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_ylim(extent[2], extent[3])
    
    # Basemap
    if use_basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron,
                          crs="EPSG:3857", alpha=1.0, attribution_size=6, zoom='auto')
        except:
            pass
    
    # Overlay IRCT
    im1 = ax1.imshow(pred_irct, extent=extent, origin="lower",
                    cmap='RdYlGn', alpha=alpha, vmin=0, vmax=1, zorder=3)
    
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("IRCT\n(0=vulnerable, 1=resiliente)", fontsize=11)
    
    # Estadísticas
    mean_irct = vals.mean()
    std_irct = vals.std()
    median_irct = np.median(vals)
    
    ax1.set_title(
        f"Índice de Resiliencia (IRCT)\n{model_key} — SSP{scenario[1:]}\n" +
        f"μ={mean_irct:.3f}, σ={std_irct:.3f}, med={median_irct:.3f}",
        fontsize=13, fontweight='bold', pad=10
    )
    ax1.set_axis_off()
    
    # === PANEL DERECHO: CLUSTERS ===
    # Extraer labels según escenario
    scenario_map = {
        'T245': ('labels_B245', 'labels_T245'),
        'T370': ('labels_B370', 'labels_T370'),
        'T585': ('labels_B585', 'labels_T585'),
    }
    labels_base_key, labels_future_key = scenario_map[scenario]
    labels_base = clustering_results[labels_base_key]
    labels_future = clustering_results[labels_future_key]
    
    # Interpolación de clusters usando clasificador KNN
    coords_all = np.column_stack([xs, ys])
    knn_cluster = KNeighborsClassifier(
        n_neighbors=min(n_neighbors, len(labels_future)),
        weights="distance"
    )
    knn_cluster.fit(coords_all, labels_future.astype(int))
    pred_clusters = knn_cluster.predict(grid_points).reshape(GX.shape)
    
    # Configurar ax2
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    
    # Basemap
    if use_basemap:
        try:
            ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron,
                          crs="EPSG:3857", alpha=1.0, attribution_size=6, zoom='auto')
        except:
            pass
    
    # Colormap categórico para clusters
    color_palette = plt.get_cmap('tab20', 20)
    cluster_colors = [color_palette(int(k) % 20) for k in range(n_clusters)]
    cmap_clusters = ListedColormap(cluster_colors)
    
    # Overlay clusters
    im2 = ax2.imshow(pred_clusters, extent=extent, origin="lower",
                    cmap=cmap_clusters, alpha=0.7, vmin=0, vmax=n_clusters-1, zorder=3)
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04,
                        ticks=range(n_clusters))
    cbar2.set_label("Cluster ID\n(estructura territorial)", fontsize=11)
    
    # Calcular retención por cluster
    retention_rates = []
    for cluster_id in range(n_clusters):
        mask = labels_base == cluster_id
        if mask.sum() > 0:
            retention = (labels_future[mask] == cluster_id).sum() / mask.sum() * 100
            retention_rates.append(retention)
        else:
            retention_rates.append(0)
    
    avg_retention = np.mean(retention_rates)
    
    ax2.set_title(
        f"Estructura de Clusters (Futuro)\n{model_key} — SSP{scenario[1:]}\n" +
        f"Retención promedio: {avg_retention:.1f}%",
        fontsize=13, fontweight='bold', pad=10
    )
    ax2.set_axis_off()
    
    plt.tight_layout()
    
    # Guardar si se especifica
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  → Figura guardada: {save_path}")
    
    plt.show()
    
    return fig, (ax1, ax2)


# ============================================================================
# ANÁLISIS DE SENSIBILIDAD DE PESOS
# ============================================================================

def sensitivity_analysis_univariate(
    model,
    clustering_results,
    X_base_orig,
    X_base_norm,
    X_future_orig,
    X_future_norm,
    scenario='T585',
    weight_to_vary='w_d',
    perturbations=None,
    base_weights=None,
    n_per_scenario=None,
    device='cpu',
    verbose=True
):
    """
    Análisis de sensibilidad UNIVARIADO del IRCT: varía un peso a la vez.
    
    Calcula IRCT bajo diferentes valores de un peso específico, manteniendo
    los demás constantes (renormalizados). Útil para evaluar robustez del índice.
    
    Parameters
    ----------
    model : nn.Module
        Modelo AE o VAE entrenado
    clustering_results : dict
        Resultados de clustering
    X_base_orig, X_base_norm : np.ndarray
        Datos BASE originales y normalizados
    X_future_orig, X_future_norm : np.ndarray
        Datos FUTURE originales y normalizados
    scenario : str, default='T585'
        Escenario a analizar: 'T245', 'T370', 'T585'
    weight_to_vary : str, default='w_d'
        Peso a variar: 'w_a', 'w_d', 'w_c', 'w_e', 'w_h'
    perturbations : array-like or None
        Valores a probar para el peso. Si None, usa np.linspace(0.05, 0.50, 10)
    base_weights : dict or None
        Pesos base. Si None, usa esquema 'theoretical'
    n_per_scenario : int or None
        Número de píxeles por escenario (para extraer segmento correcto de BASE)
    device : str, default='cpu'
        Device para PyTorch
    verbose : bool, default=True
        Mostrar progreso
        
    Returns
    -------
    dict
        Diccionario con:
        - 'perturbations': valores del peso probados
        - 'irct_results': lista de resultados IRCT para cada perturbación
        - 'spearman_correlations': correlaciones de Spearman con IRCT base
        - 'cv_by_pixel': coeficiente de variación por píxel
        - 'stability_score': score global de estabilidad [0,1]
        - 'base_ranking': ranking de píxeles con peso base
        - 'summary_df': DataFrame resumen
        
    Example
    -------
    >>> sens_results = sensitivity_analysis_univariate(
    ...     model=MODELS['VAE'],
    ...     clustering_results=CLUSTERING_RESULTS['VAE'],
    ...     X_base_orig=X_BASE_model,
    ...     X_base_norm=X_BASE_model,
    ...     X_future_orig=X585_model,
    ...     X_future_norm=X585_model,
    ...     scenario='T585',
    ...     weight_to_vary='w_d',
    ...     perturbations=np.linspace(0.10, 0.50, 9)
    ... )
    >>> print(f"Estabilidad: {sens_results['stability_score']:.3f}")
    """
    # Pesos base
    if base_weights is None:
        base_weights = get_weight_scheme('theoretical', verbose=False)
    
    # Perturbaciones a probar
    if perturbations is None:
        perturbations = np.linspace(0.05, 0.50, 10)
    
    # Determinar segmento de BASE según escenario
    if n_per_scenario is not None:
        scenario_idx = ['T245', 'T370', 'T585'].index(scenario)
        start_idx = scenario_idx * n_per_scenario
        end_idx = (scenario_idx + 1) * n_per_scenario
        X_base_orig_segment = X_base_orig[start_idx:end_idx]
        X_base_norm_segment = X_base_norm[start_idx:end_idx]
    else:
        X_base_orig_segment = X_base_orig
        X_base_norm_segment = X_base_norm
    
    if verbose:
        print(f"\nANÁLISIS DE SENSIBILIDAD UNIVARIADO")
        print("="*80)
        print(f"Peso a variar: {weight_to_vary}")
        print(f"Rango: [{perturbations.min():.2f}, {perturbations.max():.2f}]")
        print(f"Número de valores: {len(perturbations)}")
        print(f"Escenario: {scenario}")
        print()
    
    # Calcular IRCT base (con pesos originales)
    if verbose:
        print("Calculando IRCT base...")
    
    irct_base = compute_IRCT_from_clustering_results(
        model=model,
        clustering_results=clustering_results,
        X_base_orig=X_base_orig_segment,
        X_base_norm=X_base_norm_segment,
        X_future_orig=X_future_orig,
        X_future_norm=X_future_norm,
        scenario=scenario,
        weights=base_weights,
        device=device
    )
    
    base_irct_values = irct_base['IRCT']
    base_ranking = rankdata(base_irct_values, method='ordinal')
    
    # Iterar sobre perturbaciones
    irct_results_list = []
    spearman_corrs = []
    
    for i, perturb_val in enumerate(perturbations):
        if verbose:
            print(f"  [{i+1}/{len(perturbations)}] {weight_to_vary} = {perturb_val:.3f}... ", end='')
        
        # Crear pesos modificados
        modified_weights = base_weights.copy()
        modified_weights[weight_to_vary] = perturb_val
        
        # Renormalizar los demás pesos para que sumen 1.0
        total = sum(modified_weights.values())
        modified_weights = {k: v/total for k, v in modified_weights.items()}
        
        # Calcular IRCT con pesos modificados
        irct_perturbed = compute_IRCT_from_clustering_results(
            model=model,
            clustering_results=clustering_results,
            X_base_orig=X_base_orig_segment,
            X_base_norm=X_base_norm_segment,
            X_future_orig=X_future_orig,
            X_future_norm=X_future_norm,
            scenario=scenario,
            weights=modified_weights,
            device=device
        )
        
        irct_results_list.append(irct_perturbed)
        
        # Calcular correlación de Spearman con ranking base
        perturbed_ranking = rankdata(irct_perturbed['IRCT'], method='ordinal')
        spearman_corr, _ = spearmanr(base_ranking, perturbed_ranking)
        spearman_corrs.append(spearman_corr)
        
        if verbose:
            print(f"ρ={spearman_corr:.4f}")
    
    # Calcular coeficiente de variación por píxel
    irct_matrix = np.vstack([res['IRCT'] for res in irct_results_list]).T  # (n_pixels, n_perturbations)
    cv_by_pixel = np.std(irct_matrix, axis=1) / (np.mean(irct_matrix, axis=1) + 1e-8)
    
    # Score global de estabilidad: promedio de correlaciones
    stability_score = np.mean(spearman_corrs)
    
    # DataFrame resumen
    summary_df = pd.DataFrame({
        weight_to_vary: perturbations,
        'spearman_corr': spearman_corrs,
        'mean_irct': [res['IRCT'].mean() for res in irct_results_list],
        'std_irct': [res['IRCT'].std() for res in irct_results_list],
        'median_irct': [np.median(res['IRCT']) for res in irct_results_list]
    })
    
    if verbose:
        print()
        print(f"RESULTADOS:")
        print(f"  • Correlación Spearman promedio: {stability_score:.4f}")
        print(f"  • Coeficiente de variación medio (píxeles): {cv_by_pixel.mean():.4f}")
        print(f"  • Coeficiente de variación máximo: {cv_by_pixel.max():.4f}")
        print()
        
        # Interpretación
        if stability_score >= 0.90:
            print("  ✓ ROBUSTEZ ALTA: El ranking se mantiene muy estable")
        elif stability_score >= 0.75:
            print("  → ROBUSTEZ MODERADA: Cambios menores en el ranking")
        else:
            print("  ⚠ ROBUSTEZ BAJA: El índice es sensible a este peso")
        print("="*80)
    
    return {
        'weight_varied': weight_to_vary,
        'perturbations': perturbations,
        'irct_results': irct_results_list,
        'spearman_correlations': np.array(spearman_corrs),
        'cv_by_pixel': cv_by_pixel,
        'stability_score': stability_score,
        'base_ranking': base_ranking,
        'base_irct': base_irct_values,
        'summary_df': summary_df,
        'irct_matrix': irct_matrix
    }


def compare_weight_schemes(
    model,
    clustering_results,
    X_base_orig,
    X_base_norm,
    X_future_orig,
    X_future_norm,
    scenario='T585',
    schemes=['theoretical', 'uniform', 'latent_focused'],
    n_per_scenario=None,
    device='cpu',
    verbose=True
):
    """
    Compara IRCT bajo diferentes esquemas de ponderación predefinidos.
    
    Calcula IRCT para múltiples esquemas de pesos y evalúa:
    - Correlación de Spearman entre rankings
    - Coeficiente de variación espacial
    - Identificación de píxeles "estables" vs "sensibles"
    
    Parameters
    ----------
    model : nn.Module
        Modelo AE o VAE entrenado
    clustering_results : dict
        Resultados de clustering
    X_base_orig, X_base_norm : np.ndarray
        Datos BASE originales y normalizados
    X_future_orig, X_future_norm : np.ndarray
        Datos FUTURE originales y normalizados
    scenario : str, default='T585'
        Escenario a analizar
    schemes : list, default=['theoretical', 'uniform', 'latent_focused']
        Lista de esquemas a comparar
    n_per_scenario : int or None
        Número de píxeles por escenario
    device : str, default='cpu'
        Device para PyTorch
    verbose : bool, default=True
        Mostrar progreso
        
    Returns
    -------
    dict
        Diccionario con:
        - 'schemes': lista de esquemas evaluados
        - 'irct_results': dict con resultados IRCT por esquema
        - 'correlation_matrix': matriz de correlaciones de Spearman
        - 'cv_by_pixel': CV por píxel entre esquemas
        - 'stable_pixels': máscara de píxeles con CV < 0.15
        - 'sensitive_pixels': máscara de píxeles con CV > 0.30
        - 'summary_df': DataFrame comparativo
        
    Example
    -------
    >>> comparison = compare_weight_schemes(
    ...     model=MODELS['VAE'],
    ...     clustering_results=CLUSTERING_RESULTS['VAE'],
    ...     X_base_orig=X_BASE_model,
    ...     X_base_norm=X_BASE_model,
    ...     X_future_orig=X585_model,
    ...     X_future_norm=X585_model,
    ...     scenario='T585',
    ...     schemes=['theoretical', 'uniform', 'latent_focused']
    ... )
    >>> print(comparison['correlation_matrix'])
    """
    # Determinar segmento de BASE
    if n_per_scenario is not None:
        scenario_idx = ['T245', 'T370', 'T585'].index(scenario)
        start_idx = scenario_idx * n_per_scenario
        end_idx = (scenario_idx + 1) * n_per_scenario
        X_base_orig_segment = X_base_orig[start_idx:end_idx]
        X_base_norm_segment = X_base_norm[start_idx:end_idx]
    else:
        X_base_orig_segment = X_base_orig
        X_base_norm_segment = X_base_norm
    
    if verbose:
        print(f"\nCOMPARACIÓN DE ESQUEMAS DE PESOS")
        print("="*80)
        print(f"Escenario: {scenario}")
        print(f"Esquemas: {', '.join(schemes)}")
        print()
    
    # Calcular IRCT para cada esquema
    irct_results_dict = {}
    
    for scheme in schemes:
        if verbose:
            print(f"Calculando IRCT con esquema '{scheme}'...")
        
        weights = get_weight_scheme(scheme, verbose=False)
        
        irct_result = compute_IRCT_from_clustering_results(
            model=model,
            clustering_results=clustering_results,
            X_base_orig=X_base_orig_segment,
            X_base_norm=X_base_norm_segment,
            X_future_orig=X_future_orig,
            X_future_norm=X_future_norm,
            scenario=scenario,
            weights=weights,
            device=device
        )
        
        irct_results_dict[scheme] = irct_result
        
        if verbose:
            print(f"  → IRCT medio: {irct_result['IRCT'].mean():.4f} ± {irct_result['IRCT'].std():.4f}")
    
    # Matriz de correlaciones de Spearman
    n_schemes = len(schemes)
    correlation_matrix = np.zeros((n_schemes, n_schemes))
    
    for i, scheme_i in enumerate(schemes):
        for j, scheme_j in enumerate(schemes):
            rank_i = rankdata(irct_results_dict[scheme_i]['IRCT'], method='ordinal')
            rank_j = rankdata(irct_results_dict[scheme_j]['IRCT'], method='ordinal')
            corr, _ = spearmanr(rank_i, rank_j)
            correlation_matrix[i, j] = corr
    
    # Coeficiente de variación por píxel
    irct_matrix = np.vstack([irct_results_dict[s]['IRCT'] for s in schemes]).T
    cv_by_pixel = np.std(irct_matrix, axis=1) / (np.mean(irct_matrix, axis=1) + 1e-8)
    
    # Identificar píxeles estables vs sensibles
    stable_pixels = cv_by_pixel < 0.15  # CV < 15%
    sensitive_pixels = cv_by_pixel > 0.30  # CV > 30%
    
    # DataFrame resumen
    summary_data = []
    for scheme in schemes:
        irct_vals = irct_results_dict[scheme]['IRCT']
        summary_data.append({
            'scheme': scheme,
            'mean': irct_vals.mean(),
            'std': irct_vals.std(),
            'median': np.median(irct_vals),
            'p10': np.percentile(irct_vals, 10),
            'p90': np.percentile(irct_vals, 90)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if verbose:
        print()
        print("MATRIZ DE CORRELACIONES (Spearman):")
        corr_df = pd.DataFrame(correlation_matrix, index=schemes, columns=schemes)
        print(corr_df.round(4))
        print()
        print(f"VARIABILIDAD ESPACIAL:")
        print(f"  • CV medio entre esquemas: {cv_by_pixel.mean():.4f}")
        print(f"  • Píxeles estables (CV < 0.15): {stable_pixels.sum()} ({100*stable_pixels.mean():.1f}%)")
        print(f"  • Píxeles sensibles (CV > 0.30): {sensitive_pixels.sum()} ({100*sensitive_pixels.mean():.1f}%)")
        print()
        print("RESUMEN POR ESQUEMA:")
        print(summary_df.to_string(index=False))
        print("="*80)
    
    return {
        'schemes': schemes,
        'irct_results': irct_results_dict,
        'correlation_matrix': correlation_matrix,
        'correlation_df': pd.DataFrame(correlation_matrix, index=schemes, columns=schemes),
        'cv_by_pixel': cv_by_pixel,
        'stable_pixels': stable_pixels,
        'sensitive_pixels': sensitive_pixels,
        'summary_df': summary_df,
        'irct_matrix': irct_matrix
    }


def plot_sensitivity_analysis(
    sensitivity_results,
    model_key='VAE',
    scenario='T585',
    figsize=(16, 5),
    save_path=None,
    dpi=150
):
    """
    Visualiza resultados del análisis de sensibilidad univariado.
    
    Crea un panel de 3 subplots:
    1. Correlación de Spearman vs valor del peso
    2. Distribución de IRCT para diferentes valores del peso
    3. Mapa de calor de CV por píxel
    
    Parameters
    ----------
    sensitivity_results : dict
        Resultados de sensitivity_analysis_univariate
    model_key : str
        Nombre del modelo para el título
    scenario : str
        Nombre del escenario para el título
    figsize : tuple
        Tamaño de la figura
    save_path : str or None
        Ruta para guardar la figura
    dpi : int
        Resolución para guardar
        
    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    weight_varied = sensitivity_results['weight_varied']
    perturbations = sensitivity_results['perturbations']
    spearman_corrs = sensitivity_results['spearman_correlations']
    summary_df = sensitivity_results['summary_df']
    cv_by_pixel = sensitivity_results['cv_by_pixel']
    
    # Subplot 1: Correlación de Spearman
    ax1 = axes[0]
    ax1.plot(perturbations, spearman_corrs, marker='o', linewidth=2, markersize=6)
    ax1.axhline(0.90, color='green', linestyle='--', label='Robustez alta (ρ≥0.90)')
    ax1.axhline(0.75, color='orange', linestyle='--', label='Robustez moderada (ρ≥0.75)')
    ax1.set_xlabel(f'Valor de {weight_varied}', fontsize=11)
    ax1.set_ylabel('Correlación de Spearman con ranking base', fontsize=11)
    ax1.set_title(f'Estabilidad del Ranking\n{model_key} — {scenario}', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_ylim([0.5, 1.0])
    
    # Subplot 2: Distribución de IRCT
    ax2 = axes[1]
    positions = np.arange(len(perturbations))
    irct_matrix = sensitivity_results['irct_matrix']
    
    bp = ax2.boxplot(irct_matrix, positions=positions, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    
    ax2.set_xticks(positions[::2])
    ax2.set_xticklabels([f'{p:.2f}' for p in perturbations[::2]], rotation=45)
    ax2.set_xlabel(f'Valor de {weight_varied}', fontsize=11)
    ax2.set_ylabel('IRCT', fontsize=11)
    ax2.set_title(f'Distribución del IRCT\n{model_key} — {scenario}', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # Subplot 3: Histograma de CV
    ax3 = axes[2]
    ax3.hist(cv_by_pixel, bins=50, alpha=0.7, edgecolor='black', color='coral')
    ax3.axvline(cv_by_pixel.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Media: {cv_by_pixel.mean():.3f}')
    ax3.axvline(0.15, color='green', linestyle='--', linewidth=1.5, label='Umbral estable (CV=0.15)')
    ax3.axvline(0.30, color='orange', linestyle='--', linewidth=1.5, label='Umbral sensible (CV=0.30)')
    ax3.set_xlabel('Coeficiente de Variación (CV)', fontsize=11)
    ax3.set_ylabel('Frecuencia (píxeles)', fontsize=11)
    ax3.set_title(f'Variabilidad Espacial del IRCT\n{model_key} — {scenario}', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  → Figura guardada: {save_path}")
    
    plt.show()
    
    return fig, axes


def plot_scheme_comparison(
    comparison_results,
    model_key='VAE',
    scenario='T585',
    figsize=(18, 5),
    save_path=None,
    dpi=150
):
    """
    Visualiza comparación entre esquemas de pesos.
    
    Crea un panel de 3 subplots:
    1. Heatmap de correlaciones de Spearman entre esquemas
    2. Boxplot de distribuciones de IRCT por esquema
    3. Scatter plot comparando esquemas principales
    
    Parameters
    ----------
    comparison_results : dict
        Resultados de compare_weight_schemes
    model_key : str
        Nombre del modelo
    scenario : str
        Nombre del escenario
    figsize : tuple
        Tamaño de la figura
    save_path : str or None
        Ruta para guardar
    dpi : int
        Resolución
        
    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    schemes = comparison_results['schemes']
    correlation_matrix = comparison_results['correlation_matrix']
    irct_matrix = comparison_results['irct_matrix']
    
    # Subplot 1: Heatmap de correlaciones
    ax1 = axes[0]
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
               xticklabels=schemes, yticklabels=schemes, ax=ax1,
               vmin=0.5, vmax=1.0, cbar_kws={'label': 'Correlación de Spearman'})
    ax1.set_title(f'Correlación entre Esquemas\n{model_key} — {scenario}', 
                 fontsize=12, fontweight='bold')
    
    # Subplot 2: Boxplot de distribuciones
    ax2 = axes[1]
    positions = np.arange(len(schemes))
    
    bp = ax2.boxplot([irct_matrix[:, i] for i in range(len(schemes))],
                     positions=positions, widths=0.6, patch_artist=True,
                     labels=schemes)
    
    # Colorear según esquema
    colors = [WEIGHT_SCHEMES[s].get('color', 'lightblue') for s in schemes]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('IRCT', fontsize=11)
    ax2.set_title(f'Distribución del IRCT por Esquema\n{model_key} — {scenario}',
                 fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Subplot 3: Scatter plot (theoretical vs uniform)
    ax3 = axes[2]
    irct_theoretical = comparison_results['irct_results']['theoretical']['IRCT']
    irct_uniform = comparison_results['irct_results']['uniform']['IRCT']
    
    ax3.scatter(irct_theoretical, irct_uniform, alpha=0.5, s=20, edgecolors='none')
    
    # Línea 1:1
    min_val = min(irct_theoretical.min(), irct_uniform.min())
    max_val = max(irct_theoretical.max(), irct_uniform.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1')
    
    # Correlación
    rank_theo = rankdata(irct_theoretical)
    rank_unif = rankdata(irct_uniform)
    corr, _ = spearmanr(rank_theo, rank_unif)
    
    ax3.set_xlabel('IRCT (Teórico)', fontsize=11)
    ax3.set_ylabel('IRCT (Uniforme)', fontsize=11)
    ax3.set_title(f'Comparación Esquemas\nρ = {corr:.3f}', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  → Figura guardada: {save_path}")
    
    plt.show()
    
    return fig, axes
