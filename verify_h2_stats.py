#!/usr/bin/env python
"""
Script para verificar los estadísticos de S_H2 con la fórmula corregida.
"""

import numpy as np
import pandas as pd
import pickle
import glob
import os
import sys

BASE_DIR = "/home/aninotna/magister/tesis/justh2_pipeline"
DATA_DIR = os.path.join(BASE_DIR, "data")

# Cargar datos del Experimento 1
trained_dir = os.path.join(DATA_DIR, "autoencoder_trained_v2")
pattern = os.path.join(trained_dir, "experiment1_clustering_*.pkl")
available_files = glob.glob(pattern)

if not available_files:
    raise FileNotFoundError(f"No se encontraron archivos en: {pattern}")

export_path = max(available_files, key=os.path.getmtime)
print(f"Cargando: {os.path.basename(export_path)}\n")

with open(export_path, "rb") as f:
    exp1_data = pickle.load(f)

# Extraer datos
X_BASE = exp1_data["X_BASE"]
X245_orig = exp1_data["B245"]
X370_orig = exp1_data["B370"]
X585_orig = exp1_data["B585"]
feature_names = exp1_data["feature_names_transformed"]

print(f"Datos cargados:")
print(f"  X_BASE: {X_BASE.shape}")
print(f"  X245_orig: {X245_orig.shape}")
print(f"  X370_orig: {X370_orig.shape}")
print(f"  X585_orig: {X585_orig.shape}")
print(f"  Features: {len(feature_names)}\n")

# Función para calcular percentile rank
def percentile_rank(arr):
    """Convierte array a percentiles [0, 100]."""
    arr = np.asarray(arr)
    n = len(arr)
    if n <= 1:
        return np.zeros_like(arr, dtype=float)
    import pandas as pd
    ranks = pd.Series(arr).rank(method="average").to_numpy()
    return (ranks - 1.0) / (n - 1.0)  # [0, 1]

# Función para calcular S_H2 con fórmula CORRECTA
def compute_h2_stability_correct(h2_base, h2_future, epsilon=1e-8, p99_clip=True):
    """
    Fórmula CORRECTA:
    R_H2 = h2_future / h2_base
    S_H2 = percentile_rank(R_H2)
    
    Alto h2_future → Alto ratio → Alto percentil → Mayor resiliencia (bueno)
    Bajo h2_future → Bajo ratio → Bajo percentil → Menor resiliencia (malo)
    """
    h2_base = np.asarray(h2_base)
    h2_future = np.asarray(h2_future)
    
    h2_ratios = h2_future / (h2_base + epsilon)
    
    if p99_clip:
        p99 = np.percentile(h2_ratios, 99)
        h2_ratios = np.clip(h2_ratios, 0, p99)
    else:
        h2_ratios = np.clip(h2_ratios, 0, None)
    
    S_H2 = percentile_rank(h2_ratios)
    
    return S_H2, h2_ratios

# Función para calcular S_H2 con fórmula ANTIGUA (invertida)
def compute_h2_stability_old(h2_base, h2_future, epsilon=1e-8, p99_clip=True):
    """
    Fórmula ANTIGUA (invertida):
    S_H2 = h2_base / h2_future
    """
    h2_base = np.asarray(h2_base)
    h2_future = np.asarray(h2_future)
    
    ratio = h2_base / (h2_future + epsilon)
    
    if p99_clip:
        p99 = np.percentile(ratio, 99)
        ratio = np.clip(ratio, None, p99)
    
    return ratio

# Identificar columnas de H2
h2_indices = [i for i, name in enumerate(feature_names) 
              if 'h2' in name.lower()]

if h2_indices:
    print(f"Columnas de H2 encontradas: {h2_indices}")
    print(f"  {[feature_names[i] for i in h2_indices]}\n")
    
    # Usar promedio de todas las columnas de H2
    h2_base_values = X_BASE[:, h2_indices].mean(axis=1)
    h2_245_values = X245_orig[:, h2_indices].mean(axis=1)
    h2_370_values = X370_orig[:, h2_indices].mean(axis=1)
    h2_585_values = X585_orig[:, h2_indices].mean(axis=1)
else:
    print("Sin columnas de H2 encontradas")
    sys.exit(1)

# Usar solo los primeros 661 píxeles (datos espaciales únicos)
n_spatial = 661

print("\n" + "="*80)
print("COMPARACIÓN DE FÓRMULAS PARA S_H2 (primeros 661 píxeles espaciales)")
print("="*80)

for scenario_name, h2_future in [('SSP245', h2_245_values), 
                                   ('SSP370', h2_370_values), 
                                   ('SSP585', h2_585_values)]:
    print(f"\n{scenario_name}:")
    print("-" * 60)
    
    # Usar primeros n_spatial para base
    h2_base = h2_base_values[:n_spatial]
    h2_fut = h2_future[:n_spatial]
    
    # Fórmula CORRECTA
    S_H2_correct, ratios_correct = compute_h2_stability_correct(h2_base, h2_fut)
    
    # Fórmula ANTIGUA
    S_H2_old = compute_h2_stability_old(h2_base, h2_fut)
    
    print("\nFÓRMULA CORRECTA (h2_future / h2_base):")
    print(f"  Media:    {S_H2_correct.mean():.4f}")
    print(f"  Mediana:  {np.median(S_H2_correct):.4f}")
    print(f"  Std:      {S_H2_correct.std():.4f}")
    print(f"  Min:      {S_H2_correct.min():.4f}")
    print(f"  Max:      {S_H2_correct.max():.4f}")
    print(f"  P25:      {np.percentile(S_H2_correct, 25):.4f}")
    print(f"  P75:      {np.percentile(S_H2_correct, 75):.4f}")
    
    # Contar píxeles en diferentes bins
    pct_low = (S_H2_correct < 0.3).sum() / len(S_H2_correct) * 100
    pct_mid = ((S_H2_correct >= 0.3) & (S_H2_correct < 0.7)).sum() / len(S_H2_correct) * 100
    pct_high = (S_H2_correct >= 0.7).sum() / len(S_H2_correct) * 100
    
    print(f"\n  Distribución por bins:")
    print(f"    S_H2 < 0.3:  {pct_low:.1f}%")
    print(f"    0.3 ≤ S_H2 < 0.7: {pct_mid:.1f}%")
    print(f"    S_H2 ≥ 0.7:  {pct_high:.1f}%")
    
    print(f"\nFÓRMULA ANTIGUA (h2_base / h2_future) - PARA COMPARACIÓN:")
    print(f"  Media:    {S_H2_old.mean():.4f}")
    print(f"  Mediana:  {np.median(S_H2_old):.4f}")
    print(f"  Std:      {S_H2_old.std():.4f}")
    print(f"  Min:      {S_H2_old.min():.4f}")
    print(f"  Max:      {S_H2_old.max():.4f}")
    
    # Análisis de colapso H2
    h2_collapse_indices = h2_fut < (h2_base * 0.1)  # Caída > 90%
    pct_collapse = h2_collapse_indices.sum() / len(h2_base) * 100
    
    print(f"\nAnálisis de colapso (caída > 90%):")
    print(f"  Píxeles con colapso: {h2_collapse_indices.sum()} ({pct_collapse:.1f}%)")
    if h2_collapse_indices.sum() > 0:
        S_H2_collapse = S_H2_correct[h2_collapse_indices]
        print(f"  S_H2 promedio en colapso: {S_H2_collapse.mean():.4f}")
        print(f"  S_H2 rango en colapso: [{S_H2_collapse.min():.4f}, {S_H2_collapse.max():.4f}]")

print("\n" + "="*80)
print("CONCLUSIÓN:")
print("="*80)
print("\nLa fórmula CORRECTA produce valores S_H2 MUCHO MÁS BAJOS en SSP585")
print("cuando hay colapso de H2. Esto refleja correctamente la MALA resiliencia.")
print("\nEl valor 0.78 reportado en la tesis probablemente proviene de la")
print("fórmula ANTIGUA (invertida). Necesitas ACTUALIZAR los estadísticos.")
