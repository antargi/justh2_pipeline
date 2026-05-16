"""
Script de validación: recalcular IRT con el código arreglado y verificar normalización
"""

import os
import sys
import pickle
import numpy as np
import torch
import glob
import warnings

warnings.filterwarnings('ignore')

# Agregar directorio raíz al path
BASE_DIR = "/home/aninotna/magister/tesis/justh2_pipeline"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.idroverdi_autoencoder_3.resilience_lib import (
    compute_IRCT_from_clustering_results
)

DATA_DIR = os.path.join(BASE_DIR, "data")
trained_dir = os.path.join(DATA_DIR, "autoencoder_trained_v2")

# Cargar datos del último experimento
pattern = os.path.join(trained_dir, "experiment1_clustering_*.pkl")
available_files = glob.glob(pattern)

if not available_files:
    raise FileNotFoundError(f"No se encontraron archivos en: {pattern}")

export_path = max(available_files, key=os.path.getmtime)
print(f"Cargando datos desde: {os.path.basename(export_path)}\n")

with open(export_path, "rb") as f:
    exp1_data = pickle.load(f)

# Extraer información necesaria
LATENTS = exp1_data["LATENTS"]
N_PER_SCENARIO = exp1_data["N_PER_SCENARIO"]
MODEL_ORDER = exp1_data["MODEL_ORDER"]
CLUSTERING_RESULTS = exp1_data["CLUSTERING_RESULTS"]

X_BASE = exp1_data["X_BASE"]
X_BASE_norm = np.vstack([
    exp1_data["B245"],
    exp1_data["B370"],
    exp1_data["B585"]
])
X585_orig = exp1_data["X585_orig"]
X585_norm = exp1_data["T585"]

print(f"Datos cargados:")
print(f"  Modelos: {MODEL_ORDER}")
print(f"  X_BASE shape: {X_BASE.shape}")
print(f"  X585 shape: {X585_orig.shape}")
print(f"  Clustering results: {list(CLUSTERING_RESULTS.keys())}\n")

# Cargar modelos
from scripts.idroverdi_autoencoder_3.backup.resilience import AE, VAE

models_paths = exp1_data['models_path']
model_dims = exp1_data.get('model_dims', {})
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {}
for model_key, model_path in models_paths.items():
    if os.path.exists(model_path):
        print(f"Cargando modelo {model_key}...")
        
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        if model_key in model_dims:
            input_dim = model_dims[model_key]['input_dim']
            latent_dim = model_dims[model_key]['latent_dim']
        else:
            input_dim = X_BASE_norm.shape[1]
            latent_dim = exp1_data.get(f'LATENT_DIM_{model_key}', 8)
        
        if model_key == 'AE':
            model = AE(input_dim=input_dim, latent_dim=latent_dim)
        elif model_key == 'VAE':
            model = VAE(input_dim=input_dim, latent_dim=latent_dim)
        else:
            continue
        
        model.load_state_dict(state_dict)
        model.eval()
        MODELS[model_key] = model.to(DEVICE)
        print(f"  ✓ {model_key} cargado")

print(f"\nDevice: {DEVICE}\n")

# Buscar scaler
scaler = None
stack_pattern = os.path.join(DATA_DIR, "autoencoder_stack", "stack_*.pkl")
stack_files = glob.glob(stack_pattern)

if stack_files:
    stack_path = max(stack_files, key=os.path.getmtime)
    print(f"Cargando scaler desde: {os.path.basename(stack_path)}")
    
    with open(stack_path, "rb") as f:
        stack_data = pickle.load(f)
    
    scaler = stack_data.get("scaler")
    print(f"  ✓ Scaler cargado\n")

# Calcular IRT para cada modelo y método
print("="*80)
print("VALIDACIÓN DE IRT NORMALIZACIÓN")
print("="*80)

for model_key in MODEL_ORDER:
    if model_key not in MODELS:
        print(f"\nModelo {model_key} no disponible, saltando...")
        continue
    
    print(f"\n{model_key}:")
    print("-" * 80)
    
    model = MODELS[model_key]
    
    for method_key in CLUSTERING_RESULTS[model_key]:
        clustering_res = CLUSTERING_RESULTS[model_key][method_key]
        
        if clustering_res is None:
            print(f"  {method_key}: No hay resultados")
            continue
        
        try:
            # Calcular IRT
            irt_result = compute_IRCT_from_clustering_results(
                model=model,
                clustering_results=clustering_res,
                X_base_orig=X_BASE,
                X_base_norm=X_BASE_norm,
                X_future_orig=X585_orig,
                X_future_norm=X585_norm,
                scenario='T585',
                h2_base=None,
                h2_future=None,
                weights=None,
                device=DEVICE,
                recon_use_normalized=True,
                inverse_transform=scaler.inverse_transform if scaler else None,
                softmax_tau=1.0,
                expansion_p99_clip=False,
                h2_p99_clip=True,
                eps=1e-8
            )
            
            # Extraer componentes
            IRCT = irt_result['IRCT']
            A = irt_result.get('A', irt_result.get('reconstruction_anomaly'))
            S_D = irt_result.get('S_D', irt_result.get('latent_displacement'))
            S_C = irt_result.get('S_C', irt_result.get('cluster_stability'))
            S_E = irt_result.get('S_E', irt_result.get('cluster_expansion'))
            
            # Mostrar validación
            print(f"\n  {method_key}:")
            print(f"    IRCT:    min={IRCT.min():.6f}, max={IRCT.max():.6f}, mean={IRCT.mean():.6f}")
            
            # Validar que esté en [0, 1]
            if IRCT.min() < 0 or IRCT.max() > 1:
                print(f"    ⚠️  ERROR: IRCT fuera de rango [0, 1]!")
                if IRCT.max() > 1:
                    print(f"       → {(IRCT > 1).sum()} píxeles con IRCT > 1")
                if IRCT.min() < 0:
                    print(f"       → {(IRCT < 0).sum()} píxeles con IRCT < 0")
            else:
                print(f"    ✓ IRCT está correctamente normalizado")
            
            # Validar componentes
            print(f"\n    Componentes:")
            print(f"    A:    min={A.min():.6f}, max={A.max():.6f}", end="")
            if A.min() >= 0 and A.max() <= 1:
                print(" ✓")
            else:
                print(" ✗")
            
            print(f"    S_D:  min={S_D.min():.6f}, max={S_D.max():.6f}", end="")
            if S_D.min() >= 0 and S_D.max() <= 1:
                print(" ✓")
            else:
                print(" ✗")
            
            print(f"    S_C:  min={S_C.min():.6f}, max={S_C.max():.6f}", end="")
            if S_C.min() >= 0 and S_C.max() <= 1:
                print(" ✓")
            else:
                print(" ✗")
            
            print(f"    S_E:  min={S_E.min():.6f}, max={S_E.max():.6f}", end="")
            if S_E.min() >= 0 and S_E.max() <= 1:
                print(" ✓")
            else:
                print(" ✗ WARNING: S_E fuera de rango!")
            
        except Exception as e:
            print(f"  {method_key}: ERROR - {e}")
            import traceback
            traceback.print_exc()

print("\n" + "="*80)
print("Validación completada")
print("="*80)
