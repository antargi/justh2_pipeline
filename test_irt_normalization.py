"""
Script de verificación: ¿están todos los componentes del IRT normalizados a [0, 1]?
"""

import os
import sys
import pickle
import numpy as np
import glob

# Agregar directorio raíz al path
BASE_DIR = "/home/aninotna/magister/tesis/justh2_pipeline"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")

# Cargar datos del último experimento de clustering
trained_dir = os.path.join(DATA_DIR, "autoencoder_trained_v2")
pattern = os.path.join(trained_dir, "experiment1_clustering_*.pkl")
available_files = glob.glob(pattern)

if not available_files:
    raise FileNotFoundError(f"No se encontraron archivos en: {pattern}")

export_path = max(available_files, key=os.path.getmtime)
print(f"Cargando datos desde: {os.path.basename(export_path)}")

with open(export_path, "rb") as f:
    exp1_data = pickle.load(f)

# Buscar IRT en los datos cargados
if 'IRT_RESULTS' in exp1_data:
    print("\n✓ IRT_RESULTS encontrado en exp1_data")
    IRT_RESULTS = exp1_data['IRT_RESULTS']
    
    for model_key in IRT_RESULTS:
        for method in IRT_RESULTS[model_key]:
            irt_res = IRT_RESULTS[model_key][method]
            
            if irt_res is None or 'IRCT' not in irt_res:
                continue
                
            irt_values = irt_res['IRCT']
            
            print(f"\n{model_key} + {method}:")
            print(f"  Min: {irt_values.min():.6f}")
            print(f"  Max: {irt_values.max():.6f}")
            print(f"  Mean: {irt_values.mean():.6f}")
            print(f"  Std: {irt_values.std():.6f}")
            
            # Verificar si están en [0, 1]
            if irt_values.min() < 0:
                print(f"  ⚠️  WARNING: valores negativos encontrados!")
            if irt_values.max() > 1:
                print(f"  ⚠️  ERROR: valores > 1 encontrados! Max = {irt_values.max()}")
            else:
                print(f"  ✓ Todos los valores están en [0, 1]")
            
            # Verificar componentes individuales
            if 'A' in irt_res and irt_res['A'] is not None:
                A = irt_res['A']
                if A.max() > 1 or A.min() < 0:
                    print(f"    - A (Reconstruction): min={A.min():.4f}, max={A.max():.4f}")
            
            if 'S_D' in irt_res and irt_res['S_D'] is not None:
                S_D = irt_res['S_D']
                if S_D.max() > 1 or S_D.min() < 0:
                    print(f"    - S_D (Latent Displacement): min={S_D.min():.4f}, max={S_D.max():.4f}")
            
            if 'S_C' in irt_res and irt_res['S_C'] is not None:
                S_C = irt_res['S_C']
                if S_C.max() > 1 or S_C.min() < 0:
                    print(f"    - S_C (Cluster Stability): min={S_C.min():.4f}, max={S_C.max():.4f}")
            
            if 'S_E' in irt_res and irt_res['S_E'] is not None:
                S_E = irt_res['S_E']
                if S_E.max() > 1 or S_E.min() < 0:
                    print(f"    - S_E (Cluster Expansion): min={S_E.min():.4f}, max={S_E.max():.4f}")
                    
else:
    print("\n✗ No se encontró IRT_RESULTS en exp1_data")
    print(f"Claves disponibles: {list(exp1_data.keys())}")

print("\n" + "="*80)
print("Script de verificación completado")
print("="*80)
