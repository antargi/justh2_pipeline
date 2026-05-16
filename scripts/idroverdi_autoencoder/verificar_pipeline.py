#!/usr/bin/env python3
"""
Script de verificación del pipeline de preprocesamiento
Verifica que las celdas se puedan ejecutar en orden sin errores
"""

import sys
from pathlib import Path

BASE_DIR = Path("/home/aninotna/magister/tesis/justh2_pipeline")
OUT_DIR = BASE_DIR / "data/autoencoder_tensors"
MODE = "test"

print("=" * 80)
print("VERIFICACIÓN DEL PIPELINE DE PREPROCESAMIENTO")
print("=" * 80)

# 1. Verificar archivos ORIGINAL
print("\n1️⃣ Verificando archivos ORIGINAL...")
orig_files = list(OUT_DIR.glob("*_ORIGINAL.npz"))
if orig_files:
    print(f"   ❌ PROBLEMA: Encontrados {len(orig_files)} archivos ORIGINAL antiguos")
    print("   Estos archivos NO tienen las features std_T y causarán el error IndexError")
    print("\n   SOLUCIÓN: Eliminar archivos ORIGINAL antes de ejecutar el notebook:")
    print(f"   rm {OUT_DIR}/*_ORIGINAL.npz")
    sys.exit(1)
else:
    print("   ✅ OK: No hay archivos ORIGINAL antiguos")

# 2. Verificar archivos normalizados
print("\n2️⃣ Verificando archivos normalizados...")
import numpy as np

expected_features_with_std = None  # Lo calcularemos
for sc in ['ssp245', 'ssp370', 'ssp585']:
    file = OUT_DIR / f'tensors_{sc}_splits_{MODE}.npz'
    if file.exists():
        data = np.load(file)
        n_features = data['X_train'].shape[1]
        print(f"   {sc}: {n_features} features")
        if expected_features_with_std is None:
            expected_features_with_std = n_features

# 3. Verificar feature_names.csv
print("\n3️⃣ Verificando feature_names.csv...")
import pandas as pd

features_file = OUT_DIR / f'feature_names_{MODE}.csv'
if features_file.exists():
    df = pd.read_csv(features_file)
    n_features_csv = len(df)
    std_t_features = [f for f in df['feature_name'] if 'std_T' in f]
    
    print(f"   Total features: {n_features_csv}")
    print(f"   Features con std_T: {len(std_t_features)}")
    
    if len(std_t_features) == 0:
        print(f"   ❌ PROBLEMA: feature_names.csv NO tiene features std_T")
        print(f"   El notebook necesita ser re-ejecutado desde la celda de cálculo de std_T")
    else:
        print(f"   ✅ OK: feature_names.csv tiene features std_T")
        print(f"\n   Primeras 5 features std_T:")
        for f in std_t_features[:5]:
            print(f"      • {f}")
else:
    print("   ⚠️  feature_names.csv no existe")

# 4. Verificar metadata
print("\n4️⃣ Verificando metadata...")
import pickle

metadata_file = OUT_DIR / f'metadata_{MODE}.pkl'
if metadata_file.exists():
    with open(metadata_file, 'rb') as f:
        meta = pickle.load(f)
    
    scalers_source = meta.get('scalers_source', 'NO ENCONTRADO')
    scalers_fit_mode = meta.get('scalers_fit_mode', 'NO ENCONTRADO')
    x_base_rows = meta.get('x_base_rows', 'NO ENCONTRADO')
    
    print(f"   scalers_source: {scalers_source}")
    print(f"   scalers_fit_mode: {scalers_fit_mode}")
    print(f"   x_base_rows: {x_base_rows}")
    
    if scalers_source == 'X_BASE' and scalers_fit_mode == 'base':
        print("   ✅ OK: Metadata correcta")
    else:
        print("   ⚠️  Metadata puede estar desactualizada")
else:
    print("   ⚠️  metadata.pkl no existe")

# 5. Resumen
print("\n" + "=" * 80)
print("RESUMEN")
print("=" * 80)

print("\n📋 Para ejecutar el notebook correctamente:")
print("   1. ✅ Los archivos ORIGINAL ya fueron eliminados")
print("   2. 🔄 Reinicia el kernel del notebook (Kernel → Restart)")
print("   3. 🔄 Ejecuta las celdas EN ORDEN desde el principio:")
print("      • Celda 1-13: Carga de datos y creación de tensores")
print("      • Celda 14-15: ✨ CÁLCULO DE STD_T (nueva celda)")
print("      • Celda 16-19: Filtrado de features")
print("      • Celda 20-23: Normalización")
print("      • Celda 26-28: Splits y X_BASE")
print("      • Celda 29-31: Exportación")

print("\n⚠️  IMPORTANTE:")
print("   • NO ejecutar celdas individualmente fuera de orden")
print("   • Cuando llegues a la celda de X_BASE, los datos en memoria")
print("     YA tendrán las features std_T agregadas")
print("   • Los archivos ORIGINAL se crearán con std_T incluidas")

print("\n" + "=" * 80)
