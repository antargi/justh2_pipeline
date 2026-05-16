"""
Test para verificar que compute_IRCT_pixel_wise devuelve valores en [0, 1]
"""

import os
import sys
import numpy as np

BASE_DIR = "/home/aninotna/magister/tesis/justh2_pipeline"
sys.path.insert(0, BASE_DIR)

from scripts.idroverdi_autoencoder_3.resilience_lib import compute_IRCT_pixel_wise

print("Función compute_IRCT_pixel_wise importada exitosamente")
print(f"Ubicación del módulo: {compute_IRCT_pixel_wise.__module__}")

# Crear datos de prueba
n_samples = 100
n_features = 29
latent_dim = 8
n_clusters = 3

X_base = np.random.randn(n_samples, n_features)
X_future = np.random.randn(n_samples, n_features)
z_base = np.random.randn(n_samples, latent_dim)
z_future = np.random.randn(n_samples, latent_dim)
centroids = np.random.randn(n_clusters, latent_dim)
labels = np.random.randint(0, n_clusters, n_samples)

# Crear modelo dummy
class DummyModel:
    def eval(self):
        pass
    
    def __call__(self, x):
        import torch
        return torch.randn_like(x), None
    
    def encode(self, x):
        import torch
        return torch.randn(x.shape[0], 8), torch.randn(x.shape[0], 8)
    
    def reparam(self, mu, logvar):
        import torch
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
    
    def dec(self, z):
        import torch
        return torch.randn(z.shape[0], 29)

model = DummyModel()

print("\nCalculando IRCT con datos de prueba...")

try:
    result = compute_IRCT_pixel_wise(
        model=model,
        X_base_orig=X_base,
        X_base_norm=X_base,
        X_future_orig=X_future,
        X_future_norm=X_future,
        z_base_scaled=z_base,
        z_future_scaled=z_future,
        centroids_base=centroids,
        labels_base=labels,
        device='cpu'
    )
    
    irct = result['IRCT']
    
    print(f"\nResultados:")
    print(f"  IRCT min: {irct.min():.6f}")
    print(f"  IRCT max: {irct.max():.6f}")
    print(f"  IRCT mean: {irct.mean():.6f}")
    print(f"  IRCT std: {irct.std():.6f}")
    
    if irct.min() >= 0 and irct.max() <= 1:
        print("\n✓ ¡¡CORRECTO!! - IRCT está en rango [0, 1]")
    else:
        print(f"\n✗ ERROR - IRCT está FUERA del rango [0, 1]")
        print(f"  Mín aceptable: 0, obtenido: {irct.min()}")
        print(f"  Máx aceptable: 1, obtenido: {irct.max()}")
    
    # Verificar componentes
    print(f"\nComponentes individuales:")
    for comp_name in ['reconstruction_anomaly', 'latent_displacement', 'cluster_stability', 'cluster_expansion']:
        if comp_name in result:
            comp = result[comp_name]
            if comp is not None:
                print(f"  {comp_name}: [{comp.min():.4f}, {comp.max():.4f}]")
                if comp.min() < 0 or comp.max() > 1:
                    print(f"    ⚠️  FUERA DE RANGO")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
