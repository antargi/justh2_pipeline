#!/usr/bin/env python3
"""
Script para regenerar gráficos de distribución de componentes del IRT.
Carga datos de CSVs exportados y genera histogramas con KDE.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glob import glob
import re
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Rutas
BASE_DIR = Path('/home/aninotna/magister/tesis/justh2_pipeline')
RESULTS_DIR = BASE_DIR / 'data' / 'autoencoder_results'
PLOTS_DIR = BASE_DIR / 'plots' / 'results'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("REGENERACIÓN DE GRÁFICOS: DISTRIBUCIÓN DE COMPONENTES DEL IRT")
print("="*80 + "\n")

# Buscar archivos más recientes de cada modelo-escenario
csv_pattern = str(RESULTS_DIR / 'IRCT_pixel_*.csv')
all_csv_files = sorted(glob(csv_pattern))

files_by_model_scenario = {}
for csv_file in all_csv_files:
    # Patrón: IRCT_pixel_{MODEL}_{SCENARIO}_{TIMESTAMP}.csv
    # donde TIMESTAMP es YYYYMMDD_HHMMSS
    match = re.search(r'IRCT_pixel_([A-Z]+)_(SSP\d+)_(\d{8}_\d{6})\.csv', csv_file)
    if match:
        model_key = match.group(1)  # VAE o AE
        scenario = match.group(2)   # SSP245, SSP370, SSP585
        timestamp = match.group(3)  # timestamp numérico
        
        key = (model_key, scenario)
        if key not in files_by_model_scenario or timestamp > files_by_model_scenario[key][1]:
            files_by_model_scenario[key] = (csv_file, timestamp)

# Cargar datos
IRCT_DATA = {}
print(f"Directorio de resultados: {RESULTS_DIR}")
print(f"Cargando {len(files_by_model_scenario)} archivos (más recientes)\n")

for (model_key, scenario), (csv_file, timestamp) in sorted(files_by_model_scenario.items()):
    try:
        df = pd.read_csv(csv_file)
        
        if model_key not in IRCT_DATA:
            IRCT_DATA[model_key] = {}
        
        IRCT_DATA[model_key][scenario] = {
            'A': df['A_reconstruction'].values,
            'S_D': df['S_D_displacement'].values,
            'S_C': df['S_C_stability'].values,
            'S_E': df['S_E_expansion'].values,
            'S_H2': df['S_H2_energy'].values,
            'IRCT': df['IRCT'].values,
        }
        print(f"  ✓ {model_key} — {scenario}: {len(df)} píxeles")
    except Exception as e:
        print(f"  ✗ {model_key} — {scenario}: ERROR - {e}")

print()

def plot_irt_component_distributions(
    irct_components_dict,
    model_key='VAE',
    scenario='SSP585',
    figsize=(20, 12),
    save_path=None,
    dpi=300
):
    """
    Grafica histogramas + KDE de los 5 componentes del IRT.
    """
    
    components_info = [
        ('A', 'Anomalía de Reconstrucción', 'viridis'),
        ('S_D', 'Desplazamiento Latente', 'plasma'),
        ('S_C', 'Estabilidad de Cluster', 'cividis'),
        ('S_E', 'Expansión de Cluster', 'twilight'),
        ('S_H2', 'Estabilidad H₂', 'magma')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (comp_key, comp_title, cmap) in enumerate(components_info):
        ax = axes[idx]
        
        if comp_key not in irct_components_dict or irct_components_dict[comp_key] is None:
            ax.text(0.5, 0.5, f'Sin datos: {comp_key}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_axis_off()
            continue
        
        comp_vals = irct_components_dict[comp_key]
        comp_vals_clean = comp_vals[~(np.isnan(comp_vals) | np.isinf(comp_vals))]
        
        if len(comp_vals_clean) == 0:
            ax.text(0.5, 0.5, f'Sin datos válidos: {comp_key}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_axis_off()
            continue
        
        # Histograma normalizado
        ax.hist(comp_vals_clean, bins=50, density=True, 
               alpha=0.6, color='steelblue', edgecolor='black', linewidth=0.5,
               label='Histograma')
        
        # KDE
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(comp_vals_clean)
            x_range = np.linspace(comp_vals_clean.min(), comp_vals_clean.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2.5, label='KDE')
        except Exception as e:
            print(f"  ⚠ Error KDE {comp_key}: {e}")
        
        # Estadísticas
        mean_val = comp_vals_clean.mean()
        median_val = np.median(comp_vals_clean)
        std_val = comp_vals_clean.std()
        
        # Líneas de referencia
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, 
                  label=f'Media: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
                  label=f'Mediana: {median_val:.3f}')
        
        # Formato
        ax.set_xlabel(f'{comp_key} (Percentil Rank)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Densidad', fontsize=11, fontweight='bold')
        ax.set_title(f'{comp_title}\n(n={len(comp_vals_clean)}, σ={std_val:.3f})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(0, 1)
    
    # Panel 6: info
    ax = axes[5]
    ax.axis('off')
    
    info_text = f"""
    DISTRIBUCIONES DE COMPONENTES DEL IRT
    
    Modelo: {model_key}
    Escenario: {scenario}
    Período: 2090-2100
    
    COMPONENTES:
    • A: Anomalía de reconstrucción
      (error del autoencoder)
    
    • S_D: Desplazamiento latente
      (drift en espacio latente)
    
    • S_C: Estabilidad de cluster
      (retención = bueno, collapse = malo)
    
    • S_E: Expansión de cluster
      (compacidad espacial)
    
    • S_H₂: Estabilidad energética
      (producción H₂ futuro/base)
    
    Normalización: Percentile Rank [0,1]
    """
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle(
        f'Distribución Empírica de los Cinco Componentes del IRT\n'
        f'{model_key} — {scenario}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        return True
    
    return False


# Generar gráficos
print("GENERANDO GRÁFICOS:")
print("-" * 80)

for model_key in ['VAE', 'AE']:
    print(f"\n{model_key}:")
    
    if model_key not in IRCT_DATA:
        print(f"  ✗ {model_key} no encontrado en IRCT_DATA")
        print(f"     Disponibles: {list(IRCT_DATA.keys())}")
        continue
    
    if 'SSP585' not in IRCT_DATA[model_key]:
        print(f"  ✗ SSP585 no encontrado para {model_key}")
        print(f"     Escenarios disponibles: {list(IRCT_DATA[model_key].keys())}")
        continue
    
    irct_result = IRCT_DATA[model_key]['SSP585']
    
    save_filename = f"irt_component_distributions_{model_key}.png"
    save_path = PLOTS_DIR / save_filename
    
    try:
        plot_irt_component_distributions(
            irct_result,
            model_key=model_key,
            scenario='SSP585',
            figsize=(20, 12),
            save_path=str(save_path),
            dpi=300
        )
        
        if save_path.exists():
            file_size_mb = save_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Guardado: {save_filename} ({file_size_mb:.2f} MB)")
        else:
            print(f"  ✗ Error: archivo no guardado")
    except Exception as e:
        print(f"  ✗ Error generando gráfico: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("RESUMEN ESTADÍSTICO - SSP585")
print("="*80 + "\n")

for model_key in ['VAE', 'AE']:
    if model_key not in IRCT_DATA or 'SSP585' not in IRCT_DATA[model_key]:
        continue
    
    print(f"\n{model_key}:")
    print("-" * 60)
    
    irct_result = IRCT_DATA[model_key]['SSP585']
    
    for comp in ['A', 'S_D', 'S_C', 'S_E', 'S_H2']:
        if comp not in irct_result or irct_result[comp] is None:
            continue
        
        vals = irct_result[comp]
        vals_clean = vals[~(np.isnan(vals) | np.isinf(vals))]
        
        mean_val = vals_clean.mean()
        median_val = np.median(vals_clean)
        std_val = vals_clean.std()
        
        print(f"  {comp}: μ={mean_val:.4f}, median={median_val:.4f}, σ={std_val:.4f}")
        
        if comp == 'S_H2':
            collapse_mask = vals_clean < 0.1
            pct_collapse = (collapse_mask.sum() / len(vals_clean)) * 100
            print(f"    → Colapso H₂ (S_H2 < 0.1): {pct_collapse:.1f}%")

print("\n" + "="*80)
print("✓ GRÁFICOS GENERADOS EXITOSAMENTE")
print("="*80 + "\n")

# Verificar archivos
print("Archivos en directorio de salida:")
for f in sorted(PLOTS_DIR.glob('irt_component_distributions_*.png')):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  ✓ {f.name} ({size_mb:.2f} MB)")
