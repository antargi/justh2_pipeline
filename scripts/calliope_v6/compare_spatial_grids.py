import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Cargar ambos datasets
rsds_file = "/home/aninotna/magister/tesis/justh2_pipeline/data/cmip6/rsds/bias_corrected_qm/rsds_qm_access_ssp245.nc"
indicators_file = "/home/aninotna/magister/tesis/justh2_pipeline/data/cmip6/indicators_ssp/climate_indicators_ACCESS-CM2_ssp245_2015-2100.nc"

ds_rsds = xr.open_dataset(rsds_file)
ds_indicators = xr.open_dataset(indicators_file)

# Crear grillas meshgrid
lat_rsds, lon_rsds = ds_rsds.lat.values, ds_rsds.lon.values
lat_ind, lon_ind = ds_indicators.lat.values, ds_indicators.lon.values

# Crear figura
fig, ax = plt.subplots(figsize=(12, 8))

# Plot grilla RSDS (400 puntos)
lon_mesh_rsds, lat_mesh_rsds = np.meshgrid(lon_rsds, lat_rsds)
ax.scatter(lon_mesh_rsds.flatten(), lat_mesh_rsds.flatten(), 
           c='blue', marker='o', s=100, alpha=0.6, label=f'RSDS (20×20 = {len(lat_rsds)*len(lon_rsds)} puntos)')

# Plot grilla Indicators (760 puntos)
lon_mesh_ind, lat_mesh_ind = np.meshgrid(lon_ind, lat_ind)
ax.scatter(lon_mesh_ind.flatten(), lat_mesh_ind.flatten(), 
           c='red', marker='x', s=50, alpha=0.8, label=f'Indicators SSP (20×38 = {len(lat_ind)*len(lon_ind)} puntos)')

# Configuración del gráfico
ax.set_xlabel('Longitud (°E)', fontsize=12)
ax.set_ylabel('Latitud (°N)', fontsize=12)
ax.set_title('Comparación espacial: Grillas RSDS vs Indicators SSP\nValle de Aconcagua', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Añadir información de rangos
textstr = f'''RSDS:
  Lat: {lat_rsds.min():.4f}° to {lat_rsds.max():.4f}°
  Lon: {lon_rsds.min():.4f}° to {lon_rsds.max():.4f}°
  Resolución lon: ~{np.diff(lon_rsds).mean():.4f}°

Indicators SSP:
  Lat: {lat_ind.min():.4f}° to {lat_ind.max():.4f}°
  Lon: {lon_ind.min():.4f}° to {lon_ind.max():.4f}°
  Resolución lon: ~{np.diff(lon_ind).mean():.4f}°'''

ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/aninotna/magister/tesis/justh2_pipeline/reports/comparacion_grillas_rsds_indicators.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado en: reports/comparacion_grillas_rsds_indicators.png")
plt.show()

# Análisis de superposición
print("\n" + "="*60)
print("ANÁLISIS DE SUPERPOSICIÓN ESPACIAL")
print("="*60)

lat_match = np.allclose(lat_rsds, lat_ind)
print(f"\n✓ Latitudes: {'✅ COINCIDEN EXACTAMENTE' if lat_match else '⚠️ NO COINCIDEN'}")
print(f"  RSDS:       [{lat_rsds.min():.4f}, {lat_rsds.max():.4f}] - {len(lat_rsds)} puntos")
print(f"  Indicators: [{lat_ind.min():.4f}, {lat_ind.max():.4f}] - {len(lat_ind)} puntos")

print(f"\n✓ Longitudes:")
print(f"  RSDS:       [{lon_rsds.min():.4f}, {lon_rsds.max():.4f}] - {len(lon_rsds)} puntos (res: ~{np.diff(lon_rsds).mean():.4f}°)")
print(f"  Indicators: [{lon_ind.min():.4f}, {lon_ind.max():.4f}] - {len(lon_ind)} puntos (res: ~{np.diff(lon_ind).mean():.4f}°)")

# Verificar si los puntos de RSDS están contenidos en Indicators
rsds_lons_in_indicators = np.isin(np.round(lon_rsds, 4), np.round(lon_ind, 4))
print(f"\n✓ Puntos RSDS contenidos en Indicators: {rsds_lons_in_indicators.sum()}/{len(lon_rsds)}")
if rsds_lons_in_indicators.sum() == len(lon_rsds):
    print("  ✅ Todos los puntos de RSDS están en la grilla de Indicators")
    print("  ➡️ Puedes usar directamente Indicators (760 puntos) e interpolar RSDS")
else:
    print(f"  ⚠️ Solo {rsds_lons_in_indicators.sum()} puntos coinciden")
    print("  ➡️ Necesitarás interpolar una grilla a la otra")
