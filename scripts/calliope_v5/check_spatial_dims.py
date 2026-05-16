import xarray as xr
import numpy as np

# Revisar dimensiones espaciales de los datos RSDS
file = "/home/aninotna/magister/tesis/justh2_pipeline/data/cmip6/rsds/bias_corrected_qm/rsds_qm_access_ssp245.nc"
ds = xr.open_dataset(file)

print("=== Dimensiones del dataset ===")
print(ds.dims)
print("\n=== Coordenadas ===")
print(f"Lat: {ds.lat.min().values:.4f} to {ds.lat.max().values:.4f} ({len(ds.lat)} puntos)")
print(f"Lon: {ds.lon.min().values:.4f} to {ds.lon.max().values:.4f} ({len(ds.lon)} puntos)")
print(f"Time: {ds.time.min().values} to {ds.time.max().values} ({len(ds.time)} pasos)")

print(f"\n=== Total de puntos espaciales: {len(ds.lat) * len(ds.lon)} ===")

print("\n=== Variables en el dataset ===")
print(list(ds.data_vars))
