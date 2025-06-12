import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.coords.define import read_coords_from_geojson

import glob
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import xarray as xr


data_path = "/home/aninotna/magister/tesis/justh2_pipeline/data"
solar_path = data_path+"/solar/solar_diario_espacio_tiempo.geo.parquet"
output_dir = os.path.join(data_path, "solar", "gdf_por_lote")
coords = read_coords_from_geojson(os.path.join(data_path, "coords", "valle_aconcagua_coords.geojson"))
path_solar_daily = data_path + "/solar/solar_diario_continental.geo.parquet"


df = pd.read_parquet(path_solar_daily)
df["date"] = pd.to_datetime(df["date"])

df["lat"] = df["lat"].round(3)
df["lon"] = df["lon"].round(3)

ds = df.pivot_table(
    index=["date", "lat", "lon"],
    values=["ghi", "dni", "cloud", "temp", "vel"]
).reset_index()

ds_xr = ds.set_index(["date", "lat", "lon"]).to_xarray()

ds_xr.to_zarr(data_path + "/solar/solar_diario_grilla.zarr", mode="w")