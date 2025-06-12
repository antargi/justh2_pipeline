import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

def get_coordinates_from_zarr(zarr_path, lat_range=None, lon_range=None):
    ds = xr.open_zarr(zarr_path)
    if lat_range:
        ds = ds.sel(lat=slice(*lat_range))
    if lon_range:
        ds = ds.sel(lon=slice(*lon_range))
    return [(float(lat), float(lon)) for lat in ds.lat.values for lon in ds.lon.values]

def save_coords_to_csv(coords, output_path):
    df = pd.DataFrame(coords, columns=["lat", "lon"])
    df.to_csv(output_path, index=False)
    print(f"✅ Coordenadas guardadas en {output_path}")


def save_coords_to_geojson(coords, output_path):
    gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in coords], crs="EPSG:4326")
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"✅ Coordenadas guardadas en {output_path}")


def plot_coords(coords, title="Coordenadas del Valle de Aconcagua"):
    gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in coords], crs="EPSG:4326")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(ax=ax, color="orange", markersize=10)
    ax.set_title(title)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def read_coords_from_geojson(path):
    gdf = gpd.read_file(path)
    return [(point.y, point.x) for point in gdf.geometry]