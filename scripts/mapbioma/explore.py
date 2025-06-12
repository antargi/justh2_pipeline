import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import os

def crop_tif_to_geojson(input_tif, geojson_path, output_path):
    # Leer coordenadas del GeoJSON
    gdf = gpd.read_file(geojson_path)
    geoms = gdf.geometry.values
    geoms_json = [geom.__geo_interface__ for geom in geoms]

    # Abrir el tif y aplicar recorte
    with rasterio.open(input_tif) as src:
        out_image, out_transform = mask(src, geoms_json, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"✅ Recorte guardado: {output_path}")

# Uso
data_path = "/home/aninotna/magister/tesis/justh2_pipeline/data"
geojson_path = os.path.join(data_path, "coords", "valle_aconcagua_coords.geojson")

input_tif = os.path.join(data_path, "mapbioma", "chile_coverage_2020.tif")
output_tif = os.path.join(data_path, "mapbioma", "chile_coverage_2020_crop.tif")

crop_tif_to_geojson(input_tif, geojson_path, output_tif)