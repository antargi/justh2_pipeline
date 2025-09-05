import os
import glob
import pandas as pd
from shapely import points
import geopandas as gpd 
from shapely.geometry import Point

def read_all_parquet(parquet_dir):
    archivos_parquet = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    lista = [pd.read_parquet(p) for p in archivos_parquet]
    return pd.concat(lista, ignore_index=True)

def process_csv_solar(path_csv):
    with open(path_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lat = float([l for l in lines if "LATITUD" in l][0].split(",")[1])
    lon = float([l for l in lines if "LONGITUD" in l][0].split(",")[1])
    alt = float([l for l in lines if "ALTURA" in l][0].split(",")[1])

    for i, line in enumerate(lines):
        if "Fecha/Hora" in line:
            header_idx = i
            break

    df = pd.read_csv(path_csv, skiprows=header_idx)
    df.columns = df.columns.str.strip()
    df["timestamp"] = pd.to_datetime(df["Fecha/Hora"], format="%Y-%m-%d %H:%M:%S")
    df.drop(columns=["Fecha/Hora"], inplace=True)

    df["lat"] = lat
    df["lon"] = lon
    df["alt"] = alt

    columnas_utiles = ["timestamp", "dni", "ghi", "temp", "vel", "cloud", "lat", "lon", "alt"]
    df = df[[col for col in columnas_utiles if col in df.columns]]

    return df

def process_and_save_batches(directorio, output_dir):
    archivos = glob.glob(os.path.join(directorio, "*.csv"))
    total_archivos = len(archivos)

    os.makedirs(output_dir, exist_ok=True)

    for i, archivo in enumerate(archivos, start=1):
        print(f'📂 Procesando archivo {i} de {total_archivos}: {archivo}')
        try:
            df = process_csv_solar(archivo)
            nombre_archivo = os.path.splitext(os.path.basename(archivo))[0]
            ruta_output = os.path.join(output_dir, f"{nombre_archivo}.parquet")
            df.to_parquet(ruta_output)
        except Exception as e:
            print(f"❌ Error en {archivo}: {e}")


def read_coords_from_geojson(geojson_path):
    gdf_coords = gpd.read_file(geojson_path)
    gdf_coords = gdf_coords.to_crs("EPSG:4326")  # asegurarse de que está en el mismo CRS
    coords_set = set((round(geom.y, 5), round(geom.x, 5)) for geom in gdf_coords.geometry)
    return coords_set

def points(lon, lat):
    return [Point(xy) for xy in zip(lon, lat)]


def process_parquet_to_geo_batches(input_dir, output_dir, coords_validas):
    os.makedirs(output_dir, exist_ok=True)
    archivos = glob.glob(os.path.join(input_dir, "*.parquet"))
    print(f"Se encontraron {len(archivos)} archivos 📁 ")

    count_filtrados = 0

    for i, ruta in enumerate(archivos, start=1):
        try:
            df = pd.read_parquet(ruta)
            lat, lon = round(df["lat"].iloc[0], 5), round(df["lon"].iloc[0], 5)

            if (lat, lon) not in coords_validas:
                continue  # filtrar

            geometry = points(df["lon"].to_numpy(), df["lat"].to_numpy())
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

            nombre = os.path.splitext(os.path.basename(ruta))[0]
            gdf.to_parquet(os.path.join(output_dir, f"{nombre}.geo.parquet"))
            count_filtrados += 1
            print(f"({count_filtrados}) Guardado: {nombre}.geo.parquet")

        except Exception as e:
            print(f"Error en {ruta}: {e}")

    print(f"✅ Total archivos guardados: {count_filtrados}")