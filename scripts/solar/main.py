import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.coords.define import read_coords_from_geojson

import glob
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point


data_path = "/home/aninotna/magister/tesis/justh2_pipeline/data"
solar_path = data_path+"/solar/solar_diario_espacio_tiempo.geo.parquet"
output_dir = os.path.join(data_path, "solar", "gdf_por_lote")
coords = read_coords_from_geojson(os.path.join(data_path, "coords", "valle_aconcagua_coords.geojson"))
path_solar_daily = data_path + "/solar/solar_diario_continental.geo.parquet"

# coords_validas = set((round(lat, 5), round(lon, 5)) for lat, lon in coords)
# print(f"✅ Total coords esperadas: {len(coords_validas)}")  # Ej: 760


# archivos = glob.glob(os.path.join(output_dir, "*.geo.parquet"))

# coords_procesadas = set()

# for ruta in archivos:
#     df = pd.read_parquet(ruta)
#     lat, lon = round(df["lat"].iloc[0], 5), round(df["lon"].iloc[0], 5)
#     coords_procesadas.add((lat, lon))

# print(f"✅ Archivos únicos procesados: {len(coords_procesadas)}")

# coords_faltantes = coords_validas - coords_procesadas
# print(f"Coordenadas faltantes: {len(coords_faltantes)}")





# Parte 1: descarga
# from download import pipeline_solar_explorator
# pipeline_solar_explorator(
#     coords=coords_faltantes,
#     download_dir=data_path+"/solar/raw",
#     sleep_time=5
# )


# # Parte 2: procesar por lotes o guardar en disco por etapas
# from parse import process_and_save_batches
# output_dir="/home/aninotna/magister/tesis/justh2_pipeline/data/solar/lotes_solares"
# process_and_save_batches(
#     directorio=data_path+"/solar/raw",
#     output_dir=output_dir
# )


# Parte 3: pasar a geo batchs

# from parse import process_parquet_to_geo_batches

# input_dir = os.path.join(data_path, "solar", "lotes_solares")
# output_dir = os.path.join(data_path, "solar", "gdf_por_lote")
# process_parquet_to_geo_batches(input_dir, output_dir, coords)




# # Parte 4: Batches a 1 archivo
# import geopandas as gpd 

# registros = []

# for i, f in enumerate(glob.glob(data_path+"/solar/gdf_por_lote/*.geo.parquet")):
#     print(f'archivo {i} - {f}')
#     gdf = gpd.read_parquet(f)

# #    Extraer ubicación (constante por estación)
#     lat = gdf["lat"].iloc[0]
#     lon = gdf["lon"].iloc[0]
#     alt = gdf["alt"].iloc[0]

# #    Agrupar por fecha (sin hora)
#     gdf["date"] = gdf["timestamp"].dt.date
#     daily = gdf.groupby("date")[["ghi", "dni", "vel", "temp", "cloud"]].mean().reset_index()

# #    Agregar lat/lon/alt a cada fila
#     daily["lat"] = lat
#     daily["lon"] = lon
#     daily["alt"] = alt

#     registros.append(daily)

# #  Unir todos los registros
# df_diario = pd.concat(registros, ignore_index=True)

# #  Crear GeoDataFrame
# df_diario["geometry"] = df_diario.apply(lambda r: Point(r["lon"], r["lat"]), axis=1)
# gdf_diario = gpd.GeoDataFrame(df_diario, geometry="geometry", crs="EPSG:4326")

# gdf_ghi_cero = gdf_diario[gdf_diario["ghi"] == 0]
# print("Promedio de nubes en días con GHI = 0:", gdf_ghi_cero["cloud"].mean())

# gdf_diario.to_parquet(solar_path)

# gdf_diario.plot.scatter(x="cloud", y="ghi", alpha=0.2, figsize=(10, 6))
# plt.title("Relación entre nubosidad y GHI diario")
# plt.xlabel("Nubosidad (cloud)")
# plt.ylabel("GHI (W/m²)")
# plt.grid(True)
# plt.show()

# Parte 5 Exploración de la data compactada
# import geopandas as gpd 
# gdf = gpd.read_parquet(solar_path)

# print(gdf.head())

# print("\nColumnas:", gdf.columns.tolist())
# print("\nTipos de datos:\n", gdf.dtypes)

# n_estaciones = gdf[["lat", "lon"]].drop_duplicates().shape[0]
# print(f"\n📌 Número de estaciones: {n_estaciones}")

# print("📆 Fechas:", gdf["date"].min(), "→", gdf["date"].max())
# print("\n🔍 Valores nulos por columna:\n", gdf.isnull().sum())
# print("\n📊 Estadísticas generales:\n", gdf[["ghi", "dni", "temp", "vel"]].describe())

# gdf["ghi"].hist(bins=50, figsize=(8, 5), color="orange")
# plt.title("Distribución de GHI diario")
# plt.xlabel("GHI (W/m²)")
# plt.ylabel("Frecuencia")
# plt.grid(True)
# plt.show()

# ghi_diario = gdf.groupby("date")["ghi"].mean()

# ghi_diario.plot(figsize=(14, 5), title="Promedio diario de GHI en todas las estaciones")
# plt.ylabel("GHI promedio (W/m²)")
# plt.xlabel("Fecha")
# plt.grid(True)
# plt.show()


# Parte 6 Realizar mapa interativo
# import random
# import folium
# from folium import Choropleth, CircleMarker
# import geopandas as gpd 

# gdf = gpd.read_parquet(solar_path)


# fechas_validas = sorted(gdf["date"].unique())
# fecha_objetivo = random.choice(fechas_validas)
# print("📅 Fecha seleccionada aleatoriamente:", fecha_objetivo)

# gdf_dia = gdf[gdf["date"] == fecha_objetivo].copy()
# print("🔍 Filas con lat/lon nulas:", gdf_dia[["lat", "lon"]].isnull().any(axis=1).sum())
# gdf_dia = gdf_dia.dropna(subset=["lat", "lon"])

# if gdf_dia.empty:
#     print(f"⚠️ No hay datos para la fecha {fecha_objetivo}")
# else:
#     center = [gdf_dia["lat"].mean(), gdf_dia["lon"].mean()]
#     m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

#     for _, row in gdf_dia.iterrows():
#         folium.CircleMarker(
#             location=[row["lat"], row["lon"]],
#             radius=5,
#             color=None,
#             fill=True,
#             fill_color="orange",
#             fill_opacity=0.7,
#             popup=folium.Popup(
#                 f"<b>{row['date']}</b><br>"
#                 f"GHI: {row['ghi']:.1f} W/m²<br>"
#                 f"DNI: {row['dni']:.1f} W/m²<br>"
#                 f"Temp: {row['temp']:.1f} °C",
#                 max_width=250
#             )
#         ).add_to(m)

#     m.save(data_path+"/solar/mapa_interactivo_diario.html")
#     print(f"✅ Mapa para el día {fecha_objetivo} guardado como output/mapa_interactivo_diario.html")


#Parte 7 análisis de nubes
# import folium

# # Cargar el archivo con nube incluida
# gdf = gpd.read_parquet("output/solar_diario_espacio_tiempo.geo.parquet")

# # Filtrar: GHI = 0 y Cloud = 0
# gdf_inconsistente = gdf[(gdf["ghi"] == 0.0) & (gdf["cloud"] == 0.0)].copy()
# print(f"🔎 Filas con GHI=0 y Cloud=0: {len(gdf_inconsistente)}")

# # Quitar duplicados por estación si quieres ver solo una vez cada una
# gdf_inconsistente = gdf_inconsistente.drop_duplicates(subset=["lat", "lon"])

# # Crear mapa centrado
# center = [gdf_inconsistente["lat"].mean(), gdf_inconsistente["lon"].mean()]
# m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

# # Añadir estaciones como círculos
# for _, row in gdf_inconsistente.iterrows():
#     folium.CircleMarker(
#         location=[row["lat"], row["lon"]],
#         radius=6,
#         color="red",
#         fill=True,
#         fill_color="red",
#         fill_opacity=0.7,
#         popup=folium.Popup(
#             f"<b>Fecha:</b> {row['date']}<br>"
#             f"GHI: {row['ghi']}<br>"
#             f"Cloud: {row['cloud']}<br>"
#             f"Temp: {row['temp']}°C",
#             max_width=250
#         )
#     ).add_to(m)

# # Guardar mapa
# m.save("output/mapa_estaciones_inconsistentes.html")
# print("✅ Mapa guardado en output/mapa_estaciones_inconsistentes.html")

# Parte 8 quitar los puntos que están en el mar, porque tenían radiación 0
# gdf_diario = gpd.read_parquet("output/solar_diario_espacio_tiempo.geo.parquet")
# condicion_inconsistente = (gdf_diario["ghi"] == 0.0) & (gdf_diario["cloud"] == 0.0)
# gdf_diario = gdf_diario[~condicion_inconsistente].copy()
# print(f"✅ Registros después de eliminar inconsistencias: {len(gdf_diario)}")

# # # Guardar dataset limpio
# gdf_diario.to_parquet("output/solar_diario_continental.geo.parquet")

# print(f"✅ Registros diarios en Región de Valparaíso: {len(gdf_diario)}")


# Parte 9 revisar que haya quedado bien
# print("🔎 Columnas:", gdf_diario.columns.tolist())
# print("\n📊 Tipos de datos:")
# print(gdf_diario.dtypes)
# print("\n📌 Número de estaciones únicas:", gdf_diario[["lat", "lon"]].drop_duplicates().shape[0])
# print("📆 Rango de fechas:", gdf_diario["date"].min(), "→", gdf_diario["date"].max())
# print("\n🧼 Valores nulos por columna:")
# print(gdf_diario.isna().sum())

# # Estadísticas generales
# print("\n📈 Estadísticas descriptivas:")
# print(gdf_diario[["ghi", "dni", "cloud", "vel", "temp"]].describe())

# # Plot rápido: puntos válidos en el mapa
# gdf_diario.plot(column="ghi", cmap="YlOrRd", markersize=1, legend=True, figsize=(10, 8))
# plt.title("Mapa de puntos válidos (filtrados)")
# plt.xlabel("Longitud")
# plt.ylabel("Latitud")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#Parte 10 revisar estadísticas 

# Cargar el dataset limpio
# gdf = gpd.read_parquet("output/solar_diario_continental.geo.parquet")

# # Mostrar columnas
# print("📌 Columnas:", gdf.columns.tolist())

# # Tipos de datos
# print("\n🧾 Tipos de datos:")
# print(gdf.dtypes)

# # Rango de fechas
# print("\n📆 Rango de fechas:", gdf['date'].min(), "→", gdf['date'].max())

# # Número de estaciones únicas
# n_estaciones = gdf[["lat", "lon"]].drop_duplicates().shape[0]
# print(f"\n📍 Estaciones únicas: {n_estaciones}")

# # Valores nulos por columna
# print("\n🧼 Nulos por columna:")
# print(gdf.isna().sum())

# # Estadísticas descriptivas de variables clave
# print("\n📊 Estadísticas generales:")
# print(gdf[["ghi", "dni", "cloud", "vel", "temp"]].describe())

#Parte 10 revisar gráficas

# gdf = gpd.read_parquet("output/solar_diario_continental.geo.parquet")

# # Verificar fechas con GHI mínimo
# ghi_min = gdf["ghi"].min()
# ghi_min_rows = gdf[gdf["ghi"] == ghi_min][["date", "ghi", "cloud", "lat", "lon"]]
# print("🔍 Fechas con GHI mínimo:")
# print(ghi_min_rows.head())

# # Graficar histogramas
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# axes[0].hist(gdf["cloud"].dropna(), bins=30)
# axes[0].set_title("Distribución de Cloud")
# axes[0].set_xlabel("Cloud (fracción)")
# axes[0].set_ylabel("Frecuencia")

# axes[1].hist(gdf["ghi"].dropna(), bins=30)
# axes[1].set_title("Distribución de GHI")
# axes[1].set_xlabel("GHI (W/m²)")

# axes[2].hist(gdf["dni"].dropna(), bins=30)
# axes[2].set_title("Distribución de DNI")
# axes[2].set_xlabel("DNI (W/m²)")

# plt.tight_layout()
# plt.show()