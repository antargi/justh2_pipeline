import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def load_geo_parquets_as_geodataframe(directory):
    files = glob.glob(os.path.join(directory, "*.geo.parquet"))
    print(f"📂 Found {len(files)} GeoParquet files")
    
    gdfs = []
    for i, file in enumerate(files, start=1):
        print(f'reading {i} - {file}')
        try:
            gdf = gpd.read_parquet(file)
            gdfs.append(gdf)
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
    
    combined = pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(combined, crs="EPSG:4326")
