import os
import requests

def download_mapbiomas_tifs(years, output_dir="data"):
    base_url = "https://storage.googleapis.com/mapbiomas-public/initiatives/chile/coverage/chile_coverage_{}.tif"
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        url = base_url.format(year)
        file_path = os.path.join(output_dir, f"chile_coverage_{year}.tif")
        
        print(f"Descargando {url}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Guardado en {file_path}")
        else:
            print(f"Error al descargar {url}: {response.status_code}")

years = list(range(2000, 2023))
download_mapbiomas_tifs(years, '/home/aninotna/magister/tesis/justh2_pipeline/data/mapbioma')