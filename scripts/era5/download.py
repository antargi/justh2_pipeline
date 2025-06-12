import cdsapi
import os

def download_data_5(output_dir: str, years: list, area: list):
    os.makedirs(output_dir, exist_ok=True)
    client = cdsapi.Client()

    for year in years:
        if os.path.exists(output_dir):
            print(f"Datos para {year} ya descargados")
            continue
        print(f"Descargando datos para {year}")
        
        settings = {
            "product_type": "reanalysis",
            
        }
        client.retrieve()