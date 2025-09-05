import os
import rasterio
import numpy as np
import xarray as xr

data_path = "/home/aninotna/magister/tesis/justh2_pipeline/data"
input_folder = data_path + "/mapbioma/recortes"
years = list(range(2000, 2023))
arrays = []
coords_set = False


for year in years:
    path = os.path.join(input_folder, f"chile_coverage_{year}_crop.tif")
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.int16)
        transform = src.transform
        if not coords_set:
            lon = np.arange(src.width) * transform.a + transform.c
            lat = np.arange(src.height) * transform.e + transform.f
            coords_set = True
    arrays.append(data)

stack = np.stack(arrays, axis=0)  # shape: (year, lat, lon)


da = xr.DataArray(
    stack,
    dims=["year", "lat", "lon"],
    coords={"year": years, "lat": lat, "lon": lon},
    name="uso_suelo"
)


da.to_netcdf(data_path+"/mapbioma/nc/uso_suelo_stack.nc")