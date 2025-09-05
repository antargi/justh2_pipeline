import xarray as xr

data_path = "/home/aninotna/magister/tesis/justh2_pipeline/data"

ds = xr.open_dataset(data_path+"/mapbioma/nc/uso_suelo_stack.nc")
print(ds)


# vista de un año
ds.uso_suelo.sel(year=2020).plot()
