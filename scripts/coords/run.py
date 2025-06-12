from define import save_coords_to_csv, save_coords_to_geojson, get_coordinates_from_zarr

path = "/home/aninotna/magister/tesis/justh2_pipeline/data/cr2met/clima.zarr"
coords = get_coordinates_from_zarr(path,
                                   lat_range=(-33.26809, -32.2621211),
                                   lon_range=(-71.89206, -70.0035719))
save_coords_to_csv(
    coords,
    "/home/aninotna/magister/tesis/justh2_pipeline/data/coords/valle_aconcagua_coords.csv"
)
save_coords_to_geojson(
    coords,
    "/home/aninotna/magister/tesis/justh2_pipeline/data/coords/valle_aconcagua_coords.geojson"
)