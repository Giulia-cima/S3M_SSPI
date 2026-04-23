import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
import matplotlib.pyplot as plt
import glob


# -----------------------
# INPUT FILES
# -----------------------
basin_raster = "/home/idrologia/PhD_GiuliaBlandini/DAO_project/statici/Bacini_DAO_500m.tif"
legend_csv = "/home/idrologia/PhD_GiuliaBlandini/DAO_project/statici/Bacini_DAO.csv"
output_folder = "/home/idrologia/PhD_GiuliaBlandini/DAO_project/basin_shapefiles/"

os.makedirs(output_folder, exist_ok=True)

# -----------------------
# READ LEGEND
# -----------------------
legend_df = pd.read_csv(legend_csv)

# Mapping: ID -> Basin Name
id_to_name = dict(zip(legend_df['ID'], legend_df['Name']))

# -----------------------
# READ RASTER
# -----------------------
with rasterio.open(basin_raster) as src:
    basin_array = src.read(1).astype(np.int32)   # ✅ FIX HERE
    transform = src.transform
    crs = src.crs
    nodata = src.nodata

# -----------------------
# HANDLE NODATA
# -----------------------
if nodata is not None:
    mask = basin_array != nodata
else:
    mask = None

# Optional debug
print("Raster shape:", basin_array.shape)
print("Unique IDs (sample):", np.unique(basin_array)[:10])

# -----------------------
# EXTRACT POLYGONS
# -----------------------
print("Extracting polygons from raster...")

results = (
    {"properties": {"basin_id": int(v)}, "geometry": geom}
    for geom, v in shapes(basin_array, mask=mask, transform=transform)
)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame.from_features(list(results), crs=crs)

# -----------------------
# ADD BASIN NAMES
# -----------------------
gdf["basin_name"] = gdf["basin_id"].map(id_to_name)

# Remove invalid / nodata entries
gdf = gdf.dropna(subset=["basin_name"])

# -----------------------
# DISSOLVE (merge polygons per basin)
# -----------------------
print("Dissolving polygons by basin...")

gdf = gdf.dissolve(by="basin_id", as_index=False)

# Re-add basin names after dissolve
gdf["basin_name"] = gdf["basin_id"].map(id_to_name)

# -----------------------
# SAVE ONE SHAPEFILE PER BASIN
# -----------------------
print("Saving shapefiles...")

for _, row in gdf.iterrows():
    basin_id = row["basin_id"]
    basin_name = str(row["basin_name"])

    # Clean filename (very important for shapefiles)
    safe_name = basin_name.replace(" ", "_").replace("/", "_")

    out_path = os.path.join(
        output_folder,
        f"basin_{basin_id}_{safe_name}.shp"
    )

    single_gdf = gpd.GeoDataFrame([row], crs=crs)
    single_gdf.to_file(out_path)

print("✅ Done. Shapefiles saved in:", output_folder)


shp_dir = "/home/idrologia/PhD_GiuliaBlandini/DAO_project/basin_shapefiles/"

shapes = {}

for shp_path in glob.glob(os.path.join(shp_dir, "*.shp")):
    basin_name = os.path.basename(shp_path).replace(".shp", "")
    shapes[basin_name] = gpd.read_file(shp_path)
