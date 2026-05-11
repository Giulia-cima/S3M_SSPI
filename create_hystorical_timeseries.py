import os
import re
import logging
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from datetime import datetime
import warnings

warnings.filterwarnings(
    "ignore",
    message="Geometry is in a geographic CRS.*"
)
pattern = re.compile(r"ITSNOW500-(SWE|HS)_(\d{14})\.tif")

def extract_timestamp(fname):
    m = pattern.search(fname)
    if m:
        return datetime.strptime(m.group(2), "%Y%m%d%H%M%S")
    return None
# ---------------------------
# CONFIG
# ---------------------------
basin_folder = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/basin_shapefiles/"
reanalysis_folder ="/share/idrologia/s3m-italy/SWE_500m_national_reanalysis/"
intermediate_pkl = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/timeseries/basin_snow_timeseries_intermediate.pkl"
output_pkl = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/timeseries/basin_dao_timeseries.pkl"
log_file = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/timeseries/create_timeseries.log"

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    filename=log_file,
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Script started")

# ---------------------------
# LOAD BASINS
# ---------------------------
# make a list of all shapefiles in the basin folder
shapefiles = [f for f in os.listdir(basin_folder) if f.endswith(".shp")]

basins = {}

for shp_file in shapefiles:
    full_path = os.path.join(basin_folder, shp_file)
    gdf = gpd.read_file(full_path)

    # extract basin name from filename
    # expected: basin_12_Adige.shp
    name_no_ext = os.path.splitext(shp_file)[0]

    parts = name_no_ext.split("_", 2)
    if len(parts) >= 3:
        basin_id = int(parts[1])
        basin_name = parts[2]
    else:
        basin_id = -1
        basin_name = name_no_ext

    basins[basin_name] = gdf

logging.info(f"Loaded {len(basins)} basins")

# ---------------------------
# TIME RANGE (optional external)
# ---------------------------
# Example: you already have `period`
period = pd.date_range("2010-09-01", "2025-08-31", freq="D")

# ---------------------------
# OUTPUT STORAGE
# ---------------------------
results = []
basins_proj = None
pattern = re.compile(r"ITSNOW500-(SWE|HS)_(\d{14})\.tif")

# ---------------------------
# MAIN LOOP
# ---------------------------
for i, date in enumerate(period, start=1):

    folder = os.path.join(
        reanalysis_folder,
        date.strftime("%Y"),
        date.strftime("%m"),
        date.strftime("%d")
    )
    # file is named like: ITSNOW500-SWE_20100901110000.tif always at 11:00:00 UTC
    swe_file = f"ITSNOW500-SWE_{date.strftime('%Y%m%d')}110000.tif"
    hs_file = f"ITSNOW500-HS_{date.strftime('%Y%m%d')}110000.tif"

    swe_path = os.path.join(folder, swe_file)
    hs_path = os.path.join(folder, hs_file)


    # ---------------------------
    # missing files
    # ---------------------------
    if not os.path.exists(swe_path) or not os.path.exists(hs_path):
        for basin_name in basins:
            results.append({
                "date": date,
                "basin": basin_name,
                "SWE": 0.0,
                "HS": 0.0
            })
        logging.warning(f"Missing file at {date}")
        continue

    # ---------------------------
    # process rasters
    # ---------------------------
    try:
        with rasterio.open(swe_path) as swe_src, rasterio.open(hs_path) as hs_src:


            nodata_swe = swe_src.nodata
            nodata_hs = hs_src.nodata

            for basin_name, gdf in basins.items():
                geoms = [geom for geom in gdf.geometry if geom is not None]
                # suppress warnings about empty geometries

                cell_area = gdf.geometry.area.mean() if not gdf.empty else 0.0

                try:
                    swe_img, _ = mask(swe_src, geoms, crop=True, all_touched=True)
                    hs_img, _ = mask(hs_src, geoms, crop=True, all_touched=True)

                    swe = swe_img[0]
                    hs = hs_img[0]

                    if nodata_swe is not None:
                        swe[swe == nodata_swe] = 0.0
                    if nodata_hs is not None:
                        hs[hs == nodata_hs] = 0.0

                    swe = swe.astype("float32")
                    hs = hs.astype("float32")

                    swe = (swe/1000) * cell_area
                    hs = (hs/1000) * cell_area

                    swe_sum = np.nansum(swe)
                    hs_sum = np.nansum(hs)
                except ValueError:
                    swe_sum, hs_sum = 0.0, 0.0

                results.append({
                    "date": date,
                    "basin": basin_name,
                    "SWE": swe_sum,
                    "HS": hs_sum
                })

        logging.info(f"Processed {date}")

    except Exception as e:
        logging.error(f"Error processing {date}: {e}")

        for basin_name in basins:
            results.append({
                "date": date,
                "basin": basin_name,
                "SWE":  0.0,
                "HS":  0.0
            })
    # ---------------------------
    # SAVE INTERMEDIATE RESULTS
    # ---------------------------
    if i % 100 == 0:
        df_partial = pd.DataFrame(results)
        df_partial = df_partial.drop_duplicates(subset=["date", "basin"], keep="last")
        df_partial.to_pickle(intermediate_pkl)
        logging.info(f"Saved intermediate results at step {i}")

# ---------------------------
# FINAL SAVE AS A  PKL FILE
# ---------------------------
df = pd.DataFrame(results)
df_pivot = df.pivot(index="date", columns="basin",  values=["SWE", "HS"])
df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1)
df_pivot.to_pickle(output_pkl)
logging.info(f"Saved final timeseries to {output_pkl}")

print("Script finished successfully.")

