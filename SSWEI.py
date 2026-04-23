import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

# =========================================================
# CONFIG
# =========================================================
values_pkl = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/timeseries/basin_dao_timeseries.pkl"
basin_folder = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/basin_shapefiles/"
output_pkl = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/timeseries/basin_sspi_timeseries.pkl"
diagnostic_folder = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/plots/gamma_fits/"
output_plot = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/plots/sspi_map.png"
os.makedirs(diagnostic_folder, exist_ok=True)

winter_months = [10, 11, 12, 1, 2, 3]


# =========================================================
# FUNCTION SSPI (CLIMATOLOGY-BASED)
# =========================================================
def calculate_sspi(x, shape, loc, scale, p0):
    x = np.asarray(x, dtype=float)

    gamma_cdf = stats.gamma.cdf(np.where(x <= 0, 1e-6, x),
                                shape, loc=loc, scale=scale)
    cdf = p0 + (1 - p0) * gamma_cdf
    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)

    return stats.norm.ppf(cdf)


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_pickle(values_pkl)
df.index = pd.to_datetime(df.index)

# remove invalid values
swe_cols = df['SWE']
hs_cols = df['HS']

df[swe_cols <= 0] = np.nan
df[hs_cols <= 0] = np.nan
# monthly aggregation
df = df.resample("ME").mean()

df_winter = df[df.index.month.isin(winter_months)]

basins = df_winter.columns.get_level_values(1).unique()

# output container
sspi_df = pd.DataFrame(index=df_winter.index)


# =========================================================
# LOOP BASINS
# =========================================================
for basin in basins:

    swe = df_winter[('SWE', basin)]
    if isinstance(swe, pd.DataFrame):
        swe = swe.iloc[:, 0]

    sspi_series = pd.Series(index=swe.index, dtype=float)

    # =====================================================
    # MONTHLY CLIMATOLOGY FIT
    # =====================================================
    for month in winter_months:

        mask = swe.index.month == month

        subset = swe.loc[mask]

        # keep index alignment BEFORE dropping NaN
        subset_valid = subset.dropna()

        if len(subset_valid) < 10:
            continue

        x = subset_valid.values

        p0 = np.sum(x <= 0) / len(x)
        x_pos = x[x > 0]

        if len(x_pos) < 5:
            continue

        shape, loc, scale = stats.gamma.fit(x_pos, floc=0)

        # compute SSPI ONLY on valid subset
        sspi_vals = calculate_sspi(x, shape, loc, scale, p0)

        # IMPORTANT: assign using subset_valid index (NOT mask)
        sspi_series.loc[subset_valid.index] = sspi_vals

        fig, ax = plt.subplots(figsize=(6, 4))

        # sort values
        x_sorted = np.sort(x_pos)

        # empirical CDF (rank)
        y_emp = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

        # scatter empirical CDF
        ax.scatter(
            x_sorted,
            y_emp,
            s=12,
            alpha=0.6,
            label="Empirical CDF"
        )

        # fitted Gamma CDF
        xx = np.linspace(np.min(x_pos), np.max(x_pos), 200)
        ax.plot(
            xx,
            stats.gamma.cdf(xx, shape, loc=loc, scale=scale),
            color="red",
            linewidth=2,
            label="Gamma CDF fit"
        )

        ax.set_title(f"{basin} - Month {month}")
        ax.set_xlabel("SWE")
        ax.set_ylabel("Cumulative probability")
        ax.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(diagnostic_folder, f"{basin}_month{month}_gamma_scatter_fit.png"),
            dpi=300
        )

        plt.close()

    sspi_df[basin] = sspi_series


# =========================================================
# SAVE
# =========================================================
sspi_df = sspi_df.sort_index()
sspi_df.to_pickle(output_pkl)

print("SSPI saved to:", output_pkl)

# =========================================================
# JANUARY CLIMATOLOGY (AVERAGE OVER YEARS)
# =========================================================
selected_month = 1
jan_sspi = sspi_df[sspi_df.index.month == selected_month]

basin_mean = jan_sspi.mean(axis=0).to_frame("SSPI")
basin_mean["basin_name"] = basin_mean.index

# REMOVE NaNs BEFORE PLOTTING (IMPORTANT FIX)
basin_mean = basin_mean.dropna()

# =========================================================
# LOAD SHAPEFILES
# =========================================================
gdf_list = []

for shp in os.listdir(basin_folder):
    if shp.endswith(".shp"):
        gdf = gpd.read_file(os.path.join(basin_folder, shp))
        gdf["basin"] = os.path.splitext(shp)[0]
        gdf_list.append(gdf)

basins_shp = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

# merge
basins_shp = basins_shp.merge(basin_mean, on="basin_name", how="left")


# =========================================================
# SAFE COLOR SCALE (FIX YOUR ERROR)
# =========================================================
values = basins_shp["SSPI"].values
values = values[np.isfinite(values)]

if len(values) == 0:
    raise ValueError("No valid SSPI values to plot")

vmax = np.nanmax(np.abs(values))
if vmax == 0:
    vmax = 1e-6

bounds = np.linspace(-vmax, vmax, 9)
cmap = plt.get_cmap("RdBu_r")
norm = mcolors.BoundaryNorm(bounds, cmap.N)


# =========================================================
# PLOT MAP
# =========================================================
fig, ax = plt.subplots(figsize=(10, 8))

basins_shp.plot(
    column="SSPI",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    linewidth=0.5,
    ax=ax,
    missing_kwds={"color": "lightgrey"}
)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("SSPI (January climatology)")

ax.set_title("SSPI - January (mean over all years)")
ax.axis("off")

plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.close()

print("Map saved:", output_plot)