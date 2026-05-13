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
monthly_shp_folder = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/monthly_sspi_shapefiles/"
monthly_maps_folder = "/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/plots/monthly_sspi_maps/"

os.makedirs(monthly_maps_folder, exist_ok=True)
os.makedirs(diagnostic_folder, exist_ok=True)
os.makedirs(monthly_shp_folder, exist_ok=True)

winter_months = [11, 12, 1, 2, 3, 4, 5]

# =========================================================
# DIAGNOSTIC TABLE
# =========================================================
fit_results = []


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
df[swe_cols < 0] = np.nan

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

        subset = subset/(10**6)

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
        # verify with a kolmogorov-smirnov test the fitting of the gamma distribution
        ks_stat, p_value = stats.kstest(x_pos, 'gamma', args=(shape, loc, scale))


        # compute SSPI ONLY on valid subset
        sspi_vals = calculate_sspi(x, shape, loc, scale, p0)

        # IMPORTANT: assign using subset_valid index (NOT mask)
        sspi_series.loc[subset_valid.index] = sspi_vals

        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
        import os

        # ---------------------------
        # Plot empirical CDF with Gamma fit (scatter)
        # ---------------------------

        # Use a built-in style that works
        plt.style.use("ggplot")

        fig, ax = plt.subplots(figsize=(6, 4))

        # Sort data for empirical CDF
        x_sorted = np.sort(x_pos)
        y_emp = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

        # Scatter plot of empirical CDF
        ax.scatter(
            x_sorted,
            y_emp,
            s=20,  # slightly bigger markers
            alpha=0.7,  # semi-transparent
            color='blue',
            label="Empirical CDF"
        )

        # Fitted Gamma CDF
        xx = np.linspace(np.min(x_pos), np.max(x_pos), 300)
        ax.plot(
            xx,
            stats.gamma.cdf(xx, shape, loc=loc, scale=scale),
            color="red",
            linewidth=2,
            label="Gamma CDF fit"
        )


        # Labels and title
        basin_plot = basin.replace("AdBAlpiOR_", "")
        ax.set_title(f"{basin_plot} - Month {month}", fontsize=14)
        ax.set_xlabel("SWE (Mm³)", fontsize=12)
        ax.set_ylabel("Cumulative Probability", fontsize=12)

        # Grid and legend
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)

        # Save figure
        plt.tight_layout()
        output_file = os.path.join(
            diagnostic_folder,
            f"{basin_plot}_month{month}_gamma_scatter_fit.png"
        )
        plt.savefig(output_file, dpi=300)
        plt.close(fig)

        # save diagnostics
        fit_results.append({
            "basin": basin,
            "month": month,
            "shape": shape,
            "loc": loc,
            "scale": scale,
            "ks_pvalue": p_value
        })

    sspi_df[basin] = sspi_series


# =========================================================
# SAVE
# =========================================================
sspi_df = sspi_df.sort_index()
sspi_df.to_pickle(output_pkl)

print("SSPI saved to:", output_pkl)


# =========================================================
# CREATE DIAGNOSTIC TABLE
# =========================================================
fit_df = pd.DataFrame(fit_results)
fit_df["basin"] = fit_df["basin"].replace("AdBAlpiOR_", "")

gamma_df = fit_df.reset_index().pivot(
    index="basin",
    columns="month",
    values=["shape", "loc", "scale"])

p_value_df = fit_df.reset_index().pivot(
    index="basin",
    columns="month",
    values=["ks_pvalue"])

# Save as CSV
gamma_df.to_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/timeseries/gamma_parameters.csv")
p_value_df.to_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/timeseries/ks_pvalues.csv")

print("diagnostics saved")
# =========================================================
# JANUARY CLIMATOLOGY (AVERAGE OVER YEARS TO CHECK IF THE CODE WORKS)
# =========================================================
selected_month = 1
jan_sspi = sspi_df[sspi_df.index.month == selected_month]

basin_mean = jan_sspi.mean(axis=0).to_frame("SSPI")
basin_mean["basin_name"] = basin_mean.index

# REMOVE NaNs BEFORE PLOTTING (IMPORTANT FIX)
basin_mean = basin_mean.dropna()

gdf_list = []

for shp in os.listdir(basin_folder):
    if shp.endswith(".shp"):
        gdf = gpd.read_file(os.path.join(basin_folder, shp))
        gdf["basin"] = os.path.splitext(shp)[0]
        gdf_list.append(gdf)

basins_shp = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

# merge
basins_shp_jan = basins_shp.merge(basin_mean, on="basin_name", how="left")

values = basins_shp_jan["SSPI"].values
values = values[np.isfinite(values)]

if len(values) == 0:
    raise ValueError("No valid SSPI values to plot")

vmax = np.nanmax(np.abs(values))
if vmax == 0:
    vmax = 1e-6

bounds = np.linspace(-vmax, vmax, 9)
cmap = plt.get_cmap("RdBu_r")
norm = mcolors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(10, 8))

basins_shp_jan.plot(
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
plt.close(fig)

print("Map saved:", output_plot)

# =========================================================
# MONTHLY MAPS FOR EACH YEAR-MONTH
# =========================================================

# fixed color scale across all months for comparison
vmin_global = -2
vmax_global = 2
cmap = 'RdBu'

# calculate consistent map extent
map_bounds = basins_shp.total_bounds  # [minx, miny, maxx, maxy]

for year in sspi_df.index.year.unique():
    for month in winter_months:
        # select data for year-month
        selected_sspi = sspi_df[(sspi_df.index.year == year) & (sspi_df.index.month == month)]
        if selected_sspi.empty:
            continue

        selected_sspi = selected_sspi.T
        selected_sspi["basin_name"] = selected_sspi.index
        selected_sspi = selected_sspi.rename(columns={selected_sspi.columns[0]: "SSPI"})

        # merge shapefile with SSPI data
        basin_shp_selected = basins_shp.set_index("basin_name").join(selected_sspi["SSPI"])
        basin_shp_selected = basin_shp_selected.reset_index()

        # save shapefile
        output_shp = os.path.join(monthly_shp_folder, f"SSPI_{year}_{month:02d}.shp")
        basin_shp_selected.to_file(output_shp)

        x_range = map_bounds[2] - map_bounds[0]
        y_range = map_bounds[3] - map_bounds[1]
        fig_width = 4

        fig_height = fig_width * (y_range / x_range)  # keep aspect ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        basin_shp_selected.plot(
            column='SSPI',
            ax=ax,
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            edgecolor='black',
            linewidth=0.5,
            missing_kwds={"color": "lightgrey", "label": "No data"}
        )


        # consistent extent
        ax.set_xlim(map_bounds[0], map_bounds[2])
        ax.set_ylim(map_bounds[1], map_bounds[3])
        ax.set_aspect('equal')
        ax.set_axis_off()

        # colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_global, vmax=vmax_global))
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
        cbar.set_label("SSPI", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        # title
        ax.set_title(f"SSPI {year} - {month:02d}", fontsize=15)

        # save figure
        output_png = os.path.join(monthly_maps_folder, f"SSPI_{year}_{month:02d}.png")
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close(fig)

print("Monthly maps saved in:", monthly_maps_folder)
# =========================================================
# BIG FIGURE: ALL NOVEMBER (MONTH = 11) MAPS
# WITH COLORBAR OUTSIDE THE PLOT
# =========================================================
selected_month = 11  # November

# available years with data for the selected month
years_available = sorted(
    sspi_df[sspi_df.index.month == selected_month].index.year.unique()
)
n_years = len(years_available)

if n_years == 0:
    print(f"No data available for month {selected_month}")
else:
    # subplot organization
    ncols = 4
    nrows = int(np.ceil(n_years / ncols))

    # figure size
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 5 * nrows)
    )
    axes = np.array(axes).flatten()

    # fixed color scale for all maps
    vmin = -2
    vmax = 2

    for i, year in enumerate(years_available):
        ax = axes[i]

        # select SSPI data for the given year and month
        selected_sspi = sspi_df[
            (sspi_df.index.year == year) &
            (sspi_df.index.month == selected_month)
        ]

        if selected_sspi.shape[0] == 0:
            ax.set_axis_off()
            continue

        # transpose so basins are rows
        selected_sspi = selected_sspi.T
        selected_sspi["basin_name"] = selected_sspi.index

        # rename first column to 'SSPI'
        selected_sspi = selected_sspi.rename(
            columns={selected_sspi.columns[0]: "SSPI"}
        )

        # merge shapefile with SSPI data
        basin_shp_selected = basins_shp.merge(
            selected_sspi,
            left_on="basin_name",
            right_index=True,
            how="left"
        )
        basin_shp_selected = basin_shp_selected.rename(columns={
            'basin_name_x': 'basin_x',
            'basin_name_y': 'basin_y'
        })

        # plot map
        basin_shp_selected.plot(
            column='SSPI',
            ax=ax,
            cmap='RdBu',
            vmin=vmin,
            vmax=vmax,
            legend=False,
            edgecolor='black',
            linewidth=0.5
        )

        # add title for the year
        ax.set_title(f"{year}", fontsize=12)
        ax.set_axis_off()

    # remove unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()

    # -----------------------------------------------------
    # ADD COLORBAR OUTSIDE THE SUBPLOTS
    # -----------------------------------------------------
    sm = plt.cm.ScalarMappable(
        cmap='RdBu',
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm._A = []

    # leave extra space at bottom
    fig.subplots_adjust(
        bottom=0.12,
        top=0.93,
        wspace=0.05,
        hspace=0.15
    )

    # manual colorbar axis: [left, bottom, width, height]
    cax = fig.add_axes([0.25, 0.05, 0.50, 0.02])
    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation='horizontal'
    )
    cbar.set_label("SSPI", fontsize=12)

    # super title
    fig.suptitle(
        f"SSPI Maps - Month {selected_month:02d}",
        fontsize=30
    )

    # save figure
    output_bigfig = f"/home/idrologia/share/PhD_GiuliaBlandini_dati/DAO_project/plots/sspi_all_years_month_{selected_month:02d}.png"
    plt.savefig(
        output_bigfig,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    print("Big figure saved in:", output_bigfig)