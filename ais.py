#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIS Data Plotting Toolkit

Generates summary visualisations from AIS (Automatic Identification System) datasets, 
including emissions, discharges, vessel types, flag states, and spatial activity. 
This script supports exploratory data analysis and figure creation for Arctic maritime 
traffic, environmental assessments, and operational planning.

Features:
    - Horizontal bar plots of emissions and discharges by vessel type
    - Flag state and ice class distribution charts
    - High-resolution heat maps of AIS positions
    - Clean, consistent output for publication or reporting

Usage:
    Import the plotting functions into your script or notebook, or run them standalone.

Parameters:
    - df (pandas.DataFrame): AIS data containing emissions, classifications, and geolocation.
    - variables (list): Emission or discharge variables to plot.
    - group_col (str): Grouping column (e.g., 'astd_cat' or 'flagname').
    - output_dir (str): Directory where output plots will be saved.

Requirements:
    - Python 3.x
    - pandas
    - matplotlib

Author:
    Adam Garbo

Created:
    2025-05-06

Last Updated:
    2025-05-06

Repository:
    https://github.com/adamgarbo/at-819

License:
    MIT License

"""
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from shapely.geometry import Polygon
from shapely.geometry import mapping
import os
import matplotlib.cm as cm
import numpy as np

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
path = '/Users/adamgarbo/Library/CloudStorage/GoogleDrive-adam.garbo@gmail.com/My Drive/UNIS/AT-819/Module 4 - POLARIS/Python/'

df_raw = pd.read_csv(path + 'ASTD_area_level2_201501.csv', sep=';')

# Cleaned: only ships with ≥10 points
ship_counts = df_raw['shipid'].value_counts()
valid_ships = ship_counts[ship_counts >= 10].index
df_clean = df_raw[df_raw['shipid'].isin(valid_ships)]

# AOI: longitude wrap and Arctic Circle
df_nwp = df_clean[
    ((df_clean['longitude'] <= -30) | (df_clean['longitude'] >= 180)) &
    (df_clean['latitude'] >= 66.56083)
]

df_nwp.columns

df_nwp.latitude.describe(include='all')
df_nwp.longitude.describe(include='all')


# -----------------------------------------------------------------------------
# Polygon for AOI box (Shapely)
# -----------------------------------------------------------------------------
aoi_poly = Polygon([
    (-170, 66.56083),
    (-170, 80),
    (-30,  80),
    (-30,  66.56083),
    (-170, 66.56083)
])
coords = mapping(aoi_poly)["coordinates"][0]
x, y = zip(*coords)

# -----------------------------------------------------------------------------
# Plotting function
# -----------------------------------------------------------------------------
def plot_ais(df, title, save_path):
    projection = ccrs.Mercator(central_longitude=-95)
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})

    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
                     edgecolor='face', facecolor='lightblue'), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m',
                     edgecolor='face', facecolor='lightgray'), zorder=1)
    ax.coastlines(resolution='50m')
    ax.set_extent([-180, -15, 60, 82.5], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    gl.xlocator = mticker.FixedLocator(range(-180, 0, 10))
    gl.ylocator = mticker.FixedLocator(range(30, 91, 5))
    gl.top_labels = gl.right_labels = False

    # Points
    ax.scatter(df['longitude'], df['latitude'], s=2, color='red', alpha=0.7, transform=ccrs.PlateCarree())
    # AOI box (always show for context)
    ax.plot(x, y, color='blue', linewidth=2, linestyle='--', transform=ccrs.PlateCarree())

    ax.set_title(f"{title}\n{len(df):,} points", fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# Plot maps
# -----------------------------------------------------------------------------
plot_ais(df_raw,   "Northwest Passage (Raw Data)",    '/Users/adamgarbo/Downloads/map_raw.png')
plot_ais(df_clean, "Northwest Passage (Cleaned Data)",'/Users/adamgarbo/Downloads/map_clean.png')
plot_ais(df_nwp,   "Northwest Passage (Area of Interest)", '/Users/adamgarbo/Downloads/map_aoi.png')

# -----------------------------------------------------------------------------
# Plot unique ship types
# -----------------------------------------------------------------------------

# Count unique vessels per type
vessel_counts = df_nwp.groupby('astd_cat')['shipid'].nunique().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(6, 4))

vessel_counts.plot(kind='barh', color='steelblue', edgecolor='black', ax=ax)

ax.set_xlabel("Number of Unique Vessels")
ax.set_ylabel("Vessel Type")
ax.set_title("Unique Vessels per Type", fontsize=14)

# Force integer x-axis ticks
ax.xaxis.get_major_locator().set_params(integer=True)

plt.tight_layout()
plt.savefig('/Users/adamgarbo/Downloads/vessel_type_unique.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------------
# Plot ship types
# -----------------------------------------------------------------------------

# Group and count by vessel type (drop missing)
vessel_counts = df_nwp['astd_cat'].dropna().value_counts().sort_values(ascending=True)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))

vessel_counts.plot(kind='barh', color='steelblue', edgecolor='black', ax=ax)

ax.set_xlabel("Number of AIS Points")
ax.set_ylabel("Vessel Type")
ax.set_title("Number of AIS Records per Vessel Type", fontsize=14)
plt.tight_layout()

# Optional: Save
plt.savefig('/Users/adamgarbo/Downloads/vessel_type_bar.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------------------------------------------------------
# Plot ship size group
# -----------------------------------------------------------------------------

# Ensure df_nwp is safe to modify
df_nwp = df_nwp.copy()

# Clean sizegroup_gt strings
df_nwp['sizegroup_gt'] = df_nwp['sizegroup_gt'].str.replace(r'\s*GT$', '', regex=True)

# Define custom order
size_order = [
    '10000 - 24999',
    '5000 - 9999',
    '1000 - 4999',
    '< 1000'
]

# Convert to ordered categorical
df_nwp['sizegroup_gt'] = pd.Categorical(df_nwp['sizegroup_gt'], categories=size_order, ordered=True)

# Count unique ship IDs per size group
unique_ships = df_nwp.drop_duplicates(subset=['shipid', 'sizegroup_gt'])
size_counts = unique_ships['sizegroup_gt'].value_counts().reindex(size_order)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
size_counts.plot(kind='barh', color='steelblue', edgecolor='black', ax=ax)

ax.set_xlabel("Number of Unique Vessels")
ax.set_ylabel("Ship Size Group (GT)")
ax.set_title("Unique Vessels per Ship Size Group", fontsize=14)
ax.xaxis.get_major_locator().set_params(integer=True)

plt.tight_layout()
plt.savefig('/Users/adamgarbo/Downloads/ship_sizegroup_ordered.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------------------------------------------------------
# Estimated ship speed
# -----------------------------------------------------------------------------

# Estimate speed from distance and time (where time > 0)
df_nwp = df_nwp[df_nwp['sec_nextpoint'] > 0]
df_nwp['speed_knots'] = (df_nwp['dist_nextpoint'] / (df_nwp['sec_nextpoint'] / 3600)) * 0.539957

# Filter out unrealistic values
df_nwp = df_nwp[(df_nwp['speed_knots'] > 0) & (df_nwp['speed_knots'] < 50)]

fig, ax = plt.subplots(figsize=(6, 4))
df_nwp['speed_knots'].hist(bins=50, color='teal', edgecolor='black', ax=ax)

ax.set_title("Estimated Vessel Speeds (in Knots)", fontsize=14)
ax.set_xlabel("Speed (knots)")
ax.set_ylabel("Number of AIS Points")

plt.tight_layout()
plt.savefig("/Users/adamgarbo/Downloads/speed_histogram.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------------
# Plot AIS data near Greenland
# -----------------------------------------------------------------------------

# Create map with PlateCarree projection (good for high-lat zooms)
projection = ccrs.Mercator(central_longitude=-95)
fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': projection})
 
# Add land and ocean features
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='lightgray')
ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='lightblue')
ax.coastlines(resolution='10m')

# Set extent to western Greenland (in lon/lat)
ax.set_extent([-61, -46, 66, 73.5], crs=ccrs.PlateCarree())

# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--', alpha=0.6)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
gl.xlocator = mticker.FixedLocator(range(-70, -30, 2))
gl.ylocator = mticker.FixedLocator(range(65, 78, 1))
gl.top_labels = gl.right_labels = False

# Plot AIS data from df_nwp (lat/lon scatter)
ax.scatter(df_nwp['longitude'], df_nwp['latitude'], s=20, color='red', alpha=0.6, transform=ccrs.PlateCarree())

# Title and display
plt.title("AIS Positions", fontsize=14)
plt.tight_layout()
plt.savefig("/Users/adamgarbo/Downloads/west_greenland_ais.png", dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------------------------------------------------------
# Plot AIS data near Greenland by Vessel Type
# -----------------------------------------------------------------------------

# Set up projection
projection = ccrs.Mercator(central_longitude=-95)
fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': projection})

# Add map features
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='lightgray')
ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='lightblue')
ax.coastlines(resolution='10m')
ax.set_extent([-61, -46, 66, 73.5], crs=ccrs.PlateCarree())

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--', alpha=0.6)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 9, 'rotation': 0}
gl.ylabel_style = {'size': 9, 'rotation': 0}
gl.top_labels = False
gl.right_labels = False
gl.bottom_labels = True
gl.left_labels = True
gl.xlocator = mticker.FixedLocator(range(-70, -30, 2))
gl.ylocator = mticker.FixedLocator(range(65, 78, 1))

# Plot AIS by vessel type
vessel_types = df_nwp['astd_cat'].dropna().unique()
colors = cm.tab10(np.linspace(0, 1, len(vessel_types)))

for vessel_type, color in zip(vessel_types, colors):
    subset = df_nwp[df_nwp['astd_cat'] == vessel_type]
    ax.scatter(
        subset['longitude'], subset['latitude'],
        s=15, alpha=0.8, label=vessel_type,
        color=color, transform=ccrs.PlateCarree()
    )

# Legend outside plot
ax.legend(
    title="Vessel Type", title_fontsize=9, fontsize=8,
    loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True
)

# Add major communities (lon, lat, name)
greenland_communities = [
    ("Upernavik",    -56.147222,72.786944),
    ("Sisimiut",     -53.6735, 66.9395),
    ("Ilulissat",    -51.1000, 69.2167),
    ("Aasiaat",      -52.8694, 68.7098),
    ("Uummannaq",    -52.1167, 70.6761),
]

# Plot community markers and labels
for name, lon, lat in greenland_communities:
    ax.plot(lon, lat, marker='o', color='red', markersize=3,
            transform=ccrs.PlateCarree(), zorder=5)
    ax.text(
        lon + 0.2, lat, name,
        fontsize=7, ha='left', va='center', color='black',
        transform=ccrs.PlateCarree(), zorder=5,
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.8)
    )

# Title and save
#plt.title("AIS Positions by Vessel Type", fontsize=13)
plt.tight_layout()
plt.savefig("/Users/adamgarbo/Downloads/west_greenland_by_type_with_communities.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------------
# Plot AIS data near Greenland by Ice Class
# -----------------------------------------------------------------------------

# Count unique ships per ice class (excluding missing)
iceclass_counts = df_nwp.dropna(subset=['iceclass']).groupby('iceclass')['shipid'].nunique().sort_values()

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
iceclass_counts.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')

ax.set_xlabel("Number of Unique Vessels")
ax.set_ylabel("Ice Class")
ax.set_title("Distribution of Ice-Classed Vessels", fontsize=14)
ax.xaxis.get_major_locator().set_params(integer=True)

plt.tight_layout()
plt.savefig("/Users/adamgarbo/Downloads/iceclass_unique_vessels.png", dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------------------------------------------------------
# Plot AIS data near Greenland by Flag Name
# -----------------------------------------------------------------------------
df_nwp['flagname'] = df_nwp['flagname'].replace({
    'Denmark (Dis)': 'Greenland'
})

# Count unique vessels per flag (excluding missing)
flag_counts = df_nwp.dropna(subset=['flagname']).groupby('flagname')['shipid'].nunique().sort_values(ascending=False)

# Take top N for clarity (optional)
top_flags = flag_counts.head(10)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
top_flags.plot(kind='bar', ax=ax, color='darkslateblue', edgecolor='black')

ax.set_ylabel("Number of Unique Vessels")
ax.set_xlabel("Flag (Country of Registry)")
ax.set_title("Flags by Number of Unique Vessels", fontsize=14)
ax.yaxis.get_major_locator().set_params(integer=True)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig("/Users/adamgarbo/Downloads/flagname_unique_vessels.png", dpi=300)
plt.show()


# -----------------------------------------------------------------------------
# Plot AIS data in table format
# -----------------------------------------------------------------------------

# Define table data
columns = [
    'shipid', 'date_time_utc', 'flagname', 'iceclass', 'astd_cat',
    'lloyds3_cat', 'sizegroup_gt', 'fuelquality', 'fuelcons', 'co', 'co2',
    'so2', 'pm', 'nox', 'n2o', 'nmvoc', 'ch4', 'blackcarbon', 'organiccarbon',
    'oilbilgewater', 'blackwater', 'greywater', 'garbage',
    'dist_nextpoint', 'sec_nextpoint', 'longitude', 'latitude'
]

descriptions = [
    "Unique vessel identifier",
    "Timestamp of AIS message (UTC)",
    "Country of registry (ship flag)",
    "Ice classification",
    "Vessel type/category (ASTD)",
    "Lloyd's Register vessel category",
    "Size category (gross tonnage)",
    "Fuel quality classification",
    "Fuel consumption (kg)",
    "CO emissions",
    "CO₂ emissions",
    "SO₂ emissions",
    "Particulate matter emissions",
    "NOₓ emissions",
    "N₂O emissions",
    "Non-methane VOC emissions",
    "Methane emissions",
    "Black carbon emissions",
    "Organic carbon emissions",
    "Oil bilge water discharged",
    "Sewage discharged (blackwater)",
    "Greywater discharged",
    "Garbage produced/discharged",
    "Distance to next AIS point",
    "Time to next AIS point (s)",
    "Longitude",
    "Latitude"
]

# Create figure
fig, ax = plt.subplots(figsize=(6, 7))
ax.axis('off')
table_data = list(zip(columns, descriptions))
table = ax.table(cellText=table_data, colLabels=["Column", "Description"], cellLoc='left', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.8)

plt.tight_layout()
plt.savefig("/Users/adamgarbo/Downloads/column_descriptions_table.png", dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------------------------------------------------------
# Fucntion to create plot of AIS data variables
# -----------------------------------------------------------------------------

def plot_variable_sum_by_vessel_type(df, variables, group_col='astd_cat', output_dir="."):
    """
    Creates horizontal bar plots showing the total sum of each variable by vessel type,
    using a unique colour per plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a colormap with as many colours as variables
    cmap = cm.get_cmap('Set2', len(variables))
    color_list = [cmap(i) for i in range(len(variables))]

    for i, var in enumerate(variables):
        if var not in df.columns:
            print(f"Skipping missing column: {var}")
            continue

        grouped = df[[group_col, var]].dropna().groupby(group_col)[var].sum().sort_values()

        if grouped.empty:
            print(f"No data for: {var}")
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        grouped.plot(kind='barh', color=color_list[i], edgecolor='black', ax=ax)

        ax.set_xlabel(f"Total {var.upper()} (kg)")
        ax.set_ylabel("Vessel Type")
        ax.set_title(f"Total {var.upper()} by Vessel Type", fontsize=14)
        ax.xaxis.get_major_locator().set_params(integer=True)

        plt.tight_layout()
        filename = os.path.join(output_dir, f"{var}_by_vessel_type.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filename}")
        
emission_vars = [
    'co', 'co2', 'so2', 'pm', 'nox', 'n2o', 'nmvoc', 'ch4',
    'blackcarbon', 'organiccarbon', 'oilbilgewater',
    'blackwater', 'greywater', 'garbage'
]

plot_variable_sum_by_vessel_type(df_nwp, emission_vars, output_dir="/Users/adamgarbo/Downloads/emissions_by_type")
