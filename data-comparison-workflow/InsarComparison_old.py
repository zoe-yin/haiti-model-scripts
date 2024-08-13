import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import geopandas as gpd
import os
import rasterio
import seissolxdmf
import time
from scipy import spatial
from multiprocessing import Pool, cpu_count, Manager
from pyproj import Transformer
from cmcrameri import cm
import matplotlib
from rasterio.plot import show
import scipy.interpolate as interp
import xarray as xr
import pygmt

def nanrms(x, axis=None):
    return np.sqrt(np.nanmean(x**2, axis=axis))


def compute_LOS_displacement_SeisSol_data(lon_g, lat_g, vx, vy, vz, lonlat_barycenter, U,V,W):
    D_los = U * vx + V * vy + W * vz
    return -D_los
    
    
def read_seissol_surface_data(xdmfFilename):
    """
    Read unstructured free surface output and associated data.
    compute cell_barycenter
    
    
    Parameters
    ----------
    xdmfFilename : string
        Full path to the .xdmf file

    Returns
    -------
    lons, lats, lonlat_barycenter, connect, U, V, W
    lons, lats: n x 1 array the locations of each grid point where surface deformation is defined in the model. 
        I believe this is in WGS84 but need to check
        Where n is the number of points in the surface mesh
        i.e. for Haiti low res mesh this is (16347,)
    lonlat_barycenter: n x 2 array where each column is the lon and lat coords of the barycenter of a triangle
        Where n is the number of triangles (need to check this!)
        i.e. for Haiti low res mesh this is (32526, 2)
    connect: n x 3 array, where the columns contain indices for the corners of each triangle at the surface
    U, V, W: each are an nd-array where each is a component of modeled surface deformation:
        U = u1
        V = u2
        W = u3
    """
    sx = seissolxdmf.seissolxdmf(xdmfFilename)

    xyz = sx.ReadGeometry()
    connect = sx.ReadConnect()
    U = sx.ReadData("u1", sx.ndt - 1)
    V = sx.ReadData("u2", sx.ndt - 1)
    W = sx.ReadData("u3", sx.ndt - 1)
    # project the data to geocentric (lat, lon)

    # myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37.0 +lat_0=37.0"
    # UTM =  'epgs:32618'
    # WGS84 = "epsg:4326"
    transformer = Transformer.from_crs("epsg:32618", "epsg:4326", always_xy=True)
    lons, lats = transformer.transform(xyz[:, 0], xyz[:, 1])    # 
    xy = np.vstack((lons, lats)).T  # gives a grid of lon,lat values, need to check the format

    # compute triangule barycenter
    lonlat_barycenter = (
        xy[connect[:, 0], :] + xy[connect[:, 1], :] + xy[connect[:, 2], :]
    ) / 3.0

    return lons, lats, lonlat_barycenter, connect, U, V, W

def interpolate_seissol_surf_output(lonlat_barycenter, U, V , W, df):
    """interpolate SeisSol free surface output to GPS data location"""

    Fvsm = interp.LinearNDInterpolator(lonlat_barycenter, U)
    locGPS = np.vstack((df["lon"].to_numpy(), df["lat"].to_numpy())).T

    Fvsm = interp.LinearNDInterpolator(lonlat_barycenter, U)
    ui = Fvsm.__call__(locGPS)
    Fvsm = interp.LinearNDInterpolator(lonlat_barycenter, V)
    vi = Fvsm.__call__(locGPS)
    Fvsm = interp.LinearNDInterpolator(lonlat_barycenter, W)
    wi = Fvsm.__call__(locGPS)

    return ui, vi, wi


## Read in InSAR observation data from a CSV
# Write to geodataframe, with lon / lat fields as geometry
# Set geometry reference frame as WGS84

# Define SeisSol surface XDMF file
csvFilename = '/Users/hyin/ags_local/data/haiti_seissol_data/FL33_dynamic-relaxation/jobid_3437317/FL33_jobid_3437317_output_extracted-surface.xdmf'
# Extract the directory path and filename
csvpath, filename = os.path.split(csvFilename)
# Extract the filebase by removing the extension
csvfilebase, _ = os.path.splitext(filename)

datadir = '/Users/hyin/agsd/projects/insar/2021_haiti/dynamic-rupture/observation-data/insar/data/csv/'
filebase = '/A2_A042_20210101-20210827_los'
obs_df = pd.read_csv(datadir+filebase+'.csv')

## Write to a geopandas dataframe, skip this for now. 
# gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df.lon, df.lat, df.elevation, crs="EPSG:4326"), data=df)

# Create an array with observations location lat/lons for later use. 
lon_obs, lat_obs = obs_df.lon.array, obs_df.lat.array
locData = np.vstack((lon_obs,lat_obs)).T

# Convert observation data to an xarray DataSet
obs_ds = xr.Dataset.from_dataframe(obs_df.set_index(['lat', 'lon']))
# Convert "los_mm" from millimeters to meters
obs_ds['los_mm'] = obs_ds['los_mm'] / 1000
# Rrename the variable to reflect the new unit
obs_ds = obs_ds.rename({'los_mm': 'los_m'})

## Save as .nc file for later use if needed. 
# obs_ds.to_netcdf(datadir + filebase + '.nc')


#########################
# READ SYNTHETIC DATA  #
#########################


# Define SeisSol surface XDMF file
xdmfFilename = '/Users/hyin/ags_local/data/haiti_seissol_data/FL33_dynamic-relaxation/jobid_3437317/FL33_jobid_3437317_output_extracted-surface.xdmf'
# Extract the directory path and filename
path, filename = os.path.split(xdmfFilename)
# Extract the filebase by removing the extension
filebase, _ = os.path.splitext(filename)

# Output the results
print(f"path: {path}")
print(f"filebase: {filebase}")

# Read Modeled surface deformation data from surface xdmf
vertex_lon, vertex_lat, lonlat_barycenter, connect, U, V, W=read_seissol_surface_data(xdmfFilename)
lonlat_barycenter[:, 0] = lonlat_barycenter[:, 0] + 360
# There are 32526 triangles and 16347 vertices
# latlon_barycenter is a (32526,2) array, with lat and lon for every triangle
# U, V, W are (32526,) arrays
# vertex_lon and vertex_lat are (16347,) arrays

# ## Interpolate modeled data at each observed point
# Where df is the dataframe with InSAR observation points
x_synth_i, y_synth_i, z_synth_i = interpolate_seissol_surf_output(lonlat_barycenter, U, V , W, obs_df)

# Convert arrays to a pandas dataframe
synth_df = pd.DataFrame({
    'lat': obs_df['lat'],
    'lon': obs_df['lon'],
    'x_synth_i': x_synth_i.flatten(),
    'y_synth_i': y_synth_i.flatten(),
    'z_synth_i': z_synth_i.flatten()
})

# Convert df to xarray DataSet
synth_ds = xr.Dataset.from_dataframe(synth_df.set_index(['lat', 'lon']))

## Merge Observed and Synthetic Datasets
merged_ds = obs_ds.assign(x_synth_i=synth_ds['x_synth_i'],y_synth_i=synth_ds['y_synth_i'], z_synth_i=synth_ds['z_synth_i'])
merged_ds['look_U'].mean()

# Calculate LOS vectors 
# Element-by-element addition
merged_ds['los_synth_m'] = merged_ds['x_synth_i']*merged_ds['look_E'] + merged_ds['y_synth_i']*merged_ds['look_N'] + merged_ds['z_synth_i']*merged_ds['look_U']
merged_ds['los_diff'] = merged_ds['los_m'] - merged_ds['los_synth_m']
merged_ds



#### Plot results
# Set up the PyGMT figure with two panels
import pygmt
fig = pygmt.Figure()


pygmt.makecpt(cmap="viridis", series=[-0.8, 0.8])

# Define the layout for a 1-row, 2-column subplot
with fig.subplot(nrows=1, ncols=3, figsize=("18i", "6i"), sharex="b", sharey="l", frame="a"):
    
    # Left Panel: Observed LOS data
    with fig.set_panel(panel=0):
        fig.grdimage(grid=merged_ds['los_m'], projection='M6i', frame=True, cmap=True)
        fig.coast(shorelines=True, water='skyblue')
        fig.colorbar(frame='af+l"LOS displacement (m)"')
    
    # Middle Panel: Synthetic LOS data
    with fig.set_panel(panel=1):
        fig.grdimage(grid=merged_ds['los_synth_m'], projection='M6i', frame=True, cmap=True)
        fig.coast(shorelines=True, water='skyblue')
        fig.colorbar(frame='af+l"LOS displacement (m)"')

    # Right Panel: Residual (los_m - los_synth_m)
    with fig.set_panel(panel=2):
        fig.grdimage(grid=merged_ds['los_diff'], projection='M6i', frame=True, cmap=True)
        fig.coast(shorelines=True, water='skyblue')
        fig.colorbar(frame='af+l"LOS displacement (m)"')

# Show the figure
fig.show()


