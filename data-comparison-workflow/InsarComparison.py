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
    """
    Compute the root mean square of an array, ignoring NaN values.
    """
    return np.sqrt(np.nanmean(x**2, axis=axis))


def compute_LOS_displacement_SeisSol_data(lon_g, lat_g, vx, vy, vz, lonlat_barycenter, U, V, W):
    """
    Compute the Line-Of-Sight (LOS) displacement using SeisSol data.
    
    Parameters:
    - lon_g, lat_g: Grid of longitude and latitude values.
    - vx, vy, vz: Velocity components.
    - lonlat_barycenter: Barycenter of the triangles in the mesh.
    - U, V, W: Surface deformation components from SeisSol.

    Returns:
    - D_los: The LOS displacement.
    """
    D_los = U * vx + V * vy + W * vz
    return -D_los
    
    
def read_seissol_surface_data(xdmfFilename):
    """
    Read unstructured free surface output from SeisSol and compute cell barycenters.
    
    Parameters:
    ----------
    xdmfFilename : string
        Full path to the .xdmf file.

    Returns:
    -------
    - lons, lats: Arrays of the locations of each grid point where surface deformation is defined.
    - lonlat_barycenter: Array of the longitude and latitude coordinates of the barycenter of each triangle.
    - connect: Array of indices for the corners of each triangle at the surface.
    - U, V, W: Components of the modeled surface deformation.
    """
    sx = seissolxdmf.seissolxdmf(xdmfFilename)

    # Read geometry, connectivity, and deformation data
    xyz = sx.ReadGeometry()
    connect = sx.ReadConnect()
    U = sx.ReadData("u1", sx.ndt - 1)
    V = sx.ReadData("u2", sx.ndt - 1)
    W = sx.ReadData("u3", sx.ndt - 1)

    # Transform coordinates from UTM to WGS84
    transformer = Transformer.from_crs("epsg:32618", "epsg:4326", always_xy=True)
    lons, lats = transformer.transform(xyz[:, 0], xyz[:, 1])
    xy = np.vstack((lons, lats)).T

    # Compute triangle barycenters
    lonlat_barycenter = (
        xy[connect[:, 0], :] + xy[connect[:, 1], :] + xy[connect[:, 2], :]
    ) / 3.0

    return lons, lats, lonlat_barycenter, connect, U, V, W

def interpolate_seissol_surf_output(lonlat_barycenter, U, V, W, df):
    """
    Interpolate SeisSol free surface output to GPS data locations.
    
    Parameters:
    ----------
    - lonlat_barycenter: Array of barycenters' longitude and latitude.
    - U, V, W: Components of the modeled surface deformation.
    - df: DataFrame containing GPS observation data.

    Returns:
    -------
    - ui, vi, wi: Interpolated surface deformation components at the GPS data locations.
    """
    locGPS = np.vstack((df["lon"].to_numpy(), df["lat"].to_numpy())).T

    # Interpolate deformation components at GPS locations
    ui = interp.LinearNDInterpolator(lonlat_barycenter, U).__call__(locGPS)
    vi = interp.LinearNDInterpolator(lonlat_barycenter, V).__call__(locGPS)
    wi = interp.LinearNDInterpolator(lonlat_barycenter, W).__call__(locGPS)

    return ui, vi, wi


# Read InSAR observation data from CSV
datadir = '/Users/hyin/agsd/projects/insar/2021_haiti/dynamic-rupture/observation-data/insar/data/csv/'
filebase = '/A2_A042_20210101-20210827_los'
obs_df = pd.read_csv(datadir + filebase + '.csv')

# Convert observation data to an xarray Dataset
obs_ds = xr.Dataset.from_dataframe(obs_df.set_index(['lat', 'lon']))
# Convert LOS data from millimeters to meters and rename the variable
obs_ds['los_mm'] = obs_ds['los_mm'] / 1000
obs_ds = obs_ds.rename({'los_mm': 'los_m'})

# Optional: Save as .nc file for later use
# obs_ds.to_netcdf(datadir + filebase + '.nc')


#########################
# READ SYNTHETIC DATA  #
#########################

# Define SeisSol surface XDMF file
xdmfFilename = '/Users/hyin/ags_local/data/haiti_seissol_data/FL33_dynamic-relaxation/jobid_3437321/output_jobid_3437321_extracted-surface.xdmf'

# Read modeled surface deformation data from SeisSol XDMF file
vertex_lon, vertex_lat, lonlat_barycenter, connect, U, V, W = read_seissol_surface_data(xdmfFilename)
lonlat_barycenter[:, 0] = lonlat_barycenter[:, 0] + 360  # Adjust longitude

# Interpolate modeled data at each observed point
x_synth_i, y_synth_i, z_synth_i = interpolate_seissol_surf_output(lonlat_barycenter, U, V, W, obs_df)

# Convert interpolated data to a pandas DataFrame
synth_df = pd.DataFrame({
    'lat': obs_df['lat'],
    'lon': obs_df['lon'],
    'x_synth_i': x_synth_i.flatten(),
    'y_synth_i': y_synth_i.flatten(),
    'z_synth_i': z_synth_i.flatten()
})

# Convert DataFrame to xarray Dataset
synth_ds = xr.Dataset.from_dataframe(synth_df.set_index(['lat', 'lon']))

# Merge observed and synthetic datasets
merged_ds = obs_ds.assign(x_synth_i=synth_ds['x_synth_i'],
                          y_synth_i=synth_ds['y_synth_i'],
                          z_synth_i=synth_ds['z_synth_i'])

# Calculate LOS vectors
merged_ds['los_synth_m'] = (merged_ds['x_synth_i'] * merged_ds['look_E'] +
                            merged_ds['y_synth_i'] * merged_ds['look_N'] +
                            merged_ds['z_synth_i'] * merged_ds['look_U'])
merged_ds['los_diff'] = merged_ds['los_m'] - merged_ds['los_synth_m']


#### Plot results using PyGMT
fig = pygmt.Figure()

# Define color map
pygmt.makecpt(cmap="viridis", series=[-0.8, 0.8])

# Set up the PyGMT figure with three panels
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
