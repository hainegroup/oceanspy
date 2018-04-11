import numpy as np
import pandas as pd
import xarray as xr
import xgcm as xgcm
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def subsample(ds         = None, 
              varList    = ['Eta', 'Temp', 'S', 'U', 'V', 'W'], 
              lonRange   = [-180, 180],
              latRange   = [-90, 90],
              depthRange = [0, float('inf')],
              timeRange  = ['2007-09-01T00', '2008-08-31T18'],
              timeFreq   = '6H',
              sampMethod = 'snapshot',
              interpC    = False,
              saveNetCDF = False,
              plotMap    = False):
    """
    Create a subDataset and its Grid from another Dataset.
    
    Parameters
    ----------
    ds: xarray.Dataset or None
       Dataset that will be cropped. 
       If None: open the whole dataset first, then subsample.
    varList: list
            Variables to keep
    lonRange: list
             Longitude limits (based on Xp1 dimension)
    latRange: list
             Latitude limits  (based on Yp1 dimension)
    depthRange: list
               Depth limits   (based on Zp1 dimension)
    timeRange: list
              Time limits
    timeFreq: str
             Time frequency. Available optionts are pandas Offset Aliases (e.g., '6H'):
             http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    sampMethod: str
               Sampling method: 'snapshot' or 'mean'
    interpC: bool
            Interpolate all variables to C-grid
    saveNetCDF: str or False
               Provide a filename to save the subset in NetCDF format.
    plotMap: bool
            If True plot a map of the model domain and its resolution
              
    Returns
    -------
    ds : xarray.Dataset
        subDataset of the original Dataset
    grid : xgcm.Grid
          Grid with all the staggered grids
    """

    # Check parameters
    if not isinstance(ds, xr.Dataset) and ds!=None: 
        raise RuntimeError("'ds' needs to be a xarray.Dataset or None")
    if not isinstance(varList, list): raise RuntimeError("'varList' needs to be a list")
    if not isinstance(lonRange, list): raise RuntimeError("'lonRange' needs to be a list")
    if not isinstance(latRange, list): raise RuntimeError("'latRange' needs to be a list")
    if not isinstance(depthRange, list): raise RuntimeError("'depthRange' needs to be a list")
    if not isinstance(timeRange, list): raise RuntimeError("'timeRange' needs to be a list")
    if not isinstance(timeFreq, str): raise RuntimeError("'timeFreq' needs to be a string")
    if not isinstance(sampMethod, str): raise RuntimeError("'sampMethod' needs to be a string")
    if not isinstance(interpC, bool): raise RuntimeError("'interpC' needs to be a Boolean")
    if not isinstance(saveNetCDF, str) and saveNetCDF!=False: 
        raise RuntimeError("'saveNetCDF' needs to be False or a string")
    if not isinstance(plotMap, bool): raise RuntimeError("'plotMap' needs to be a Boolean")
    
    # Load dataset if not provided
    if ds is None: ds, grid = generate_ds_grid()
    
    # Choose variables
    notimeList = []
    for varName in ds.variables:
        if all(x != 'time' for x in ds[varName].dims) or (varName=='time'):
                notimeList.append(varName)
    toDrop = list(set(ds.variables)-set(varList)-set(notimeList))
    ds = ds.drop(toDrop)
    
    # Cut dataset
    ds = ds.sel(time = slice(min(timeRange),        max(timeRange)),
                Xp1  = slice(min(lonRange),         max(lonRange)),
                Yp1  = slice(min(latRange),         max(latRange)),
                Zp1  = slice(min(depthRange),       max(depthRange)))
    ds = ds.sel(X    = slice(min(ds['Xp1'].values), max(ds['Xp1'].values)),
                Y    = slice(min(ds['Yp1'].values), max(ds['Yp1'].values)),
                Z    = slice(min(ds['Zp1'].values), max(ds['Zp1'].values)))
    ds = ds.sel(Zu   = slice(min(ds['Z'].values)  , max(ds['Zp1'].values)),
                Zl   = slice(min(ds['Zp1'].values), max(ds['Z'].values)))

    # Resample
    ds_time   = ds.drop(set(notimeList)-set(['time']))
    ds_notime = ds.drop(set(ds.variables)-set(notimeList))
    ds_notime = ds_notime.drop('time')
    if sampMethod=='snapshot': 
        ds_time = ds_time.resample(time=timeFreq,keep_attrs=True).first(skipna=False)
    elif sampMethod=='mean':   
        ds_time = ds_time.resample(time=timeFreq,keep_attrs=True).mean()
    ds = xr.merge([ds_time,ds_notime])
    for coord in ds.coords: 
        if coord!='time': ds[coord]=ds_notime[coord]

    # Create grid
    grid = xgcm.Grid(ds,periodic=False)
    
    # Interpolate
    if interpC:
        for varName in varList:
            for dim in ds[varName].dims:
                if len(dim)>1 and dim!='time': 
                    ds[varName] = grid.interp(ds[varName], axis=dim[0], boundary='fill', fill_value=float('nan'))
    
    # Chunk array
    from oceanspy.useful_funcs import smart_chunking    
    ds = smart_chunking(ds)
    
    # Save to NetCDF
    if saveNetCDF:
        # Hello
        start_time = time.time()
        print('Saving to ['+saveNetCDF+']:', end=' ')   
        ds.to_netcdf(saveNetCDF)
        # ByeBye
        elapsed_time = time.time() - start_time
        print(time.strftime('done in %H:%M:%S', time.gmtime(elapsed_time)))
    
    # Plot map
    if plotMap:
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        ax = plt.axes(projection=ccrs.Mercator(ds['X'].values.mean(), ds['Y'].values.min(), ds['Y'].values.max()))
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        mask = ds['HFacC'].isel(Z=0)
        rA = ds['rA']*1.E-6
        rA.where(mask>0).plot.pcolormesh(ax=ax, 
                                         transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label':'[km^2]'});
        plt.title(ds['rA'].attrs.get('description'))
        plt.show()
        
    return ds, grid

