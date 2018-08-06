"""
Subsample: extract samples from the original dataset.
"""

# Comments for developers:

# 1) Functions should aim to create synthetic observing networks
#    (e.g., moorings, surveys, drifters, floats, gliders, ...)

# 2) Always add deep_copy option, and print a message

# 3) Keep imported modules secret using _

import numpy as _np
import pandas as _pd
import xarray as _xr
import xgcm as _xgcm
import xesmf as _xe
import time as _time
from . import utils as _utils

def cutout(ds,
           info,
           varList    = None,
           latRange   = None,
           lonRange   = None,
           depthRange = None,
           timeRange  = None,
           timeFreq   = None,
           sampMethod = 'snapshot',
           deep_copy  = False):
    """
    Cutout the original dataset in space and time. 

    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    varList: list or None
        List of variables
    latRange: list or None
        Latitude limits (e.g, [-90, 90])     
    lonRange: list or None
        Longitude limits. (e.g, [-180, 180]) 
    depthRange: list or None
        Depth limits.  (e.g, [0, -float('Inf')])  
    timeRange: list or None
        Time limits.   (e.g, ['2007-09-01T00', '2008-09-01T00'])
    timeFreq: str or None
        Time frequency. Available optionts are pandas Offset Aliases (e.g., '6H'):
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    sampMethod: {'snapshot', 'mean'}
        Downsampling method (only if timeFreq is not None)
    deep_copy: bool
        If True, deep copy ds and info

    Returns
    -------
    ds: xarray.Dataset
    info: open_dataset._info
    """
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
    
    # Message
    print('Cutting out')
    
    # Drop variables
    if varList: 
        ds = ds.drop([v for v in ds.variables if (v not in ds.dims and v not in ds.coords and v not in varList)])
    
    # Cut dataset. 
    # The periodic variable should work if periodic dimensions are used, 
    # but it hasn't been tested
    periodic = [] 
    if latRange: 
        tmp_len = len(ds['Yp1'])
        ds = ds.sel(Yp1  = slice(min(latRange),   max(latRange)))
        if len(ds['Yp1'])<2:  raise RuntimeError("Select a larger latitude range.")
        ds = ds.sel(Y  = slice(min(ds['Yp1'].values), max(ds['Yp1'].values)))
        if (len(ds['Yp1']) == tmp_len and info.grid.axes['Y']._periodic is True): periodic.append('Y') 
    if lonRange: 
        tmp_len = len(ds['Xp1'])
        ds = ds.sel(Xp1  = slice(min(lonRange),   max(lonRange)))
        if len(ds['Xp1'])<2:  raise RuntimeError("Select a larger longitude range.")
        ds = ds.sel(X  = slice(min(ds['Xp1'].values), max(ds['Xp1'].values)))
        if (len(ds['Xp1']) == tmp_len and info.grid.axes['X']._periodic is True): periodic.append('X')        
    if depthRange: 
        tmp_len = len(ds['Zp1'])
        ds = ds.sel(Zp1  = slice(max(depthRange),   min(depthRange)))
        if len(ds['Zp1'])<2:  raise RuntimeError("Select a larger depth range.")
        ds = ds.sel(Z  = slice(max(ds['Zp1'].values), min(ds['Zp1'].values)),
                    Zu = slice(max(ds['Zp1'].values), min(ds['Zp1'].values)))
        ds = ds.sel(Zl = slice(max(ds['Zp1'].values), min(ds['Z'].values)))
        if (len(ds['Zp1']) == tmp_len and info.grid.axes['Z']._periodic is True): periodic.append('Z')     
    if timeRange: 
        tmp_len = len(ds['time'])
        ds = ds.sel(time = slice(min(timeRange),  max(timeRange)))
        if (len(ds['time']) == tmp_len) and (info.grid.axes['time']._periodic is True): periodic.append('time')   
            
    # Resample in time
    if timeFreq:      
        print('Resampling timeseries')
        start_time = _time.time()
        ds_withtime = ds.drop([ var for var in ds.variables if not 'time' in ds[var].dims ])
        ds_timeless = ds.drop([ var for var in ds.variables if     'time' in ds[var].dims ])
        if   sampMethod=='snapshot' and timeFreq!=_pd.infer_freq(ds.time.values):
            ds = _xr.merge([ds_timeless, ds_withtime.resample(time=timeFreq, keep_attrs=True).first(skipna=False)])
        elif sampMethod=='mean':
            ds = _xr.merge([ds_timeless, ds_withtime.resample(time=timeFreq, keep_attrs=True).mean()])
        elapsed_time = _time.time() - start_time

    # Create grid 
    info.grid = _xgcm.Grid(ds, periodic=periodic)
    
    return ds, info
        
def survey(ds,
           info,
           lat1,   
           lon1,
           lat2,
           lon2,
           delta_km,
           add_bdr    = 0.2,
           varList    = None,
           depthRange = None,
           timeRange  = None,
           timeFreq   = None,
           sampMethod = 'snapshot',
           method_xe  = 'bilinear',
           deep_copy  = False):
                    
    """
    Carry out a survey along a great circle trajectory.

    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    lat1: number
        Latitude of vertex 1 [degrees N]
    lon1: number
        Longitude of vertex 1 [degrees E]
    lat2: number
        Latitude of vertex 2 [degrees N]
    lon2: number
        Longitude of vertex 2 [degrees E]
    delta_km: number
        Horizontal resolution [km]
    add_bdr: number
        Increase cutout limits, so vertices don't coincide with the boundaries [degrees].
        Increase add_bdr if there are zeros or nans at the extremities of the survey.
    varList: list or None
        List of variables
    depthRange: list or None
        Depth limits.  (e.g, [0, -float('Inf')])  
    timeRange: list or None
        Time limits.   (e.g, ['2007-09-01T00', '2008-09-01T00'])
    timeFreq: str or None
        Time frequency. Available options are pandas Offset Aliases (e.g., '6H'):
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    sampMethod: {'snapshot', 'mean'}
        Downsampling method (only if timeFreq is not None)
    method_xe: srt
        Regridding alghoritm. Available options are xesmf methods: 
        http://xesmf.readthedocs.io/en/latest/Compare_algorithms.html 
    deep_copy: bool
        If True, deep copy ds and info

    Returns
    -------
    ds: xarray.Dataset
    info: open_dataset._info
    """   
        
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Message
    print('Carrying out survey') 
    
    # Great circle trajectory
    lats, lons, dists = _utils.great_circle_path(lat1,lon1,lat2,lon2,delta_km)
    
    # Cutout
    lat_cutout = sorted([lat1, lat2])
    lat_cutout[0] = lat_cutout[0]-add_bdr
    lat_cutout[1] = lat_cutout[1]+add_bdr
    lon_cutout = sorted([lon1, lon2])
    lon_cutout[0] = lon_cutout[0]-add_bdr
    lon_cutout[1] = lon_cutout[1]+add_bdr
    ds, info = cutout(ds,
                      info,
                      varList    = varList,
                      latRange   = lat_cutout,
                      lonRange   = lon_cutout,
                      depthRange = depthRange,
                      timeRange  = timeRange,
                      timeFreq   = timeFreq,
                      sampMethod = sampMethod)    
    
    # Move everything on same grid
    for v in ds.variables:
        if v in ds.coords: continue
        for d in ds[v].dims:
            if d[0] in ['X', 'Y', 'Z'] and len(d)>1: 
                attrs  = ds[v].attrs
                ds[v]  = info.grid.interp(ds[v], axis=d[0], to='center', boundary='fill', fill_value=_np.nan)
                ds[v].attrs = attrs
    ds = ds.drop([d for d in ds.dims if d not in ['X', 'Y', 'Z', 'time']])
    
    # Interpolate
    ds_in  = ds.rename({'Y': 'lat', 'X': 'lon'})
    ds     = _xr.Dataset({'lat':  (['lat'], lats),
                          'lon':  (['lon'], lons)})
    regridder = _xe.Regridder(ds_in, ds, method_xe)                
    for v in ds_in.variables:
        if   v in ['lon', 'lat']: continue
        elif any(d in ['lat', 'lon'] for d in ds_in[v].dims): 
            ds[v] = regridder(ds_in[v])
            for a in ds_in[v].attrs: ds[v].attrs[a] = ds_in[v].attrs[a]
        else: ds[v] = ds_in[v]
    regridder.clean_weight_file()    
    
    # Extract transect (diagonal)
    ds = ds.sel(lat=_xr.DataArray(lats, dims='dist'),
                        lon=_xr.DataArray(lons, dims='dist'))
    ds['dist'] = dists
    ds['dist'].attrs['long_name'] = 'Distance from vertex 1'
    ds['dist'].attrs['units'] = 'km'
    
    if 'units' in ds_in['lat'].attrs: ds['lat'].attrs['units']  = ds_in['lat'].attrs['units']
    ds['lat'].attrs['long_name'] = 'Latitude'
    if 'units' in ds_in['lat'].attrs: ds['lon'].attrs['units']  = ds_in['lon'].attrs['units']
    ds['lon'].attrs['long_name'] = 'Longitude'
    
    # Update var_names
    ds = ds.rename({'dist': 'dist_VS',
                    'lon' : 'lon_VS',
                    'lat' : 'lat_VS'})
    info.var_names['dist_VS'] = 'dist_VS'
    info.var_names['lon_VS']  = 'lon_VS'
    info.var_names['lat_VS']  = 'lat_VS'
    
    # Remove grid
    info.grid    = 'Vertical section'
    
    return ds, info
