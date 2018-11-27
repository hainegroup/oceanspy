"""
Subsample: extract samples from the original dataset.
"""

# Comments for developers:

# 1) Functions should aim to create synthetic observing networks
#    (e.g., moorings, surveys, drifters, floats, gliders, ...)

# 2) Always add deep_copy option, and print a message

# 3) Keep imported modules secret using _


import xarray as _xr
import xgcm as _xgcm
import numpy as _np
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
    import pandas as _pd

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
        ds = ds.sel(time_midp  = slice(min(ds['time'].values), max(ds['time'].values)))
            
    # Resample in time
    if timeFreq:      
        print('Resampling timeseries')
        ds_withtime = ds.drop([ var for var in ds.variables if not 'time' in ds[var].dims ])
        ds_timeless = ds.drop([ var for var in ds.variables if     'time' in ds[var].dims ])
        if   sampMethod=='snapshot' and timeFreq!=_pd.infer_freq(ds.time.values):
            ds = _xr.merge([ds_timeless, ds_withtime.resample(time=timeFreq, keep_attrs=True).first(skipna=False)])
        elif sampMethod=='mean':
            ds = _xr.merge([ds_timeless, ds_withtime.resample(time=timeFreq, keep_attrs=True).mean()])
            ds.attrs['history']   = timeFreq + '-mean computed offline by OceanSpy'
            
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
    sampMethod: {'snapshot', 'mean'}xgc
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
    import xesmf as _xe

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
            print('Regridding variable [',v,']')
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


def mooring_array(ds,
                  info,
                  lats,  
                  lons,
                  add_bdr    = 0.2,
                  varList    = None,
                  depthRange = None,
                  timeRange  = None,
                  timeFreq   = None,
                  sampMethod = 'snapshot',
                  deep_copy  = False):
    
    """
    Extract a mooring array section: no interpolation!
    The vertical section is obtained following the grid (ZigZag).
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    lats: list
        Latitudes of moorings [degrees N]. 
        It must be monotonically increasing/decreasing or constant.
    lons: list
        Longitudes of moorings [degrees E]
        It must be monotonically increasing/decreasing or constant.
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
    sampMethod: {'snapshot', 'mean'}xgc
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
    
    if   all(sign in _np.sign(_np.diff(lons)) for sign in [-1, 1]):
        raise ValueError('[lons] must be monotonically increasing/decreasing, or constant')
    elif all(sign in _np.sign(_np.diff(lats)) for sign in [-1, 1]):
        raise ValueError('[lats] must be monotonically increasing/decreasing, or constant')
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Message
    print('Extracting mooring array') 
    
    # Cutout
    lat_cutout = sorted([_np.min(lats), _np.max(lats)])
    lat_cutout[0] = lat_cutout[0]-add_bdr
    lat_cutout[1] = lat_cutout[1]+add_bdr
    lon_cutout = sorted([_np.min(lons), _np.max(lons)])
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
    
    # Find path along grid
    indX  = []
    indY  = []
    actualX = []
    actualY = []
    X = ds['X'].values
    Y = ds['Y'].values
    
    for i in range(len(lons)-1): # Loop adjacent moorings
        x0 = lons[i]; y0 = lats[i]
        x1 = lons[i+1]; y1 = lats[i+1]
        x0ind = _np.abs(X - x0).argmin(); y0ind = _np.abs(Y - y0).argmin()
        x1ind = _np.abs(X - x1).argmin(); y1ind = _np.abs(Y - y1).argmin()
        xs = X[[x0ind,x1ind]]; 
        ys = Y[[y0ind,y1ind]];
        thisX = X[_np.where(_np.logical_and(X>=_np.min(xs), X<=_np.max(xs)))]
        thisY = Y[_np.where(_np.logical_and(Y>=_np.min(ys), Y<=_np.max(ys)))]
        
        # Add stations between moorings
        if x0!=x1 and y0!=y1: 
            coefficients = _np.polyfit([x0, x1], [y0, y1], 1)
            m = coefficients[0]
            q = coefficients[1]
            moreX = thisX
            moreY = m*thisX + q

            coefficients = _np.polyfit([y0, y1], [x0, x1], 1)
            m = coefficients[0]
            q = coefficients[1]
            moreX = _np.append(moreX, m*thisY + q)
            moreY = _np.append(moreY, thisY)
        elif x0==x1: # Meridional
            moreX = thisY*0 + thisX
            moreY = thisY
        elif y0==y1: # Zonal
            moreX = thisX
            moreY = thisX*0 + thisY

        # Order extra stations
        if  _np.diff([x0, x1])<0 and _np.diff([y0, y1])<0: # Direction: SouthWest-ward
            moreX, moreY = zip(*sorted(zip(-moreX, -moreY)))
            moreX = [-x for x in moreX]
            moreY = [-y for y in moreY]
        elif _np.diff([x0, x1])<0:                         # Direction: NorthWest-ward
            moreX, moreY = zip(*sorted(zip(-moreX, moreY)))
            moreX = [-x for x in moreX]
            moreY = [y for y in moreY]
        elif _np.diff([y0, y1])<0:                         # Direction: SouthEast-ward
            moreX, moreY = zip(*sorted(zip(moreX, -moreY)))
            moreX = [x for x in moreX]
            moreY = [-y for y in moreY]
        else:                                              # Direction: NorthEast-ward
            moreX, moreY = zip(*sorted(zip(moreX, moreY)))
            moreX = [x for x in moreX]
            moreY = [y for y in moreY]
        
        # Find indexes corresponding to extra stations
        for j in range(len(moreX)):
            thisX = X
            thisY = Y
            if len(indX):
                if   lons[0]<lons[-1]:
                    thisX = _np.where(X>=X[indX[-1]],X,_np.nan)
                elif lons[0]>lons[-1]:
                    thisX = _np.where(X<=X[indX[-1]],X,_np.nan)
                if   lats[0]<lats[-1]:
                    thisY = _np.where(Y>=Y[indY[-1]],Y,_np.nan)
                elif lats[0]>lats[-1]:
                    thisY = _np.where(Y<=Y[indY[-1]],Y,_np.nan)   
            indX = _np.append(indX,_np.nanargmin(_np.abs(thisX - moreX[j]))).astype(int)
            indY = _np.append(indY,_np.nanargmin(_np.abs(thisY - moreY[j]))).astype(int)
            actualX = _np.append(actualX,moreX[j])
            actualY = _np.append(actualY,moreY[j])
    diffX = _np.diff(indX)
    diffY = _np.diff(indY)
    mask = _np.where(_np.logical_or(diffX!=0,diffY!=0))
    indX = indX[mask]
    indY = indY[mask]  
    actualX = actualX[mask]
    actualY = actualY[mask]
    
    # Move along grid (meridional or zonal)
    path_ind_x = _np.asarray([indX[0]])
    path_ind_y = _np.asarray([indY[0]])
    for i in range(1,len(indX)):
        if indX[i]-path_ind_x[-1]!=0 and indY[i]-path_ind_y[-1]!=0:

            coefficients = _np.polyfit([actualX[i], actualX[i-1]], [actualY[i], actualY[i-1]], 1)
            m = coefficients[0]
            q = coefficients[1]
            this_actualX = _np.mean(X[[indX[i],path_ind_x[-1]]])
            this_actualY = m*this_actualX + q

            if lats[-1]>lats[0] and this_actualY>_np.mean(Y[[indY[i],path_ind_y[-1]]]):   # meridional
                path_ind_x = _np.append(path_ind_x,path_ind_x[-1])
                path_ind_y = _np.append(path_ind_y,indY[i])
            elif lats[-1]<lats[0] and this_actualY<_np.mean(Y[[indY[i],path_ind_y[-1]]]): # meridional
                path_ind_x = _np.append(path_ind_x,path_ind_x[-1])
                path_ind_y = _np.append(path_ind_y,indY[i])
            else: # zonal
                path_ind_x = _np.append(path_ind_x,indX[i])
                path_ind_y = _np.append(path_ind_y,path_ind_y[-1])

        path_ind_x = _np.append(path_ind_x,indX[i])
        path_ind_y = _np.append(path_ind_y,indY[i])
    
    # Extract paths
    for var in [var for var in ds.variables if var not in ds.dims]:
        if all(dim in ds[var].dims for dim in ['X', 'Y']):
            ds[var] = ds[var].isel(X = _xr.DataArray(path_ind_x  , dims='dists'),
                                   Y = _xr.DataArray(path_ind_y  , dims='dists'))
        elif all(dim in ds[var].dims for dim in ['Xp1', 'Y']):
            var_x0  = ds[var].isel(Xp1 = _xr.DataArray(path_ind_x  , dims='dists'),
                                   Y   = _xr.DataArray(path_ind_y  , dims='dists')).expand_dims('Xpath')
            var_x1  = ds[var].isel(Xp1 = _xr.DataArray(path_ind_x+1, dims='dists'),
                                   Y   = _xr.DataArray(path_ind_y  , dims='dists')).expand_dims('Xpath')
            ds[var] = _xr.concat([var_x0, var_x1],dim='Xpath') 
        elif all(dim in ds[var].dims for dim in ['X', 'Yp1']):
            var_y0  = ds[var].isel(X   = _xr.DataArray(path_ind_x  , dims='dists'),
                                   Yp1 = _xr.DataArray(path_ind_y  , dims='dists')).expand_dims('Ypath')
            var_y1  = ds[var].isel(X   = _xr.DataArray(path_ind_x  , dims='dists'),
                                   Yp1 = _xr.DataArray(path_ind_y+1, dims='dists')).expand_dims('Ypath')
            ds[var] = _xr.concat([var_y0, var_y1],dim='Ypath')
        elif all(dim in ds[var].dims for dim in ['Xp1', 'Yp1']):
            var_x0_y0 = ds[var].isel(Xp1 = _xr.DataArray(path_ind_x  , dims='dists'),
                                     Yp1 = _xr.DataArray(path_ind_y  , dims='dists')).expand_dims('XYpath')
            var_x1_y0 = ds[var].isel(Xp1 = _xr.DataArray(path_ind_x+1, dims='dists'),
                                     Yp1 = _xr.DataArray(path_ind_y  , dims='dists')).expand_dims('XYpath')
            var_x0_y1 = ds[var].isel(Xp1 = _xr.DataArray(path_ind_x  , dims='dists'),
                                     Yp1 = _xr.DataArray(path_ind_y+1, dims='dists')).expand_dims('XYpath')
            var_x1_y1 = ds[var].isel(Xp1 = _xr.DataArray(path_ind_x+1, dims='dists'),
                                     Yp1 = _xr.DataArray(path_ind_y+1, dims='dists')).expand_dims('XYpath')
            ds[var]   = _xr.concat([var_x0_y0, var_x1_y0, var_x0_y1, var_x1_y1],dim='XYpath')
        elif 'Xp1' in ds[var].dims:
            var_x0  = ds[var].isel(Xp1 = _xr.DataArray(path_ind_x  , dims='dists')).expand_dims('Xpath')
            var_x1  = ds[var].isel(Xp1 = _xr.DataArray(path_ind_x+1, dims='dists')).expand_dims('Xpath')
            ds[var] = _xr.concat([var_x0, var_x1],dim='Xpath') 
        elif 'Yp1' in ds[var].dims:
            var_y0  = ds[var].isel(Yp1 = _xr.DataArray(path_ind_y  , dims='dists')).expand_dims('Ypath')
            var_y1  = ds[var].isel(Yp1 = _xr.DataArray(path_ind_y+1, dims='dists')).expand_dims('Ypath')
            ds[var] = _xr.concat([var_y0, var_y1],dim='Ypath') 
    if 'Xp1' in ds.variables:
        var_x0  = ds['Xp1'].isel(Xp1 = _xr.DataArray(path_ind_x  , dims='dists')).expand_dims('Xpath')
        var_x1  = ds['Xp1'].isel(Xp1 = _xr.DataArray(path_ind_x+1, dims='dists')).expand_dims('Xpath')
        ds['Xp1_array'] = _xr.concat([var_x0, var_x1],dim='Xpath')
    if 'Yp1' in ds.variables:
        var_y0  = ds['Yp1'].isel(Yp1 = _xr.DataArray(path_ind_y  , dims='dists')).expand_dims('Ypath')
        var_y1  = ds['Yp1'].isel(Yp1 = _xr.DataArray(path_ind_y+1, dims='dists')).expand_dims('Ypath')
        ds['Yp1_array'] = _xr.concat([var_y0, var_y1],dim='Ypath')
    if 'XYpath' in ds.dims:
        var_x0 = var_x0.rename({'Xpath':'XYpath'})
        var_x1 = var_x1.rename({'Xpath':'XYpath'})
        var_y0 = var_y0.rename({'Ypath':'XYpath'})
        var_y1 = var_y1.rename({'Ypath':'XYpath'})
        ds['Xp1_XY_array'] = _xr.concat([var_x0, var_x1, var_x0, var_x1],dim='XYpath')
        ds['Yp1_XY_array'] = _xr.concat([var_y0, var_y0, var_y1, var_y1],dim='XYpath')
        
    # Update final dimensions    
    from geopy.distance import great_circle as _great_circle
    lons  = X[path_ind_x]
    lats  = Y[path_ind_y]
    dists = _np.zeros(lons.shape)
    for i in range(1,len(lons)):
        coord1   = (lats[i-1],lons[i-1])
        coord2   = (lats[i],lons[i])
        dists[i] = _great_circle(coord1,coord2).km
    dists = _np.cumsum(dists)
    ds = ds.rename({'dists': 'dist_array'})
    ds['dist_array'] = dists
    ds['dist_array'].attrs['long_name'] = 'Distance from mooring 1'
    ds['dist_array'].attrs['units'] = 'km'
    ds['X_array'] = lons
    ds['X_array'].attrs['long_name'] = 'Longitude'
    ds['Y_array'] = lats
    ds['Y_array'].attrs['long_name'] = 'Latitude'
    for dim in ['X', 'Y', 'Xp1', 'Yp1']:
        if dim in ds.variables or dim in ds.dims: ds = ds.drop(dim)
    info.var_names['dist_array'] = 'dist_array'
    info.var_names['X_array']    = 'X_array'
    info.var_names['Y_array']    = 'Y_array'
    
    # Remove grid
    info.grid    = 'ZigZag path'
    
    return ds, info


def extract_properties(ds, 
                       info, 
                       lats          = None,
                       lons          = None,
                       deps          = None,
                       times         = None,
                       varList       = None,
                       interpmethod  = 'nearest',
                       deep_copy     = False):
    """
    Interpolate Eulerian tracer fields on particle positions.
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    lats, lons, deps: positions (latititudes (deg), longitudes (deg), and depths (m)) of particles to which to interpolate
    lats,lons, deps shape: [ntimes,npart] array
    times: times when properties are requested
    varList: requested properties; default is all variables!
    interpmethod: Interpolation method: only 'nearest' (default) and 'linear' are available
    deep_copy: bool
        If True, deep copy ds and info
        
    Returns
    -------
    ds: xarray.Dataset
    info: open_dataset._info
    """
    
    from scipy.interpolate import RegularGridInterpolator as _RGI
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
    
    # Message
    print('Interpolating')
    
    # Drop variables
    if varList: 
        ds = ds.drop([v for v in ds.variables if (v not in ds.dims and v not in ds.coords and v not in varList)])
            
    
    # Create cutout for faster processing
    bdr = 0.5
    ds_cut, info_cut = cutout(ds,
                                             info,
                                             varList    = varList,
                                             latRange   = [_np.amin(lats.values.ravel())-bdr,
                                                           _np.amax(lats.values.ravel())+bdr],
                                             lonRange   = [_np.amin(lons.values.ravel())-bdr, 
                                                           _np.amax(lons.values.ravel())+bdr],
                                             depthRange = [max([_np.amin(ds['Z'].values.ravel()),_np.amin(deps.values.ravel())-30]),
                                                           min([0,_np.amax(deps.values.ravel())+30])],
                                             timeRange  = [times[0], times[-1]],
                                             timeFreq   = '6H',
                                             sampMethod = 'snapshot',
                                             deep_copy  = True)
    
   
    # Make sure depths are positive
    deps = _xr.ufuncs.fabs(deps)
    ds_cut.coords['Z'] = _xr.ufuncs.fabs(ds_cut['Z'])
        
    # Find interpolated values
    for v in varList:
        if v is varList[0]:
            # Initialize new dataset
            varname = [v, '_lagr']
            ds_lagr = _xr.Dataset({''.join(varname): (['time','particles'], _np.ones((len(times),deps.shape[1])))},
                                  coords={'time': times,
                                          'particles': _np.arange(deps.shape[1])})
        else:
            # add variable to dataset
            ds_lagr.assign({''.join(varname): _np.ones((len(times),deps.shape[1]))})
        
        for t in times:
            partpos = _np.concatenate((deps.sel(time=t),lats.sel(time=t),lons.sel(time=t))).reshape(3,len(deps.sel(time=t))).T
            
            my_interpolating_function = _RGI((ds_cut['Z'].values, ds_cut['Y'].values, ds_cut['X'].values), 
                                                                 ds_cut[v].sel(time=t).values,
                                                                 method=interpmethod)
            
            ds_lagr[v + '_lagr'][_np.where(times == t)[0],:] = my_interpolating_function(partpos)
            
    
    return ds_lagr, info
    


