"""
Subsample: extract samples from the original dataset.
"""

# Comments for developers:

# 1) Functions should aim to create synthetic observing networks
#    (e.g., moorings, surveys, drifters, floats, gliders, ...)

# 2) Always add deep_copy option, and print a message

# 3) Return ds applying _utils.reorder_ds(ds)

# 4) Keep imported modules secret using _


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
        Y0 = ds['Yp1'].sel(Yp1=min(latRange), method='ffill')
        Y1 = ds['Yp1'].sel(Yp1=max(latRange), method='bfill')
        inds = _np.where(ds['Yp1'].isin([Y0, Y1]))
        Yi0 = min(inds[0]); Yi1=max(inds[0])
        if Yi0==Yi1:
            if Yi0>0: Yi0=Yi0-1
            else:     Yi1=Yi1+1
        ds = ds.isel(Yp1  = slice(Yi0, Yi1+1))
        ds = ds.sel(Y  = slice(min(ds['Yp1'].values), max(ds['Yp1'].values)))
        if (len(ds['Yp1']) == tmp_len and info.grid.axes['Y']._periodic is True): periodic.append('Y') 
    if lonRange: 
        tmp_len = len(ds['Xp1'])
        X0 = ds['Xp1'].sel(Xp1=min(lonRange), method='ffill')
        X1 = ds['Xp1'].sel(Xp1=max(lonRange), method='bfill')
        inds = _np.where(ds['Xp1'].isin([X0, X1]))
        Xi0 = min(inds[0]); Xi1=max(inds[0])
        if Xi0==Xi1:
            if Xi0>0: Xi0=Xi0-1
            else:     Xi1=Xi1+1
        ds = ds.isel(Xp1  = slice(Xi0, Xi1+1))
        ds = ds.sel(X  = slice(min(ds['Xp1'].values), max(ds['Xp1'].values)))
        if (len(ds['Xp1']) == tmp_len and info.grid.axes['X']._periodic is True): periodic.append('X')        
    if depthRange: 
        tmp_len = len(ds['Zp1'])
        Z0 = ds['Zp1'].sel(Zp1=max(depthRange), method='ffill')
        Z1 = ds['Zp1'].sel(Zp1=min(depthRange), method='bfill')
        inds = _np.where(ds['Zp1'].isin([Z0, Z1]))
        Zi0 = min(inds[0]); Zi1=max(inds[0])
        if Zi0==Zi1:
            if Zi0>0: Zi0=Zi0-1
            else:     Zi1=Zi1+1
        ds = ds.isel(Zp1  = slice(Zi0, Zi1+1))
        ds = ds.sel(Z  = slice(max(ds['Zp1'].values), min(ds['Zp1'].values)),
                    Zu = slice(max(ds['Zp1'].values), min(ds['Zp1'].values)))
        ds = ds.sel(Zl = slice(max(ds['Zp1'].values), min(ds['Z'].values)))
        if (len(ds['Zp1']) == tmp_len and info.grid.axes['Z']._periodic is True): periodic.append('Z')     
    if timeRange: 
        tmp_len = len(ds['time'])
        T0 = ds['time'].sel(time=min(timeRange), method='ffill')
        T1 = ds['time'].sel(time=max(timeRange), method='bfill')
        inds = _np.where(ds['time'].isin([T0.values, T1.values]))
        Ti0 = min(inds[0]); Ti1=max(inds[0])
        if Ti0==Ti1:
            if Ti0>0: Ti0=Ti0-1
            else:     Ti1=Ti1+1
        ds = ds.isel(time  = slice(Ti0, Ti1+1))
        ds = ds.sel(time_midp  = slice(min(ds['time'].values), max(ds['time'].values)))
        if (len(ds['time']) == tmp_len) and (info.grid.axes['time']._periodic is True): periodic.append('time')    
            
    # Resample in time
    if timeFreq:      
        print('Resampling timeseries')
        ds_time = ds.drop([ var for var in ds.variables if not 'time' in ds[var].dims ])
        ds_time_midp = ds.drop([ var for var in ds.variables if not 'time_midp' in ds[var].dims ])
        ds_timeless = ds.drop([var for var in ds.variables if  any(dim in ['time', 'time_midp'] for dim in ds[var].dims)])
        if   sampMethod=='snapshot' and timeFreq!=_pd.infer_freq(ds.time.values):
            ds = _xr.merge([ds_timeless, 
                            ds_time.resample(time=timeFreq, keep_attrs=True).first(skipna=False),
                            ds_time_midp.resample(time_midp=timeFreq, keep_attrs=True).first(skipna=False)])
        elif sampMethod=='mean':
            ds = _xr.merge([ds_timeless, 
                            ds_time.resample(time=timeFreq, keep_attrs=True).mean(),
                            ds_time_midp.resample(time_midp=timeFreq, keep_attrs=True).mean()])
            ds.attrs['history']   = timeFreq + '-mean computed offline by OceanSpy'
            
    # Create grid 
    info.grid = _xgcm.Grid(ds, periodic=periodic)
    
    return _utils.reorder_ds(ds), info
        
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
    
    return _utils.reorder_ds(ds), info

def mooring_array(ds,
                  info,
                  lats,  
                  lons,
                  varList    = None,
                  depthRange = None,
                  timeRange  = None,
                  timeFreq   = None,
                  sampMethod = 'snapshot',
                  deep_copy  = False):
    
    """
    Extract a mooring array section avoiding interpolation.
    Sections are obtained following the grid.
    Every grid cell can have multiple moorings: 
    1 at the center (e.g., for Temp and S);
    2 at the meridional boundaries (e.g., for V). Dimension Yv=['Yb', 'Yf']
    2 at the zonal boundaries (e.g., for U). Dimension Xu=['Xb', 'Xf']
    4 at the corners (e.g., for momVort3). Dimension XYg=['Xb Yb', 'Xb Yf', 'Xf Yb', 'Xf Yf']
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    lats: list
        Latitudes of moorings [degrees N]. 
    lons: list
        Longitudes of moorings [degrees E]
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
    print('Extracting mooring array')
    
    # Cutout
    ds, info = cutout(ds,
                      info,
                      varList    = varList,
                      latRange   = [min(lats), max(lats)],
                      lonRange   = [min(lons), max(lons)],
                      depthRange = depthRange,
                      timeRange  = timeRange,
                      timeFreq   = timeFreq,
                      sampMethod = sampMethod)
    
    # Add missing variables
    varList = ['X', 'Y', 'Xp1', 'Yp1']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Define variables
    X   = ds['X'];   Y   = ds['Y'];
    Xp1 = ds['Xp1']; Yp1 = ds['Yp1']
    
    # Find nearest coordinates
    near_lons = [X.sel(X=this_lons, method='nearest').values for this_lons in lons]
    near_lats = [Y.sel(Y=this_lats, method='nearest').values for this_lats in lats]

    # Remove duplicates
    diff_lons = _np.diff(near_lons); diff_lats = _np.diff(near_lats)
    to_rem = []
    for k, (dlon, dlat) in enumerate(zip(diff_lons, diff_lats)):
        if dlon==0 and dlat==0: to_rem = to_rem + [k]
    near_lons = [i for j, i in enumerate(near_lons) if j not in to_rem]
    near_lats = [i for j, i in enumerate(near_lats) if j not in to_rem]

    # Add points using linear fit
    # Could it generate infinite loops?
    k=0
    diff_lons = _np.diff(near_lons); diff_lats = _np.diff(near_lats)
    while k!=len(diff_lons)-1:
        K = k
        diff_lons = _np.diff(near_lons); diff_lats = _np.diff(near_lats)
        for k, (dlon, dlat) in enumerate(zip(diff_lons, diff_lats)):
            if k<K: continue
                
            # Additional points
            more_lons= X.where(_xr.ufuncs.logical_and(X>=_np.min(near_lons[k:k+2]),
                                                      X<=_np.max(near_lons[k:k+2])),
                                     drop=True).values
            more_lats= Y.where(_xr.ufuncs.logical_and(Y>=_np.min(near_lats[k:k+2]),
                                                      Y<=_np.max(near_lats[k:k+2])),
                                    drop=True).values
            
            # Sort
            if dlon<0: more_lons = -_np.sort(-more_lons)
            if dlat<0: more_lats = -_np.sort(-more_lats)
            
            if dlon!=0 and dlat!=0:
                if len(more_lons)>2 or len(more_lats)>2:
                    x0=more_lons[0]; x1=more_lons[-1]
                    y0=more_lats[0]; y1=more_lats[-1]
                    if len(more_lats)>len(more_lons):
                        new_lons=(more_lats-y0)*(x1-x0)/(y1-y0)+x0
                        new_lats=more_lats
                    else:
                        new_lons=more_lons
                        new_lats=(more_lons-x0)*(y1-y0)/(x1-x0)+y0

                    # Find nearest coordinates
                    new_lons = [X.sel(X=this_lons, method='nearest').values 
                                for this_lons in new_lons][1:-1]
                    new_lats = [Y.sel(Y=this_lats, method='nearest').values 
                                for this_lats in new_lats][1:-1]

                    # Remove duplicates
                    diff_new_lons = _np.diff(new_lons); diff_new_lats = _np.diff(new_lats)
                    to_rem = []
                    for kk, (dnlon, dnlat) in enumerate(zip(diff_new_lons, diff_new_lats)):
                        if dnlon==0 and dnlat==0: to_rem = to_rem + [kk]
                    new_lons = [i for j, i in enumerate(new_lons) if j not in to_rem]
                    new_lats = [i for j, i in enumerate(new_lats) if j not in to_rem]

                    # Insert
                    near_lons = _np.insert(near_lons,k+1,new_lons)
                    near_lats = _np.insert(near_lats,k+1,new_lats)
                    diff_lons = _np.diff(near_lons); diff_lats = _np.diff(near_lats)
                    break      
                

    # Add points where there is a gap
    diff_lons = _np.diff(near_lons); diff_lats = _np.diff(near_lats)
    add_lons = []
    add_lats = []
    for k, (dlon, dlat) in enumerate(zip(diff_lons, diff_lats)):
        add_lons = add_lons + [float(near_lons[k])]
        add_lats = add_lats + [float(near_lats[k])]

        # Additional points
        more_lons= X.where(_xr.ufuncs.logical_and(X>=_np.min(near_lons[k:k+2]),
                                                  X<=_np.max(near_lons[k:k+2])),
                                 drop=True).values
        more_lats= Y.where(_xr.ufuncs.logical_and(Y>=_np.min(near_lats[k:k+2]),
                                                        Y<=_np.max(near_lats[k:k+2])),
                                drop=True).values

        # Sort
        if dlon<0: more_lons = -_np.sort(-more_lons)
        if dlat<0: more_lats = -_np.sort(-more_lats)

        # Meridional and Zonal
        if   len(more_lons)==1 and len(more_lats)>2: # Meridional
            for _, thislat in enumerate(more_lats[1:-1]):
                add_lons = add_lons + [float(more_lons)]
                add_lats = add_lats + [float(thislat)]
        elif len(more_lats)==1 and len(more_lons)>2: # Zonal
            for _, thislon in enumerate(more_lons[1:-1]):
                add_lons = add_lons + [float(thislon)]
                add_lats = add_lats + [float(more_lats)]

        # Diagonal
        if dlon!=0 and dlat!=0:
            if len(more_lons)==2 and len(more_lats)==2: 
                x_diff = near_lons[k+1] - near_lons[k]
                y_diff = near_lats[k+1] - near_lats[k]
                den = _np.sqrt(y_diff**2 + x_diff**2)

                dist1 = _np.fabs(y_diff*near_lons[k+1] - 
                                 x_diff*near_lats[k] + 
                                 near_lons[k+1]*near_lats[k] - 
                                 near_lats[k+1]*near_lons[k])/den
                dist2 = _np.fabs(y_diff*near_lons[k] - 
                                 x_diff*near_lats[k+1] + 
                                 near_lons[k+1]*near_lats[k] - 
                                 near_lats[k+1]*near_lons[k])/den
                if dist1<dist2:
                    add_lons = add_lons + [float(near_lons[k+1])]
                    add_lats = add_lats + [float(near_lats[k])]
                else:
                    add_lons = add_lons + [float(near_lons[k])]
                    add_lats = add_lats + [float(near_lats[k+1])]
            else: raise ValueError('There is a bug: At this point of the function the gap should be smaller!')
    
    # Add
    add_lons = add_lons + [float(near_lons[k+1])]
    add_lats = add_lats + [float(near_lats[k+1])]
            
    # Create dimensions
    Xc  = _np.asarray(add_lons)
    Yc  = _np.asarray(add_lats)
    Xb  = _np.asarray([Xp1.sel(Xp1=this_lons, method='bfill').values for this_lons in Xc])
    Xf  = _np.asarray([Xp1.sel(Xp1=this_lons, method='ffill').values for this_lons in Xc])
    Yb  = _np.asarray([Yp1.sel(Yp1=this_lats, method='bfill').values for this_lats in Yc])
    Yf  = _np.asarray([Yp1.sel(Yp1=this_lats, method='ffill').values for this_lats in Yc])
    cell = _np.arange(len(Xc))
    Xu  = ['Xb', 'Xf']
    Yv  = ['Yb', 'Yf']
    XYg = ['Xb Yb', 'Xb Yf', 'Xf Yb', 'Xf Yf']
    
    # Create DataArrays
    cell = _xr.DataArray(cell, coords={'cell': cell}, dims=('cell'), 
                        attrs={'long_name': 'cells where moorings are located (adjacent cells have consecutive numbers)'})
    Xu  = _xr.DataArray(Xu,  coords={'Xu': Xu}, dims=('Xu'), 
                        attrs={'long_name': 'X coordinates of moorings on cell boundaries'})
    Yv  = _xr.DataArray(Yv,  coords={'Yv': Yv}, dims=('Yv'), 
                        attrs={'long_name': 'Y coordinates of moorings on cell boundaries'})
    XYg = _xr.DataArray(XYg, coords={'XYg': XYg}, dims=('XYg'), 
                        attrs={'long_name': 'XY coordinates of moorings on cell corners'})
    Xc  = _xr.DataArray(Xc,  coords={'cell': cell}, dims=('cell'), 
                        attrs={'long_name': 'longitude of moorings on cell center',
                               'units': 'degrees_east'})
    Yc  = _xr.DataArray(Yc,  coords={'cell': cell}, dims=('cell'),
                        attrs={'long_name': 'latitude of moorings on cell center',
                               'units': 'degrees_north'})
    Xb  = _xr.DataArray(Xb,  coords={'cell': cell}, dims=('cell'), 
                        attrs={'long_name': 'longitude of moorings on cell boundaries (bfill of Xc)',
                               'units': 'degrees_east'})
    Xf  = _xr.DataArray(Xf,  coords={'cell': cell}, dims=('cell'), 
                        attrs={'long_name': 'longitude of moorings on cell boundaries (ffill of Xc)',
                               'units': 'degrees_east'})
    Yb  = _xr.DataArray(Yb,  coords={'cell': cell}, dims=('cell'), 
                        attrs={'long_name': 'latitude of moorings on cell boundaries (bfill of Yc)',
                               'units': 'degrees_north'})
    Yf  = _xr.DataArray(Yf,  coords={'cell': cell}, dims=('cell'),
                        attrs={'long_name': 'latitude of moorings on cell boundaries (ffill of Yc)',
                               'units': 'degrees_north'})
    
    # Add to dataset
    coords = _xr.Dataset({'cell': cell, 'Xu': Xu, 'Yv': Yv, 'XYg': XYg,
                         'Xc': Xc, 'Yc': Yc, 'Xb': Xb, 'Xf': Xf, 'Yb': Yb, 'Yf': Yf})
    ds = _xr.merge([ds, coords])
    
    # Extract paths
    for var in [var for var in ds.variables if var not in ds.dims]:
        if all(dim in ds[var].dims for dim in ['X', 'Y']):
            ds[var] = ds[var].sel(X = Xc,
                                  Y = Yc)
        elif all(dim in ds[var].dims for dim in ['Xp1', 'Y']):
            var_x0  = ds[var].sel(Xp1 = Xb,
                                  Y   = Yc).expand_dims('Xu')
            var_x1  = ds[var].sel(Xp1 = Xf,
                                   Y  = Yc).expand_dims('Xu')
            ds[var] = _xr.concat([var_x0, var_x1],dim='Xu')
        elif all(dim in ds[var].dims for dim in ['X', 'Yp1']):
            var_y0  = ds[var].sel(X   = Xc,
                                  Yp1 = Yb).expand_dims('Yv')
            var_y1  = ds[var].sel(X   = Xc,
                                  Yp1 = Yf).expand_dims('Yv')
            ds[var] = _xr.concat([var_y0, var_y1],dim='Yv')
        elif all(dim in ds[var].dims for dim in ['Xp1', 'Yp1']):
            var_x0_y0 = ds[var].sel(Xp1 = Xb,
                                    Yp1 = Yb).expand_dims('XYg')
            var_x1_y0 = ds[var].sel(Xp1 = Xb,
                                    Yp1 = Yf).expand_dims('XYg')
            var_x0_y1 = ds[var].sel(Xp1 = Xf,
                                    Yp1 = Yb).expand_dims('XYg')
            var_x1_y1 = ds[var].sel(Xp1 = Xf,
                                    Yp1 = Yf).expand_dims('XYg')
            ds[var]   = _xr.concat([var_x0_y0, var_x1_y0, var_x0_y1, var_x1_y1],dim='XYg')
        elif 'Xp1' in ds[var].dims:
            var_x0  = ds[var].isel(Xp1 = Xb).expand_dims('Xu')
            var_x1  = ds[var].isel(Xp1 = Xf).expand_dims('Xu')
            ds[var] = _xr.concat([var_x0, var_x1],dim='Xu') 
        elif 'Yp1' in ds[var].dims:
            var_y0  = ds[var].isel(Yp1 = Yb).expand_dims('Yv')
            var_y1  = ds[var].isel(Yp1 = Yf).expand_dims('Yv')
            ds[var] = _xr.concat([var_y0, var_y1],dim='Yv') 

    # Drop old variables
    for var in ds.variables:
        if var in ['X', 'Y', 'Xp1', 'Yp1']: ds = ds.drop(var)
    for dim in ds.dims:
        if dim in ['X', 'Y', 'Xp1', 'Yp1']: ds = ds.drop(dim)  
    
    # Add distances
    from geopy.distance import great_circle as _great_circle
    dist = _np.zeros(Xc.shape)
    for i in range(1,len(Xc)):
        coord1   = (Yc[i-1],Xc[i-1])
        coord2   = (Yc[i]  ,Xc[i])
        dist[i] = _great_circle(coord1,coord2).km
    ds['dist_array']  = _xr.DataArray(_np.cumsum(dist), coords={'cell': cell}, dims=('cell'),
                                      attrs={'long_name': 'Cumulative distance from mooring 0',
                                             'units': 'km'})
    
    # Update info
    info.grid    = 'Mooring array'
    for name in ['cell', 'Xu', 'Yv', 'XYg', 'Xc', 'Yc', 'Xb', 'Xf', 'Yb', 'Yf', 'dist_array']: info.var_names[name] = name
    
    return _utils.reorder_ds(ds), info

def extract_properties(ds, 
                       info, 
                       lats          = None,
                       lons          = None,
                       deps          = None,
                       times         = None,
                       varList       = None,
                       interp_method = 'nearest',
                       deep_copy     = False):
    
    """
    Interpolate Eulerian tracer fields on particle positions.
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    lats, lons, deps: numpy arrays of size (ntimes,nparticles)
         positions (latititudes (deg), longitudes (deg), and depths (m)) of particles to which to interpolate
    times: times when properties are requested
    varList: requested properties; default is all variables!
    interpmethod: Interpolation method: default is 'nearest' 
    deep_copy: bool
        If True, deep copy ds and info
        
    Returns
    -------
    ds_lagr: xarray.Dataset
    info: open_dataset._info
    """

    # Make sure lats, lons, deps all have the same size and that times has the correct length
    if not (_np.array_equal(lats.shape,lons.shape) or _np.array_equal(lats.shape,deps.shape) or _np.array_equal(lons.shape,deps.shape)):
        raise RuntimeError("lats, lons, and deps need to have the same shape")
    npart  = lats.shape[1]
    ntimes = lats.shape[0]
    if ntimes != times.size: raise RuntimeError("Length of 'times' array inconsistent with length of particle locations time series")
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
    
    # Message
    print('Interpolating')
    
    # Drop variables
    if varList: 
        ds = ds.drop([v for v in ds.variables if (v not in ds.dims and v not in ds.coords and v not in varList)])
            
    # Create cutout for faster processing
    ds, info = cutout(ds,
                      info,
                      varList    = varList,
                      latRange   = [_np.amin(lats.ravel()),_np.amax(lats.ravel())],
                      lonRange   = [_np.amin(lons.ravel()),_np.amax(lons.ravel())],
                      depthRange = [_np.amin(deps.ravel()),_np.amax(deps.ravel())],
                      timeRange  = [times[0], times[-1]],
                      sampMethod = 'snapshot',
                      deep_copy  = False)    
   
    # Find interpolated values
    ds_lagr = _xr.Dataset({'tempvar': (['time','particle'], _np.ones((len(times),deps.shape[1])))},
                                  coords={'time': times,
                                          'particle': _np.arange(deps.shape[1])})

    for v in varList:
        for t in _np.arange(len(times)):
            xtest = _xr.DataArray(lons[t,:], dims='particle')
            ytest = _xr.DataArray(lats[t,:], dims='particle')
            ztest = _xr.DataArray(deps[t,:], dims='particle')
            var = ds[v].isel(time=t).interp(X=xtest,Y=ytest,Z=ztest, method=interp_method).expand_dims('time')
            if t == 0:
                vartemp = var
            else:
                vartemp = _xr.concat([vartemp,var],dim='time')
        ds_lagr = ds_lagr.assign({v:vartemp})
    ds_lagr = ds_lagr.drop('tempvar')

    # Remove grid
    info.grid  = 'Particle locations'
    
    return _utils.reorder_ds(ds_lagr), info

