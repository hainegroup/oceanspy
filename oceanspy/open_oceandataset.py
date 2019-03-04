"""
Open OceanDataset objects stored on SciServer.
"""
# TODO: check dask warnings

import xarray as _xr
import warnings as _warnings
from ._oceandataset import OceanDataset as _OceanDataset

def from_netcdf(path):
    """
    Load and decode an oceandataset from a netcdf file.
    
    Parameters
    ----------
    path: str
        Path from which to read
    
    Returns
    -------
    od: OceanDataset
    """
    
    # Check parameters
    if not isinstance(path, str):
        raise TypeError('`path` must be str')
    
    # Open
    print('Opening dataset from [{}]'.format(path))
    ds = _xr.open_dataset(path)
    od = _OceanDataset(ds)
    
    return od



def EGshelfIIseas2km_ERAI(daily     = False, 
                          gridpath  = '/home/idies/workspace/OceanCirculation/exp_ERAI/grid_glued.nc',
                          kppspath  = '/home/idies/workspace/OceanCirculation/exp_ERAI/kpp_state_glued.nc',
                          fldspath  = '/home/idies/workspace/OceanCirculation/exp_ERAI/result_*/output_glued/*.*_glued.nc',
                          dailypath = '/home/idies/workspace/OceanCirculation/exp_ERAI/result_*/output_glued/daily/*.*_glued.nc'):
    """
    High-resolution (~2km) numerical simulation covering the east Greenland shelf (EGshelf), and the Iceland and Irminger Seas (IIseas). 
    Surface forcing based on the global atmospheric reanalysis ERA-Interim (ERAI).
    Model setup: [AHPM17]_.
    
    Parameters
    ----------
    daily: bool
        If True, include diagnostics stored with daily resolution (SI, oce).
        Return everything with daily time frequency (instead of 6H).
    gridpath: str
        grid path. Default is SciServer's path.
    kppspath: str
        kpp_state path. Default is SciServer's path.
    fldspath: str
        Fields path (use * for multiple files). Default is SciServer's path.
    dailypath: str
        Daily fields path (use * for multiple files). Default is SciServer's path.

    Returns
    -------
    od: OceanDataset
        
    References
    ----------
    .. [AHPM17] Almansi, M., T.W. Haine, R.S. Pickart, M.G. Magaldi, R. Gelderloos, and D. Mastropole, 2017: High-Frequency Variability in the Circulation and Hydrography of the Denmark Strait Overflow from a High-Resolution Numerical Model. J. Phys. Oceanogr., 47, 2999–3013, https://doi.org/10.1175/JPO-D-17-0129.1 
    """

    # Check input
    if not isinstance(daily, bool):    raise TypeError('`daily` must be a bool')
    if not isinstance(gridpath, str):  raise TypeError('`gridpath` must be a str')
    if not isinstance(kppspath, str):  raise TypeError('`kppspath` must be a str')
    if not isinstance(fldspath, str):  raise TypeError('`fldspath` must be a str')
    if not isinstance(dailypath, str): raise TypeError('`dailypath` must be a str')
        
    # Message
    name = 'EGshelfIIseas2km_ERAI'
    description = 'High-resolution (~2km) numerical simulation covering the east Greenland shelf (EGshelf), and the Iceland and Irminger Seas (IIseas). Citation: Almansi et al., 2017 - JPO.'
    print('Opening [{}]:\n[{}]'.format(name, description))
    
    # Open, concatenate, and merge
    gridset = _xr.open_dataset(gridpath,
                               drop_variables = ['RC', 'RF', 'RU', 'RL'], 
                               chunks={})
    kppset  = _xr.open_dataset(kppspath, 
                               chunks={})
    fldsset = _xr.open_mfdataset(fldspath,
                                 drop_variables = ['diag_levels','iter'])
    ds = _xr.merge([gridset, kppset, fldsset])

    # Read daily files and resample
    if daily:
        # Open, and concatenate daily files
        dailyset = _xr.open_mfdataset(dailypath,
                                      drop_variables = ['diag_levels','iter'])
        
        # Resample and merge
        ds = _xr.merge([ds.isel(T=slice(0,None,4)), dailyset])
    
    # Squeeze 1D Zs and create Z, Zp1, Zu, and Zl only
    ds = ds.rename({'Z': 'Ztmp'}) 
    ds = ds.rename({'Ztmp': 'Z', 'Zmd000216': 'Z'})
    ds = ds.squeeze('Zd000001')
    for dim in ['Z', 'Zp1', 'Zu', 'Zl']: ds[dim].attrs.update({'positive': 'up'}) 
        
    # Rename time
    ds = ds.rename({'T': 'time'}) 
    
    # Add attribute (snapshot vs average)
    for var in [var for var in ds.variables if ('time' in ds[var].coords and var!='time')]:
        ds[var].attrs.update({'original_output': 'snapshot'}) 
        
    # Add missing names
    ds['U'].attrs['long_name']         = 'Zonal Component of Velocity'
    ds['V'].attrs['long_name']         = 'Meridional Component of Velocity'
    ds['W'].attrs['long_name']         = 'Vertical Component of Velocity'
    ds['phiHyd'].attrs['long_name']    = 'Hydrostatic Pressure Pot.(p/rho) Anomaly'
    ds['phiHydLow'].attrs['long_name'] = 'Depth integral of (rho -rhoconst) * g * dz / rhoconst'
    
    # Add missing units
    for varName in ['drC', 'drF', 'dxC', 'dyC', 'dxF', 'dyF', 'dxG', 'dyG', 'dxV', 'dyU', 'R_low']:
        ds[varName].attrs['units'] = 'm'
    for varName in ['rA', 'rAw', 'rAs', 'rAz']:
        ds[varName].attrs['units'] = 'm^2'
    for varName in ['fCori', 'fCoriG']:
        ds[varName].attrs['units'] = '1/s'
    for varName in ['Ro_surf']:
        ds[varName].attrs['units'] = 'kg/m^3'
    for varName in ['Depth']:
        ds[varName].attrs['units'] = 'm'
    for varName in ['HFacC', 'HFacW', 'HFacS']:
        ds[varName].attrs['units'] = '-'
    for varName in ['S']:
        ds[varName].attrs['units'] = 'psu'
    for varName in ['phiHyd', 'phiHydLow']:
        ds[varName].attrs['units'] = 'm^2/s^2'
    
    # Consistent chunkink
    chunks = {**ds.sizes,
              'time': ds['Temp'].chunks[ds['Temp'].dims.index('time')]}
    ds = ds.chunk(chunks)
    
    # Initialize OceanDataset
    od = _OceanDataset(ds).set_name(name).set_description(description)
    od = od.set_coords(fillna=True, coords1Dfrom2D=True)
    grid_coords = {'Y'    : {'Y': None, 'Yp1': 0.5},
                   'X'    : {'X': None, 'Xp1': 0.5},
                   'Z'    : {'Z': None, 'Zp1': 0.5, 'Zu': -0.5, 'Zl': 0.5},
                   'time' : {'time': -0.5}}
    od = od.set_grid_coords(grid_coords = grid_coords, add_midp=True, overwrite=False)
    od = od.set_parameters({'rSphere' : 6.371E3,                # km None: cartesian
                            'eq_state': 'jmd95',                # jmd95, mdjwf
                            'rho0'    : 1027,                   # kg/m^3  TODO: None: compute volume weighted average
                            'g'       : 9.81,                   # m/s^2
                            'eps_nh'  : 0,                      # 0 is hydrostatic
                            'omega'   : 7.292123516990375E-05,  # rad/s
                            'c_p'     : 3.986E3                 # specific heat [J/kg/K]
                            })
    od = od.set_projection('Mercator', 
                           central_longitude=od.dataset['X'].mean().values, 
                           min_latitude=od.dataset['Y'].min().values, 
                           max_latitude=od.dataset['Y'].max().values, 
                           globe=None, 
                           latitude_true_scale=od.dataset['Y'].mean().values)
    
    return od


def EGshelfIIseas2km_ASR(cropped   = False, 
                         gridpath = '/home/idies/workspace/OceanCirculation/exp_ASR/grid_glued.nc',
                         kppspath = '/home/idies/workspace/OceanCirculation/exp_ASR/kpp_state_glued.nc',
                         fldspath = '/home/idies/workspace/OceanCirculation/exp_ASR/result_*/output_glued/*.*_glued.nc',
                         croppath = '/home/idies/workspace/OceanCirculation/exp_ASR/result_*/output_glued/cropped/*.*_glued.nc'):
    """
    High-resolution (~2km) numerical simulation covering the east Greenland shelf (EGshelf),and the Iceland and Irminger Seas (IIseas). 
    Surface forcing based on the regional atmospheric Arctic System Reanalysis (ASR).
    Model setup: [AHPM17]_.

    Parameters
    ----------
    cropped: bool
        If True, include diagnostics to close the heat/salt budget.
        Return everything in a smaller domain: [ 72N , 69N] [-22E , -13E]
    gridpath: str
        grid path. Default is SciServer's path.
    kppspath: str
        kpp_state path. Default is SciServer's path.
    fldspath: str
        Fields path (use * for multiple files). Default is SciServer's path.
    croppath: str
        Croppped fields path (use * for multiple files). Default is SciServer's path.

    Returns
    -------
    od: OceanDataset
      
    References
    ----------
    .. [AHPM17] Almansi, M., T.W. Haine, R.S. Pickart, M.G. Magaldi, R. Gelderloos, and D. Mastropole, 2017: High-Frequency Variability in the Circulation and Hydrography of the Denmark Strait Overflow from a High-Resolution Numerical Model. J. Phys. Oceanogr., 47, 2999–3013, https://doi.org/10.1175/JPO-D-17-0129.1
    """

    # Check input
    if not isinstance(cropped, bool): raise TypeError('`cropped` must be bool')
    if not isinstance(gridpath, str): raise TypeError('`gridpath` must be str')
    if not isinstance(kppspath, str): raise TypeError('`kppspath` must be str')
    if not isinstance(fldspath, str): raise TypeError('`fldspath` must be str')
    if not isinstance(croppath, str): raise TypeError('`croppath` must be str')
    
    # Message
    name = 'EGshelfIIseas2km_ASR'
    description = 'High-resolution (~2km) numerical simulation covering the east Greenland shelf (EGshelf), and the Iceland and Irminger Seas (IIseas). Citation: Almansi et al., 2017 - JPO.'
    print('Opening [{}]:\n[{}]'.format(name, description))
    
    # Open, concatenate, and merge
    gridset = _xr.open_dataset(gridpath,
                               drop_variables = ['RC', 'RF', 'RU', 'RL'],
                               chunks={})
    kppset  = _xr.open_dataset(kppspath,
                               chunks={})
    fldsset = _xr.open_mfdataset(fldspath,
                                 drop_variables = ['diag_levels','iter'])
    ds = _xr.merge([gridset, kppset, fldsset])
    
    # Squeeze 1D Zs and create Z, Zp1, Zu, and Zl only
    ds = ds.rename({'Z': 'Ztmp'}) 
    ds = ds.rename({'Ztmp': 'Z', 'Zmd000216': 'Z'})
    ds = ds.squeeze('Zd000001')
    
    # Read cropped files and resample
    if cropped:
        
        # Open, and concatenate cropped files
        cropset = _xr.open_mfdataset(croppath,
                                     drop_variables = ['diag_levels','iter'])
        
        # Fix vertical dimensions 
        cropset = cropset.rename({'Zld000216': 'Zl', 'Zmd000216': 'Z'})
        cropset = cropset.squeeze('Zd000001')
        for dim in ['Zl', 'Z']: cropset[dim] = ds[dim].isel({dim: cropset[dim]})
            
        # Crop and merge
        ds = ds.sel({dim:slice(cropset[dim].isel({dim: 0}).values, cropset[dim].isel({dim: -1}).values) for dim in cropset.dims})
        ds = ds.isel(Zp1=slice(len(cropset['Z'])+1), Zu=slice(len(cropset['Z'])))
        ds = _xr.merge([ds, cropset])
    
    # Add sign
    for dim in ['Z', 'Zp1', 'Zu', 'Zl']: ds[dim].attrs.update({'positive': 'up'}) 
    
    # Rename time
    ds = ds.rename({'T': 'time'}) 
    
    # Add attribute (snapshot vs average)
    for var in [var for var in ds.variables if ('time' in ds[var].coords and var!='time')]:
        if cropped and var in cropset.variables: 
            ds[var] = ds[var].drop('time').isel(time=slice(1, None)).rename({'time': 'time_midp'})
            Time = 'average'
        else:      
            Time = 'snapshot'
        ds[var].attrs.update({'original_output': Time}) 
    
    # Add missing names
    ds['U'].attrs['long_name']         = 'Zonal Component of Velocity'
    ds['V'].attrs['long_name']         = 'Meridional Component of Velocity'
    ds['W'].attrs['long_name']         = 'Vertical Component of Velocity'
    ds['phiHyd'].attrs['long_name']    = 'Hydrostatic Pressure Pot.(p/rho) Anomaly'
    ds['phiHydLow'].attrs['long_name'] = 'Depth integral of (rho -rhoconst) * g * dz / rhoconst'
    
    # Add missing units
    for varName in ['drC', 'drF', 'dxC', 'dyC', 'dxF', 'dyF', 'dxG', 'dyG', 'dxV', 'dyU', 'R_low']:
        ds[varName].attrs['units'] = 'm'
    for varName in ['rA', 'rAw', 'rAs', 'rAz']:
        ds[varName].attrs['units'] = 'm^2'
    for varName in ['fCori', 'fCoriG']:
        ds[varName].attrs['units'] = '1/s'
    for varName in ['Ro_surf']:
        ds[varName].attrs['units'] = 'kg/m^3'
    for varName in ['Depth']:
        ds[varName].attrs['units'] = 'm'
    for varName in ['HFacC', 'HFacW', 'HFacS']:
        ds[varName].attrs['units'] = '-'
    for varName in ['S']:
        ds[varName].attrs['units'] = 'psu'
    for varName in ['phiHyd', 'phiHydLow']:
        ds[varName].attrs['units'] = 'm^2/s^2'
    
    # Consistent chunkink
    chunks = {**ds.sizes,
              'time': ds['Temp'].chunks[ds['Temp'].dims.index('time')]}
    if cropped:
        chunks = {**chunks, 
                  'time_midp': ds['ADVr_SLT'].chunks[ds['ADVr_SLT'].dims.index('time_midp')]}
    ds = ds.chunk(chunks)
    
    # Initialize OceanDataset
    od = _OceanDataset(ds).set_name(name).set_description(description)
    od = od.set_coords(fillna=True, coords1Dfrom2D=True)
    grid_coords = {'Y'    : {'Y': None, 'Yp1': 0.5},
                   'X'    : {'X': None, 'Xp1': 0.5},
                   'Z'    : {'Z': None, 'Zp1': 0.5, 'Zu': 0.5, 'Zl': -0.5},
                   'time' : {'time': -0.5}}
    od = od.set_grid_coords(grid_coords = grid_coords, add_midp=True)
    od = od.set_projection('Mercator', 
                           central_longitude=od.dataset['X'].mean().values, 
                           min_latitude=od.dataset['Y'].min().values, 
                           max_latitude=od.dataset['Y'].max().values, 
                           globe=None, 
                           latitude_true_scale=od.dataset['Y'].mean().values)
        
    return od        




def exp_Arctic_Control(gridpath  = '/home/idies/workspace/OceanCirculation/exp_Arctic_Control/GRID/grid_glued_swapped.nc',
                       fldspath  = '/home/idies/workspace/OceanCirculation/exp_Arctic_Control/days*/DIAGS/*.nc',
                       statepath = '/home/idies/workspace/OceanCirculation/exp_Arctic_Control/days*/STATE/*.nc'):
    """
    Arctic configuration 

    Parameters
    ----------
    gridpath: str
        grid path. Default is SciServer's path.
    fldspath: str
        Fields path (use * for multiple files). Default is SciServer's path.
    statepath: str. 
        state path. Default is SciServer's path.
        
    Returns
    -------
    od: OceanDataset
    """
    
    # Check input
    if not isinstance(gridpath, str):  raise TypeError('`gridpath` must be str')
    if not isinstance(fldspath, str):  raise TypeError('`fldspath` must be str')
    if not isinstance(statepath, str): raise TypeError('`statepath` must be str')
       
    # Message
    name = 'exp_Arctic_Control'
    description = 'Curvilinear grid test. Setup by Dr. Renske Gelderloos'
    print('Opening [{}]:\n[{}]'.format(name, description))
    
    # TODO: inform Renske!
    # It looks like there's something weird with SIGMA0 and xarray can't infer the concat dimension.
    # Not sure where is the problem, and I'm not reading it for now.
    import glob as _glob
    fldspath = [path for path in _glob.glob(fldspath) if path[-9:]!='SIGMA0.nc']
    
    # Open, concatenate, and merge
    gridset  = _xr.open_dataset(gridpath,
                                drop_variables = ['RC','RF','RU','RL'],
                                chunks={})
    fldsset  = _xr.open_mfdataset(fldspath,
                                  drop_variables = ['diag_levels','iter'])
    stateset = _xr.open_mfdataset(statepath)
    ds = _xr.merge([gridset, fldsset, stateset])  

    # Fix vertical dimensions 
    ds = ds.rename({'Z': 'Ztmp'}) 
    ds = ds.rename({'Ztmp': 'Z', 'Zmd000050': 'Z', 'Zld000050': 'Zl'})
    ds = ds.squeeze('Zd000001')
    for dim in ['Z', 'Zp1', 'Zu', 'Zl']: ds[dim].attrs.update({'positive': 'up'}) 

    # Rename time and add attributes
    ds = ds.rename({'T': 'time'})
    
    # Add attribute (snapshot vs average)
    for var in [var for var in ds.variables if ('time' in ds[var].coords and var!='time')]:
        if var in fldsset.variables: 
            ds[var] = ds[var].drop('time').isel(time=slice(1, None)).rename({'time': 'time_midp'})
            Time = 'average'
        else:      
            Time = 'snapshot'
        ds[var].attrs.update({'original_output': Time}) 
    
    # Add missing names
    ds['U'].attrs['long_name'] = 'Zonal Component of Velocity'
    ds['V'].attrs['long_name'] = 'Meridional Component of Velocity'
    ds['W'].attrs['long_name'] = 'Vertical Component of Velocity'
    
    # Add missing units
    for varName in ['drC', 'drF', 'dxC', 'dyC', 'dxF', 'dyF', 'dxG', 'dyG', 'dxV', 'dyU', 'R_low']:
        ds[varName].attrs['units'] = 'm'
    for varName in ['rA', 'rAw', 'rAs', 'rAz']:
        ds[varName].attrs['units'] = 'm^2'
    for varName in ['fCori', 'fCoriG']:
        ds[varName].attrs['units'] = '1/s'
    for varName in ['Ro_surf']:
        ds[varName].attrs['units'] = 'kg/m^3'
    for varName in ['Depth']:
        ds[varName].attrs['units'] = 'm'
    for varName in ['HFacC', 'HFacW', 'HFacS']:
        ds[varName].attrs['units'] = '-'
    for varName in ['S']:
        ds[varName].attrs['units'] = 'psu'
    for varName in ['AngleCS', 'AngleSN']:
        ds[varName].attrs['units'] = '-'
    
    # Consistent chunkink
    chunks = {**ds.sizes,
              'time': ds['Temp'].chunks[ds['Temp'].dims.index('time')],
              'time_midp': ds['TFLUX'].chunks[ds['TFLUX'].dims.index('time_midp')]}
    ds = ds.chunk(chunks)
    
    # Initialize OceanDataset
    od = _OceanDataset(ds).set_name(name).set_description(description)
    od = od.set_coords(coordsUVfromG=True)
    grid_coords = {'Y'    : {'Y': None, 'Yp1': 0.5},
                   'X'    : {'X': None, 'Xp1': 0.5},
                   'Z'    : {'Z': None, 'Zp1': 0.5, 'Zu': 0.5, 'Zl': -0.5},
                   'time' : {'time': -0.5}}
    od = od.set_grid_coords(grid_coords = grid_coords, add_midp=True)
    od = od.set_parameters({'rSphere' : 6.371E3,                # km None: cartesian
                            'eq_state': 'jmd95',                # jmd95, mdjwf
                            'rho0'    : 1027.5,                   # kg/m^3  TODO: None: compute volume weighted average
                            'g'       : 9.81,                   # m/s^2
                            'eps_nh'  : 0,                      # 0 is hydrostatic
                            'omega'   : 7.292123516990375E-05,  # rad/s
                            'c_p'     : 3.986E3                 # specific heat [J/kg/K]
                            })
    od = od.set_projection('NorthPolarStereo')
    return od
    


def EGshelfSJsec500m(Hydrostatic = True,
                     sixH        = False,
                     resultpath  = '/home/idies/workspace/OceanCirculation/fromMarcello/'):
    """
    Very high-resolution (500m) numerical simulation covering the east Greenland shelf (EGshelf) and the Spill Jet section (SJsec).
    Both hydrostatic and non-hydrostatic setup are available.
    
    Model setup: [MaHa15]_.

    Parameters
    ----------
    sixH: bool
        If True, include diagnostics stored with 6-hour resolution (EXF, MXLDEPTH).
        Return everything with 6H time frequency (instead of 3H).
    Hydrostatic: bool
        If True,  return hydrostatic setup.
        If False, return non-hydrostatic setup.
    resultpath: str
        Path containing .data and .meta files. Default is SciServer's path.
        
    Returns
    -------
    od: OceanDataset
        
    References
    ----------
    .. [MaHa15] Marcello G. Magaldi, Thomas W.N. Haine, Hydrostatic and non-hydrostatic simulations of dense waters cascading off a shelf: The East Greenland case, Deep Sea Research Part I: Oceanographic Research Papers, Volume 96, 2015, Pages 89-104, ISSN 0967-0637, https://doi.org/10.1016/j.dsr.2014.10.008.
    """
    
    # Check input
    if not isinstance(Hydrostatic, bool): raise TypeError('`Hydrostatic` must be bool')
    if not isinstance(sixH, bool):        raise TypeError('`sixH` must be bool')
    if not isinstance(resultpath, str):   raise TypeError('`resultpath` must be str')
    
    # Import modules
    import xmitgcm as _xmitgcm
    
    # Hydrostatic switch
    if Hydrostatic: expname = 'exp1/'
    else:           expname = 'exp6/'
    resultpath = resultpath + expname + 'all_result'
    
    # Message
    name = 'EGshelfSJsec500m'
    description = 'Very high-resolution (500m) numerical simulation covering the east Greenland shelf (EGshelf) and the Spill Jet section (SJsec). Citation: Magaldi and Haine, 2015 - DSR.'
    print('Opening [{}]:\n[{}]'.format(name, description))
    
    # We can't open everything at once because: EXF and state have different frequency, 
    # and PH and PHL are missing the first iter.
    
    # Var names
    var3H     = ['Eta', 'S', 'T', 'U', 'V', 'W']
    var3H_no0 = ['PH', 'PHL']
    var6H     = ['EXFaqh', 'EXFatemp', 'EXFempmr', 'EXFhl', 'EXFhs', 'EXFlwnet', 'EXFswnet', 'EXFuwind', 'EXFvwind']
    var6H_no0 = ['MXLDEPTH']
    
    # Add EXF
    varFirst = var3H
    var_no0  = var3H_no0 
    varAll   = varFirst + var_no0 
    diter    = 1800
    fiter    = 1324800
    if sixH:
        var_no0  = var_no0 + var6H_no0
        varAll   = varAll + var6H + var6H_no0
        varFirst = varFirst + var6H
        diter    = 3600
        fiter    = 1321200

    # Open dataset (ignore xmitgcm warnings)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        ds = _xmitgcm.open_mdsdataset(resultpath,
                                      iters   = list(range(diter,fiter+diter,diter)),
                                      prefix  = varAll,
                                      delta_t = 6,
                                      geometry = 'sphericalpolar',
                                      ref_date='2003-06-01',
                                      ignore_unknown_vars=True)

        # Add first iter
        ds_i0 = _xmitgcm.open_mdsdataset(resultpath,
                                         iters   = 0,
                                         prefix  = varFirst,
                                         delta_t = 6,
                                         geometry = 'sphericalpolar',
                                         ref_date='2003-06-01',
                                         ignore_unknown_vars=True)
        
    for var in var_no0:
        ds_i0[var] = _xr.zeros_like(ds[var].isel(time=0).expand_dims('time'))
        ds_i0[var] = ds_i0[var].where(ds_i0['time']!=ds_i0['time'])
    ds = _xr.concat([ds_i0, ds], 'time')
    
    # Vertical dimensions
    for dim in ['Z', 'Zp1', 'Zu', 'Zl']: ds[dim].attrs.update({'positive': 'up'})
    
    # Add attribute (snapshot vs average)
    for var in [var for var in ds.variables if ('time' in ds[var].coords and var!='time')]:
        if var=='MXLDEPTH': Time = 'average'
        else:               Time = 'snapshot'
        ds[var].attrs.update({'original_output': Time}) 
    
    # Rename variables (We could use aliases instead)
    ds = ds.rename({'XC': 'X', 'YC': 'Y', 'XG': 'Xp1', 'YG': 'Yp1',
                    'T': 'Temp', 'hFacC': 'HFacC', 'hFacW': 'HFacW', 'hFacS': 'HFacS'})
    
    # Add missing units
    for varName in ['HFacC', 'HFacW', 'HFacS', 'iter']:
        ds[varName].attrs['units'] = '-'
        
    # Consistent chunkink
    chunks = {**ds.sizes,
              'time': ds['Temp'].chunks[ds['Temp'].dims.index('time')]}
    ds = ds.chunk(chunks)
    
    # Initialize OceanDataset
    od = _OceanDataset(ds).set_name(name).set_description(description)
    od = od.set_coords(coords2Dfrom1D=True)
    grid_coords = {'Y'    : {'Y': None, 'Yp1': 0.5},
                   'X'    : {'X': None, 'Xp1': 0.5},
                   'Z'    : {'Z': None, 'Zp1': 0.5, 'Zu': 0.5, 'Zl': -0.5},
                   'time' : {'time': -0.5}}
    od = od.set_grid_coords(grid_coords = grid_coords, add_midp=True)
    if Hydrostatic is False:
        od = od.set_parameters({'eps_nh': 1})
    od = od.set_projection('PlateCarree')
    return od


