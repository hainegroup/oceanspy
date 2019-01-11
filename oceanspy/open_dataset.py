"""
Open dataset: import datasets stored on SciServer.
"""

# Comments for developers: 

# 1) Define new functions similar to exp_ASR to import different datasets (return ds and info).

# 2) Return ds applying _utils.reorder_ds(ds)

# 3) OceanSpy was built around exp_ASR, so the variable names must be consistent with OceanSpy.
#    Use oceanspy.open_dataset._info.var_names to specify variable names, 
#    or rename the variables so they are consistent with exp_ASR.

# 4) Dimension names must be the same as exp_ASR (implement a smarter way in the future??).

# 5) Keep imported modules secret using _

import xarray as _xr
import xgcm as _xgcm
import numpy as _np
import pandas as _pd
from . import utils as _utils

class _info:
    """
    Class containg info for OceanSpy, such as xgcm.Grid, parameters, and variable names.
    """
    def __init__(self, name, grid, parameters, var_names):
        """
        Parameters
        ----------
        name: str
            Name of the dataset
        grid: xgcm.Grid
            Grid
        parameters: dict
            Parameters of the dataset
        var_names: dict
            Variable names using exp_ASR as reference.
            key is exp_ASR name, value is custom name
        """
        self.name       = name
        self.grid       = grid
        self.parameters = parameters
        self.var_names  = var_names

    def __repr__(self):
        summary = ['<oceanspy.open_dataset._info>']
        summary.append("* name: %s" % (self.name)) 
        summary.append("* grid: %s" % isinstance(self.grid, _xgcm.Grid))
        if isinstance(self.grid, str): summary.append("        %s" % self.grid)
        summary.append("* Other info: parameters")
        summary.append("              var_names")
        return '\n'.join(summary)
    
    def to_obj(self, path):
        """
        Save info to object file
    
        Parameters
        ---------- 
        path: str
            Path to which to save info
        """
        import pickle as _pickle
        import os as _os

        print('Saving info to', path)
        if not _os.path.exists(_os.path.dirname(path)): _os.makedirs(_os.path.dirname(path))
        f = open(path,'wb')
        _pickle.dump(self, f)
        f.close()
        
        
        
def exp_ASR(cropped = False, 
            machine = 'sciserver'):
    """
    Same configuration as Almansi et al., 2017 [1]_.
    The atmospheric forcing is provided using the Arctic System Reanalysis (ASR).

    Parameters
    ----------
    cropped: bool
        If True, include diagnostics to close the heat/salt budget. 
        The numerical domain is: [ 72N , 69N] [-22E , -13E]
    machine: str
        Available options: {'sciserver', 'datascope'}.

    Returns
    -------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
        
    REFERENCES
    ----------
    .. [1] Almansi et al., 2017 http://doi.org/10.1175/JPO-D-17-0129.1
    """

    # Message
    print('Opening exp_ASR')

    # Import grid and fields separately, then merge
    if machine.lower() == 'sciserver':
        gridpath = '/home/idies/workspace/OceanCirculation/exp_ASR/grid_glued.nc'
        fldspath = '/home/idies/workspace/OceanCirculation/exp_ASR/result_*/output_glued/*.*_glued.nc'
        croppath = '/home/idies/workspace/OceanCirculation/exp_ASR/result_*/output_glued/cropped/*.*_glued.nc'
    elif machine.lower() == 'datascope':
        gridpath = '/sciserver/oceanography/exp_ASR/grid_glued.nc'
        fldspath = '/sciserver/oceanography/exp_ASR/result_*/output_glued/*.*_glued.nc'
        croppath = '/sciserver/oceanography/exp_ASR/result_*/output_glued/cropped/*.*_glued.nc'
    else: raise RuntimeError("machine = %s not available" % machine.lower())
    gridset = _xr.open_dataset(gridpath,
                               drop_variables = ['XU','YU','XV','YV','RC','RF','RU','RL'])
    fldsset = _xr.open_mfdataset(fldspath,
                                 concat_dim     = 'T',
                                 drop_variables = ['diag_levels','iter'])
    ds = _xr.merge([gridset, fldsset])
    
    # Create horizontal vectors (remove zeros due to exch2)
    ds['X'].values   = ds.XC.mean(dim='Y',   skipna=True)
    ds['Xp1'].values = ds.XG.mean(dim='Yp1', skipna=True)
    ds['Y'].values   = ds.YC.mean(dim='X',   skipna=True)
    ds['Yp1'].values = ds.YG.mean(dim='Xp1', skipna=True)
    ds = ds.drop(['XC','YC','XG','YG'])

    # Read cropped files and crop ds
    if cropped:
        cropset = _xr.open_mfdataset(croppath,
                                     concat_dim     = 'T',
                                     drop_variables = ['diag_levels','iter'])
        cropset = cropset.rename({'Zld000216': 'Zl'})
        ds = ds.isel(X   = slice(min(cropset['X'].values).astype(int)-1, 
                                 max(cropset['X'].values).astype(int)),
                     Xp1 = slice(min(cropset['Xp1'].values).astype(int)-1, 
                                 max(cropset['Xp1'].values).astype(int)),
                     Y   = slice(min(cropset['Y'].values).astype(int)-1, 
                                 max(cropset['Y'].values).astype(int)),
                     Yp1 = slice(min(cropset['Yp1'].values).astype(int)-1,
                                 max(cropset['Yp1'].values).astype(int)))
        ds = ds.isel(Z         = slice(0,cropset['Zmd000216'].size),
                     Zl        = slice(0,cropset['Zmd000216'].size),
                     Zmd000216 = slice(0,cropset['Zmd000216'].size),
                     Zp1       = slice(0,cropset['Zmd000216'].size+1),
                     Zu        = slice(0,cropset['Zmd000216'].size))
        for dim in ['X', 'Xp1', 'Y', 'Yp1']: cropset[dim]=ds[dim]
        ds = _xr.merge([ds, cropset])
    
    # Adjust dimensions creating conflicts and add time midpoints
    ds = ds.rename({'Z': 'Ztmp'})
    ds = ds.rename({'T': 'time', 'Ztmp': 'Z', 'Zmd000216': 'Z'})
    ds = ds.squeeze('Zd000001')
    for dim in ['Z','Zp1', 'Zu','Zl']:
        ds[dim].values   = ds[dim].values
        ds[dim].attrs.update({'positive': 'up'}) 
    time_midp = _xr.DataArray(ds['time'].values[:-1]+
                              ds['time'].diff('time')/2,
                              attrs={'long_name':'model_time midpoints',
                                     'history': 'Computed offline by OceanSpy'})
    time_midp = time_midp.rename({'time': 'time_midp'}).rename('time_midp')
    ds['time_midp'] = time_midp
    
    # Add attribute (snapshot vs average)
    for var in [var for var in ds.variables if 'time' in ds[var].coords]:
        if var in ['ADVr_SLT', 'ADVx_SLT', 'ADVy_SLT', 'ADVr_TH', 'ADVx_TH', 'ADVy_TH', 'DFrI_SLT',
                     'DFrI_TH', 'KPPg_SLT', 'KPPg_TH', 'SFLUX', 'TFLUX', 'oceSPtnd', 'oceQsw_AVG']:
            this_var = ds[var].isel(time=slice(1,None)).drop('time').rename({'time': 'time_midp'})
            this_var['time_midp'] = ds['time_midp']
            ds[var] = this_var
            ds[var].attrs.update({'original_output': 'average'}) 
        elif var!='time':
            ds[var].attrs.update({'original_output': 'snapshot'}) 

    # Adjust attributes creating conflicts
    for var in ds.variables:
        if 'coordinates' in ds[var].attrs: del ds[var].attrs['coordinates']

    # Add parameters
    parameters = {}
    parameters['rho0']     = 1027 # kg/m^3
    parameters['g']        = 9.81 # m/s^2
    parameters['omega']    = 7.292123516990375E-05 # rad/s
    parameters['eq_state'] = 'jmd95' # equation of state
    parameters['c_p']      = 3.986E3 # specific heat [J/kg/K]
    
    # Assign the axis attribute to each dimension, 
    # because it is used by both xgcm and OceanSpy!
    for dim in ['Z', 'X', 'Y', 'time']: ds[dim].attrs.update({'axis': dim})  
    for dim in ['Zp1','Zu','Zl','Xp1','Yp1']: 
        if min(ds[dim].values)<min(ds[dim[0]].values):
            ds[dim].attrs.update({'axis': dim[0], 'c_grid_axis_shift': -0.5})
        elif min(ds[dim].values)>min(ds[dim[0]].values):
            ds[dim].attrs.update({'axis': dim[0], 'c_grid_axis_shift': +0.5})
    ds['time_midp'].attrs.update({'axis': 'time', 'c_grid_axis_shift': +0.5})
    
    # Create xgcm.Grid
    grid = _xgcm.Grid(ds, periodic=False)

    # Create delta time
    ds['dt'] = grid.diff(ds['time'], 'time')/_np.timedelta64(1, 's')
    ds['dt'].attrs.update({'long_name': 'delta time',
                           'units': 's'})
    
    # Variable names dictionary
    # key is exp_ASR name, value is custom name
    # This is exp_ASR, so key and value are the same
    var_names = {}
    for var in ds.variables: var_names[var] = var 
        
    # Create info
    info = _info(name       = 'exp_ASR',
                 grid       = grid,
                 parameters = parameters,
                 var_names  = var_names)
    
    return _utils.reorder_ds(ds), info
    
def exp_ERAI(daily   = False, 
             machine = 'sciserver'):
    """
    Same configuration as Almansi et al., 2017 [1]_.

    Parameters
    ----------
    daily: bool
        If True, include diagnostics stored with daily resolution (SI, oce)
    machine: str
        Available options: {'sciserver', 'datascope'}.

    Returns
    -------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
        
    REFERENCES
    ----------
    .. [1] Almansi et al., 2017 http://doi.org/10.1175/JPO-D-17-0129.1
    """

    # Message
    print('Opening exp_ERAI')

    # Import grid and fields separately, then merge
    if machine.lower() == 'sciserver':
        gridpath = '/home/idies/workspace/OceanCirculation/exp_ERAI/grid_glued.nc'
        fldspath = '/home/idies/workspace/OceanCirculation/exp_ERAI/result_*/output_glued/*.*_glued.nc'
        dailypath = '/home/idies/workspace/OceanCirculation/exp_ERAI/result_*/output_glued/daily/*.*_glued.nc'
    elif machine.lower() == 'datascope':
        gridpath = '/sciserver/oceanography/exp_ERAI/grid_glued.nc'
        fldspath = '/sciserver/oceanography/exp_ERAI/result_*/output_glued/*.*_glued.nc'
        dailypath = '/sciserver/oceanography/exp_ERAI/result_*/output_glued/daily/*.*_glued.nc'
    else: raise RuntimeError("machine = %s not available" % machine.lower())
    gridset = _xr.open_dataset(gridpath,
                               drop_variables = ['XU','YU','XV','YV','RC','RF','RU','RL'])
    fldsset = _xr.open_mfdataset(fldspath,
                                 concat_dim     = 'T',
                                 drop_variables = ['diag_levels','iter'])
    ds = _xr.merge([gridset, fldsset])
    
    # Create horizontal vectors (remove zeros due to exch2)
    ds['X'].values   = ds.XC.mean(dim='Y',   skipna=True)
    ds['Xp1'].values = ds.XG.mean(dim='Yp1', skipna=True)
    ds['Y'].values   = ds.YC.mean(dim='X',   skipna=True)
    ds['Yp1'].values = ds.YG.mean(dim='Xp1', skipna=True)
    ds = ds.drop(['XC','YC','XG','YG'])

    # Read daily files and resample in time
    if daily:
        dailyset = _xr.open_mfdataset(dailypath,
                                      concat_dim     = 'T',
                                      drop_variables = ['diag_levels','iter'])
        # dailyset = dailyset.rename({'Zld000216': 'Zl'})
        
        # Resample in time    
        ds_withtime = ds.drop([ var for var in ds.variables if not 'T' in ds[var].dims ])
        ds_timeless = ds.drop([ var for var in ds.variables if     'T' in ds[var].dims ])
        ds = _xr.merge([ds_timeless, ds_withtime.resample(T='1D', keep_attrs=True).first(skipna=False)])
        for dim in ['X', 'Xp1', 'Y', 'Yp1']: dailyset[dim]=ds[dim]
        ds = _xr.merge([ds, dailyset])
    
    # Adjust dimensions creating conflicts and add time midpoints
    ds = ds.rename({'Z': 'Ztmp'})
    ds = ds.rename({'T': 'time', 'Ztmp': 'Z', 'Zmd000216': 'Z'})
    ds = ds.squeeze('Zd000001')
    for dim in ['Z','Zp1', 'Zu','Zl']:
        ds[dim].values   = ds[dim].values
        ds[dim].attrs.update({'positive': 'up'}) 
    time_midp = _xr.DataArray(ds['time'].values[:-1]+
                              ds['time'].diff('time')/2,
                              attrs={'long_name':'model_time midpoints',
                                     'history': 'Computed offline by OceanSpy'})
    time_midp = time_midp.rename({'time': 'time_midp'}).rename('time_midp')
    ds['time_midp'] = time_midp
        
    # Add attribute (snapshot vs average)
    for var in [var for var in ds.variables if 'time' in ds[var].coords]:
        if var!='time':
            ds[var].attrs.update({'original_output': 'snapshot'}) 

    # Adjust attributes creating conflicts
    for var in ds.variables:
        if 'coordinates' in ds[var].attrs: del ds[var].attrs['coordinates']

    # Add parameters
    parameters = {}
    parameters['rho0']     = 1027 # kg/m^3
    parameters['g']        = 9.81 # m/s^2
    parameters['omega']    = 7.292123516990375E-05 # rad/s
    parameters['eq_state'] = 'jmd95' # equation of state
    parameters['c_p']      = 3.986E3 # specific heat [J/kg/K] 
    
    # Assign the axis attribute to each dimension, 
    # because it is used by both xgcm and OceanSpy!
    for dim in ['Z', 'X', 'Y', 'time']: ds[dim].attrs.update({'axis': dim})  
    for dim in ['Zp1','Zu','Zl','Xp1','Yp1']: 
        if min(ds[dim].values)<min(ds[dim[0]].values):
            ds[dim].attrs.update({'axis': dim[0], 'c_grid_axis_shift': -0.5})
        elif min(ds[dim].values)>min(ds[dim[0]].values):
            ds[dim].attrs.update({'axis': dim[0], 'c_grid_axis_shift': +0.5})
    ds['time_midp'].attrs.update({'axis': 'time', 'c_grid_axis_shift': +0.5})
    
    # Create xgcm.Grid
    grid = _xgcm.Grid(ds, periodic=False)

    # Create delta time
    ds['dt'] = grid.diff(ds['time'], 'time')/_np.timedelta64(1, 's')
    ds['dt'].attrs.update({'long_name': 'delta time',
                           'units': 's'})
    
    # Variable names dictionary
    # key is exp_ASR name, value is custom name
    # This is exp_ASR, so key and value are the same
    var_names = {}
    for var in ds.variables: var_names[var] = var 
        
    # Create info
    info = _info(name       = 'exp_ERAI',
                 grid       = grid,
                 parameters = parameters,
                 var_names  = var_names)
    
    return _utils.reorder_ds(ds), info
