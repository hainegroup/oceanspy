"""
Subsample OceanDataset objects.
"""

# Instructions for developers:
# 1) Every function operates on od, and returns od.
# 2) Only use od._ds and od._grid 
# 3) Make sure you don't lose global attributes (e.g., when merging)
# 4) Always cutout at the beginning, and use **kwargs for customized cutouts
# 5) Add new functions in _subsampleMethdos
# 6) Add new functions in docs/api.rst
# 7) Add tests

import xarray   as _xr
import pandas   as _pd
import numpy    as _np
import copy     as _copy
import warnings as _warnings
import oceanspy as _ospy
import functools as _functools

from . import utils as _utils

try: from geopy.distance import great_circle as _great_circle
except ImportError: pass
try: import xesmf as _xe
except ImportError: pass

def cutout(od,
           varList      = None,
           YRange       = None,
           XRange       = None,
           add_Hbdr     = False,
           mask_outside = False,
           ZRange       = None,
           add_Vbdr     = False,
           timeRange    = None,
           timeFreq     = None,
           sampMethod   = 'snapshot',
           dropAxes     = False):
    """
    Cutout the original dataset in space and time preserving the original grid structure. 

    Parameters
    ----------
    od: OceanDataset
        oceandataset to subsample
    varList: 1D array_like, str, or None
        List of variables (strings). 
    YRange: 1D array_like, scalar, or None
        Y axis limits (e.g., latitudes).   
        If len(YRange)>2, max and min values are used.
    XRange: 1D array_like, scalar, or None
        X axis limits (e.g., longitudes).   
        If len(XRange)>2, max and min values are used.
    add_Hbdr: bool, scal
        If scalar, add (subtract) add_Hbdr to the max (min) values of the horizontal ranges.
        If True, automatically estimate add_Hbdr.
        If False, add_Hbdr is set to zero.
    mask_outside: bool 
        If True, set all values in areas outside specified (Y,X)ranges to NaNs.
        (Useless for rectilinear grids).
    ZRange: 1D array_like, scalar, or None
        Z axis limits.  
        If len(ZRange)>2, max and min values are used.
    timeRange: 1D array_like np.ScalarType, or None
        time axis limits.  
        If len(timeRange)>2, max and min values are used.
    timeFreq: str or None
        Time frequency. Available optionts are pandas Offset Aliases (e.g., '6H'):
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    sampMethod: {'snapshot', 'mean'}
        Downsampling method (only if timeFreq is not None)
    dropAxes: 1D array_like, str, or bool
        List of axes to remove from Grid object if one point only is in the range.
        If True, set dropAxes=od.grid_coords.
        If False, preserve original grid.
    
    Returns
    -------
    od: OceanDataset
        Subsampled oceandataset
        
    Notes
    -----
    If any of the horizontal ranges is not None, 
    the horizontal dimensions of the cutout will have len(Xp1)>len(X) and len(Yp1)>len(Y) 
    even if the original oceandataset had len(Xp1)==len(X) or len(Yp1)==len(Y). 
    """
    
    # Check
    for wrong_dim in ['mooring', 'station', 'particle']:
        if wrong_dim in od._ds.dims and (XRange is not None or YRange is not None):
            raise ValueError('`cutout` cannot subsample in the horizontal plain oceandatasets with dimension [{}]'.format(wrong_dim))
    
    # Convert variables to numpy arrays and make some check
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
    
    if varList is not None:
        varList = _np.asarray(varList, dtype='str')
        if varList.ndim == 0: varList = varList.reshape(1)
        elif varList.ndim >1: raise TypeError('Invalid `varList`')
    
    if not isinstance(add_Hbdr, (float, int, bool)):
        raise TypeError('`add_Hbdr` must be float, int, or bool')
    
    if not isinstance(mask_outside, bool):
        raise TypeError('`add_Hbdr` must be bool')
        
    if YRange is not None:
        YRange  = _np.asarray(YRange, dtype=od._ds['YG'].dtype)
        if YRange.ndim == 0: YRange = YRange.reshape(1)
        elif YRange.ndim >1: raise TypeError('Invalid `YRange`')
    
    if XRange is not None:
        XRange = _np.asarray(XRange, dtype=od._ds['XG'].dtype)
        if XRange.ndim == 0: XRange = XRange.reshape(1)
        elif XRange.ndim >1: raise TypeError('Invalid `XRange`')
    
    if ZRange is not None:
        ZRange = _np.asarray(ZRange, dtype=od._ds['Zp1'].dtype)
        if ZRange.ndim == 0: ZRange = ZRange.reshape(1)
        elif ZRange.ndim >1: raise TypeError('Invalid `ZRange`')
        # Only check ZRange because it's a common mistake
        Zmax = od._ds['Zp1'].max().values
        Zmin = od._ds['Zp1'].min().values
        if any(ZRange<Zmin) or any(ZRange>Zmax):
            _warnings.warn("\nThe depth range is: {}"
                           "\nZRange has values outside the vertical range of the oceandataset.".format([Zmin, Zmax]), stacklevel=2)
            
    if timeRange is not None:
        timeRange  = _np.asarray(timeRange, dtype=od._ds['time'].dtype)
        if timeRange.ndim == 0: timeRange = timeRange.reshape(1)
        elif timeRange.ndim >1: raise TypeError('Invalid `timeRange`')
    
    if not isinstance(timeFreq, (str, type(None))):
        raise TypeError('`timeFreq` must None or str')
        
    sampMethod_list = ['snapshot', 'mean']
    if sampMethod not in sampMethod_list:
        raise ValueError('[{}] is not an available `sampMethod`.'
                         '\nOptions: {}'.format(sampMethod, sampMethod_list))
        
    if not isinstance(dropAxes, bool):
        dropAxes = _np.asarray(dropAxes, dtype='str')
        if dropAxes.ndim == 0: dropAxes = dropAxes.reshape(1)
        elif dropAxes.ndim >1: raise TypeError('Invalid `dropAxes`')
        axis_error = [axis for axis in dropAxes if axis not in od.grid_coords]
        if len(axis_error)!=0:
            raise ValueError('{} are not in od.grid_coords and can not be dropped'.format(axis_error))
        dropAxes = {d: od.grid_coords[d] for d in dropAxes}
    elif dropAxes is True:
        dropAxes = od.grid_coords
        if YRange is None   : dropAxes.pop('Y', None)
        if XRange is None   : dropAxes.pop('X', None)
        if ZRange is None   : dropAxes.pop('Z', None)
        if timeRange is None: dropAxes.pop('time', None)
    else:
        dropAxes = {}
            
    # Message
    print('Cutting out the oceandataset.')
                     
    # Unpack from oceandataset
    od = _copy.copy(od)
    ds = od._ds
    periodic = od.grid_periodic
    
    # Drop variables
    if varList is not None: 
        if isinstance(varList, str): varList = [varList]
        ds = ds.drop([v for v in ds.variables if (v not in ds.dims and v not in ds.coords and v not in varList)])
    
    # ---------------------------
    # Horizontal CUTOUT
    # ---------------------------
    
    if add_Hbdr is True:
        add_Hbdr = (_np.mean([_np.fabs(od._ds['XG'].max() - od._ds['XG'].min()),
                              _np.fabs(od._ds['YG'].max() - od._ds['YG'].min())]) / 
                   _np.mean([len(od._ds['X']), len(od._ds['Y'])]))
    elif add_Hbdr is False:
        add_Hbdr = 0
      
    if add_Vbdr is True:
        add_Vbdr = _np.fabs(od._ds['Zp1'].diff('Zp1')).max().values
    elif add_Vbdr is False:
        add_Vbdr = 0
        
    # Initialize horizontal mask
    if XRange is not None or YRange is not None:
        maskH = _xr.ones_like(ds['XG'])

        if YRange is not None: 
            # Use arrays
            YRange = _np.asarray([_np.min(YRange)-add_Hbdr, _np.max(YRange)+add_Hbdr]).astype(ds['YG'].dtype)

            # Get the closest 
            for i, Y in enumerate(YRange):
                diff = _np.fabs(ds['YG']-Y)
                YRange[i] = ds['YG'].where(diff==diff.min()).min().values     
            maskH = maskH.where(_np.logical_and(ds['YG']>=YRange[0], ds['YG']<=YRange[-1]), 0)
            maskHY = maskH

        if XRange is not None:
            # Use arrays
            XRange = _np.asarray([_np.min(XRange)-add_Hbdr, _np.max(XRange)+add_Hbdr]).astype(ds['XG'].dtype)

            # Get the closest 
            for i, X in enumerate(XRange):
                diff = _np.fabs(ds['XG']-X)
                XRange[i] = ds['XG'].where(diff==diff.min()).min().values     
            maskH = maskH.where(_np.logical_and(ds['XG']>=XRange[0], ds['XG']<=XRange[-1]), 0)

        # Can't be all zeros
        if maskH.sum()==0: raise ValueError('Zero grid points in the horizontal range')

        # Find horizontal indexes
        maskH['Yp1'].values = _np.arange(len(maskH['Yp1']))
        maskH['Xp1'].values = _np.arange(len(maskH['Xp1']))
        dmaskH = maskH.where(maskH, drop=True)
        dYp1 = dmaskH['Yp1'].values
        dXp1 = dmaskH['Xp1'].values
        iY = [_np.min(dYp1), _np.max(dYp1)]
        iX = [_np.min(dXp1), _np.max(dXp1)]
        maskH['Yp1'] = ds['Yp1']
        maskH['Xp1'] = ds['Xp1']
        
        # Original length
        lenY = len(ds['Yp1'])
        lenX = len(ds['Xp1']) 
        
        # Indexis
        if iY[0]==iY[1]:
            if 'Y' not in dropAxes:
                if iY[0]>0: iY[0]=iY[0]-1
                else:       iY[1]=iY[1]+1
        else: dropAxes.pop('Y', None)
                    

        if iX[0]==iX[1]:
            if 'X' not in dropAxes:
                if iX[0]>0: iX[0]=iX[0]-1
                else:       iX[1]=iX[1]+1
        else: dropAxes.pop('X', None)
        
        # Cutout
        ds = ds.isel(Yp1 = slice(iY[0], iY[1]+1),
                     Xp1 = slice(iX[0], iX[1]+1))
        
        if 'X' in dropAxes:
            if iX[0]==len(ds['X']):
                iX[0]=iX[0]-1
                iX[1]=iX[1]-1
            ds = ds.isel(X = slice(iX[0], iX[1]+1))
        elif (('outer' in od._grid.axes['X'].coords and od._grid.axes['X'].coords['outer'].name == 'Xp1') or  
              ('left'  in od._grid.axes['X'].coords and od._grid.axes['X'].coords['left'].name  == 'Xp1')):
            ds = ds.isel(X = slice(iX[0], iX[1]))
        elif   'right' in od._grid.axes['X'].coords and od._grid.axes['X'].coords['right'].name =='Xp1':
            ds = ds.isel(X = slice(iX[0]+1, iX[1]+1))   
        
        if 'Y' in dropAxes:
            if iY[0]==len(ds['Y']):
                iY[0]=iY[0]-1
                iY[1]=iY[1]-1
            ds = ds.isel(Y = slice(iY[0], iY[1]+1))
        elif (('outer' in od._grid.axes['Y'].coords and od._grid.axes['Y'].coords['outer'].name == 'Yp1') or  
              ('left'  in od._grid.axes['Y'].coords and od._grid.axes['Y'].coords['left'].name  == 'Yp1')):
            ds = ds.isel(Y = slice(iY[0], iY[1]))
        elif   'right' in od._grid.axes['Y'].coords and od._grid.axes['Y'].coords['right'].name =='Yp1':
            ds = ds.isel(Y = slice(iY[0]+1, iY[1]+1))
        
        # Cut axis can't be periodic
        if (len(ds['Yp1'])  < lenY or 'Y' in dropAxes) and 'Y' in periodic: periodic.remove('Y')
        if (len(ds['Xp1'])  < lenX or 'X' in dropAxes) and 'X' in periodic: periodic.remove('X')
    
    # ---------------------------
    # Vertical CUTOUT
    # ---------------------------
                            
    # Initialize vertical mask
    maskV = _xr.ones_like(ds['Zp1'])
    
    if ZRange is not None:
        # Use arrays
        ZRange = _np.asarray([_np.min(ZRange)-add_Vbdr, _np.max(ZRange)+add_Vbdr]).astype(ds['Zp1'].dtype)
        
        # Get the closest 
        for i, Z in enumerate(ZRange):
            diff = _np.fabs(ds['Zp1']-Z)
            ZRange[i] = ds['Zp1'].where(diff==diff.min()).min().values     
        maskV = maskV.where(_np.logical_and(ds['Zp1']>=ZRange[0], ds['Zp1']<=ZRange[-1]), 0)                        
    
        # Find vertical indexes
        maskV['Zp1'].values = _np.arange(len(maskV['Zp1']))
        dmaskV = maskV.where(maskV, drop=True)
        dZp1 = dmaskV['Zp1'].values
        iZ = [_np.min(dZp1), _np.max(dZp1)]
        maskV['Zp1'] = ds['Zp1']
        
        # Original length
        lenZ = len(ds['Zp1']) 
        
        # Indexis
        if iZ[0]==iZ[1]:
            if 'Z' not in dropAxes:
                if iZ[0]>0: iZ[0]=iZ[0]-1
                else:       iZ[1]=iZ[1]+1
        else: dropAxes.pop('Z', None)
        
        # Cutout
        ds = ds.isel(Zp1 = slice(iZ[0], iZ[1]+1))
        if 'Z' in dropAxes:
            if iZ[0]==len(ds['Z']):
                iZ[0]=iZ[0]-1
                iZ[1]=iZ[1]-1
            ds = ds.isel(Z = slice(iZ[0], iZ[1]+1))
        else:
            ds = ds.isel(Z = slice(iZ[0], iZ[1]))
        
        if 'Zu' in ds.dims:
            ds = ds.sel(Zu = slice(ds['Zp1'].isel(Zp1=0).values,   ds['Zp1'].isel(Zp1=-1).values))
        
        if 'Zl' in ds.dims:
            ds = ds.sel(Zl = slice(ds['Zp1'].isel(Zp1=0).values,  ds['Z'].isel(Z=-1).values))
        
        # Cut axis can't be periodic
        if (len(ds['Z']) < lenZ or 'Z' in dropAxes) and 'Z' in periodic: periodic.remove('Z')
    
    # ---------------------------
    # Time CUTOUT
    # ---------------------------
    
    # Initialize vertical mask
    maskT = _xr.ones_like(ds['time']).astype('int')
    
    if timeRange is not None:
        
        # Use arrays
        timeRange = _np.asarray([_np.min(timeRange), _np.max(timeRange)]).astype(ds['time'].dtype)
        
        # Get the closest 
        for i, time in enumerate(timeRange):
            if _np.issubdtype(ds['time'].dtype, _np.datetime64):
                diff = _np.fabs(ds['time'].astype('float64') - time.astype('float64'))
            else:
                diff = _np.fabs(ds['time']-time)
            timeRange[i] = ds['time'].where(diff==diff.min()).min().values     
        # return maskT, ds['time'], timeRange[0], timeRange[-1]
        maskT = maskT.where(_np.logical_and(ds['time']>=timeRange[0], ds['time']<=timeRange[-1]), 0)  
        
        # Find vertical indexes
        maskT['time'].values = _np.arange(len(maskT['time']))
        dmaskT = maskT.where(maskT, drop=True)
        dtime = dmaskT['time'].values
        iT = [min(dtime), max(dtime)]
        maskT['time'] = ds['time']
    
        # Original length
        lenT = len(ds['time'])
        
        # Indexis
        if iT[0]==iT[1]:
            if 'time' not in dropAxes:
                if iT[0]>0: iT[0]=iT[0]-1
                else:       iT[1]=iT[1]+1
        else: dropAxes.pop('time', None)
        
        # Cutout
        ds = ds.isel(time = slice(iT[0], iT[1]+1))
        if 'time' in dropAxes:
            if iT[0]==len(ds['time_midp']):
                iT[0]=iT[0]-1
                iT[1]=iT[1]-1
            ds = ds.isel(time_midp = slice(iT[0], iT[1]+1))
        else:
            ds = ds.isel(time_midp = slice(iT[0], iT[1]))
        
        # Cut axis can't be periodic
        if (len(ds['time']) < lenT or 'T' in dropAxes) and 'time' in periodic: periodic.remove('time')
    
    # ---------------------------
    # Horizontal MASK
    # ---------------------------
    
    if mask_outside and (YRange is not None or XRange is not None):
        if YRange is not None: minY = YRange[0]; maxY = YRange[1]
        else:                  minY = ds['YG'].min().values; maxY = ds['YG'].max().values
        if XRange is not None: minX = XRange[0]; maxX = XRange[1]
        else:                  minX = ds['XG'].min().values; maxX = ds['XG'].max().values    
        
        maskC = _xr.where(_np.logical_and(_np.logical_and(ds['YC']>=minY, ds['YC']<=maxY),
                                          _np.logical_and(ds['XC']>=minX, ds['XC']<=maxX)), 1,0).persist()
        maskG = _xr.where(_np.logical_and(_np.logical_and(ds['YG']>=minY, ds['YG']<=maxY),
                                                 _np.logical_and(ds['XG']>=minX, ds['XG']<=maxX)), 1,0).persist()
        maskU = _xr.where(_np.logical_and(_np.logical_and(ds['YU']>=minY, ds['YU']<=maxY),
                                                 _np.logical_and(ds['XU']>=minX, ds['XU']<=maxX)), 1,0).persist()
        maskV = _xr.where(_np.logical_and(_np.logical_and(ds['YV']>=minY, ds['YV']<=maxY),
                                                 _np.logical_and(ds['XV']>=minX, ds['XV']<=maxX)), 1,0).persist()
        for var in ds.data_vars:
            if   set(['X',   'Y']).issubset(ds[var].dims):   ds[var] = ds[var].where(maskC)
            elif set(['Xp1', 'Yp1']).issubset(ds[var].dims): ds[var] = ds[var].where(maskG)
            elif set(['Xp1', 'Y']).issubset(ds[var].dims):   ds[var] = ds[var].where(maskU)
            elif set(['X',   'Yp1']).issubset(ds[var].dims): ds[var] = ds[var].where(maskV)
    
    # ---------------------------
    # TIME RESAMPLING
    # ---------------------------
    # Resample in time
    if timeFreq:
        
        # Infer original frequency
        inFreq=_pd.infer_freq(ds.time.values); 
        if timeFreq[0].isdigit() and not inFreq[0].isdigit(): inFreq='1'+inFreq
            
        # Same frequency: Skip
        if timeFreq==inFreq:
            _warnings.warn("\nInput time freq: [{}] = Output time frequency: [{}]:"
                           "\nSkip time resampling.".format(inFreq, timeFreq), stacklevel=2)
        
        else:
        
            # Remove time_midp and warn
            vars2drop = [var for var in ds.variables if 'time_midp' in ds[var].dims]
            if vars2drop:
                _warnings.warn("\nTime resampling drops variables on `time_midp` dimension."
                               "\nDropped variables: {}.".format(vars2drop), stacklevel=2)
                ds = ds.drop(vars2drop)
            if 'time_midp' in ds.dims: ds = ds.drop('time_midp')
                
            # Snapshot
            if sampMethod=='snapshot': 
                # Find new times
                newtime = ds['time'].sel(time=ds['time'].resample(time=timeFreq).first())

                # Use slice when possible
                inds = [i for i, t in enumerate(ds['time'].values) if t in newtime.values]
                inds_diff = _np.diff(inds)
                if all(inds_diff==inds_diff[0]): 
                    ds = ds.isel(time = slice(inds[0], inds[-1]+1, inds_diff[0]))
                else: 
                    # TODO: is this an xarray bug od just bad chunking/bad coding/bad SciServe compute performances?
                    #       Make test case and open issue!
                    attrs = ds.attrs
                    ds = _xr.concat([ds.sel(time = time) for i, time in enumerate(newtime)], dim='time')
                    ds.attrs = attrs
            # Mean
            elif sampMethod=='mean':

                # Separate time and timeless
                attrs       = ds.attrs
                ds_dims     = ds.drop([var for var in ds.variables if not var in ds.dims])
                ds_time     = ds.drop([var for var in ds.variables if not 'time' in ds[var].dims])
                ds_timeless = ds.drop([var for var in ds.variables if     'time' in ds[var].dims])

                # Resample
                ds_time = ds_time.resample(time=timeFreq).mean('time')

                # Add all dimensions to ds, and fix attributes
                for dim in ds_time.dims:
                    if dim=='time': ds_time[dim].attrs = ds_dims[dim].attrs
                    else:           ds_time[dim] = ds_dims[dim]

                # Merge
                ds = _xr.merge([ds_time, ds_timeless])
                ds.attrs = attrs
                
    # Update oceandataset
    od._ds = ds
    
    # Add time midp
    if timeFreq and 'time' not in dropAxes:
        od = od.set_grid_coords({**od.grid_coords, 'time' : {'time': -0.5}}, add_midp=True, overwrite=True)

    # Drop axes
    grid_coords = od.grid_coords
    for coord in list(grid_coords): 
        if coord in dropAxes: grid_coords.pop(coord, None)
    od = od.set_grid_coords(grid_coords, overwrite=True)
    
    # Cut axis can't be periodic 
    od = od.set_grid_periodic(periodic, overwrite = True)
    
    return od


def mooring_array(od, Ymoor, Xmoor, 
                  **kwargs):
    
    """
    Extract a mooring array section following the grid.
    Trajectories are great circle paths when coordinates are spherical.
    
    Parameters
    ----------
    od: OceanDataset
        od that will be subsampled
    Ymoor: 1D array_like, scalar
        Y coordinates of moorings. 
    Xmoor: 1D array_like, scalar
        X coordinates of moorings.
    **kwargs: 
        Keyword arguments for subsample.cutout

    Returns
    -------
    od: OceanDataset
        Subsampled oceandataset
        
    See Also
    --------
    subsample.cutout
    """       
    
    # Check
    for wrong_dim in ['mooring', 'station', 'particle']:
        if wrong_dim in od._ds.dims:
            raise ValueError('`mooring_array` cannot subsample oceandatasets with dimension [{}]'.format(wrong_dim))
    
    # Convert variables to numpy arrays and make some check
    Ymoor = _np.asarray(Ymoor, dtype=od._ds['YC'].dtype)
    if   Ymoor.ndim == 0: Ymoor = Ymoor.reshape(1)
    elif Ymoor.ndim>1: raise TypeError('Invalid `Y`')

    Xmoor = _np.asarray(Xmoor, dtype=od._ds['XC'].dtype)
    if   Xmoor.ndim == 0: Xmoor = Xmoor.reshape(1)
    elif Xmoor.ndim >1: raise TypeError('Invalid `X`')
            
    # Cutout
    if "YRange"   not in kwargs: kwargs['YRange']   = Ymoor
    if "XRange"   not in kwargs: kwargs['XRange']   = Xmoor
    if "add_Hbdr" not in kwargs: kwargs['add_Hbdr'] = True
    od = od.subsample.cutout(**kwargs)

    # Message
    print('Extracting mooring array.')
    
    # Unpack ds
    ds = od._ds
    
    # Useful variables
    YC = od._ds['YC']
    XC = od._ds['XC']
    R  = od.parameters['rSphere']
    shape   = XC.shape
    Yindex  = XC.dims.index('Y')
    Xindex  = XC.dims.index('X')
    
    # Convert to cartesian if spherical
    if R is not None:
        x, y, z = _utils.spherical2cartesian(Y = Ymoor, X = Xmoor, R = R)
    else:
        x = Xmoor; y = Ymoor; z = _np.zeros(Ymoor.shape)
        
    # Create tree
    tree = od.create_tree(grid_pos = 'C')
            
    # Indexes of nearest grid points
    _, indexes = tree.query(_np.column_stack((x, y, z)))
    indexes    = _np.unravel_index(indexes, shape)
    iY = _np.ndarray.tolist(indexes[Yindex])
    iX = _np.ndarray.tolist(indexes[Xindex])
    
    # Remove duplicates
    diff_iY = _np.diff(iY)
    diff_iX = _np.diff(iX) 
    to_rem = []
    for k, (diY, diX) in enumerate(zip(diff_iY, diff_iX)): 
        if diY==0 and diX==0: to_rem = to_rem + [k]
    iY = _np.asarray([i for j, i in enumerate(iY) if j not in to_rem])
    iX = _np.asarray([i for j, i in enumerate(iX) if j not in to_rem])
    
    # Nearest coordinates
    near_Y   = YC.isel(Y=_xr.DataArray(iY, dims=('tmp')), 
                          X=_xr.DataArray(iX, dims=('tmp'))).values
    near_X   = XC.isel(Y=_xr.DataArray(iY, dims=('tmp')), 
                          X=_xr.DataArray(iX, dims=('tmp'))).values
    
    # Steps
    diff_iY    = _np.fabs(_np.diff(iY))
    diff_iX    = _np.fabs(_np.diff(iX))
    
    # Loop until all steps are 1
    while any(diff_iY+diff_iX!=1):
        
        # Find where need to add grid points
        k = _np.argwhere(diff_iY+diff_iX!=1)[0][0]
        lat0 = near_Y[k];   lon0 = near_X[k]
        lat1 = near_Y[k+1]; lon1 = near_X[k+1]
        
        # Find grid point in the middle 
        if R is not None:
            # SPHERICAL: follow great circle path
            dist = _great_circle((lat0, lon0), (lat1, lon1), radius = R).km
            
            # Divide dist by 2.1 to make sure that returns 3 points
            dist = dist/2.1
            this_Y, this_X, this_dists = _utils.great_circle_path(lat0, lon0, lat1, lon1, dist) 
        
            # Cartesian coordinate of point in the middle
            x, y, z = _utils.spherical2cartesian(this_Y[1], this_X[1], R)
        else:
            # CARTESIAN: take the average
            x = (lon0 + lon1)/2
            y = (lat0 + lat1)/2
            z = 0
            
        # Indexes of 3 nearest grid point
        _, indexes = tree.query(_np.column_stack((x, y, z)), k=3)
        indexes    = _np.unravel_index(indexes, shape)
        new_iY = _np.ndarray.tolist(indexes[Yindex])[0]
        new_iX = _np.ndarray.tolist(indexes[Xindex])[0]

        # Extract just one point
        to_rem = []
        for i, (this_iY, this_iX) in enumerate(zip(new_iY, new_iX)):
            if (this_iY==iY[k] and this_iX==iX[k]) or (this_iY==iY[k+1] and this_iX==iX[k+1]): to_rem = to_rem+[i]
        new_iY = _np.asarray([i for j, i in enumerate(new_iY) if j not in to_rem])[0]
        new_iX = _np.asarray([i for j, i in enumerate(new_iX) if j not in to_rem])[0]
        
        # Extract new lat and lon
        new_lat = YC.isel(Y=new_iY, X=new_iX).values
        new_lon = XC.isel(Y=new_iY, X=new_iX).values
        
        # Insert
        near_Y = _np.insert(near_Y,k+1,new_lat)
        near_X = _np.insert(near_X,k+1,new_lon)
        iY = _np.insert(iY,k+1,new_iY)
        iX = _np.insert(iX,k+1,new_iX)

        # Steps
        diff_iY = _np.fabs(_np.diff(iY))
        diff_iX = _np.fabs(_np.diff(iX))
    
    # New dimensions
    mooring = _xr.DataArray(_np.arange(len(iX)), 
                            dims=('mooring'),
                            attrs={'long_name': 'index of mooring',
                                   'units': 'none'})
    y   = _xr.DataArray(_np.arange(1), 
                        dims=('y'),
                        attrs={'long_name': 'j-index of cell center',
                               'units': 'none'})
    x   = _xr.DataArray(_np.arange(1), 
                        dims=('x'),
                        attrs={'long_name': 'i-index of cell corner',
                               'units': 'none'})
    yp1 = _xr.DataArray(_np.arange(2), 
                        dims=('yp1'),
                        attrs={'long_name': 'j-index of cell center',
                               'units': 'none'})
    xp1 = _xr.DataArray(_np.arange(2), 
                        dims=('xp1'),
                        attrs={'long_name': 'i-index of cell corner',
                               'units': 'none'})
    
    # Transform indexes in DataArray
    iy = _xr.DataArray(_np.reshape(iY, (len(mooring), len(y))), 
                       coords={'mooring': mooring, 'y': y}, 
                       dims=('mooring', 'y'))
    ix = _xr.DataArray(_np.reshape(iX, (len(mooring), len(x))), 
                       coords={'mooring': mooring, 'x': x},
                       dims=('mooring', 'x'))
    iyp1 = _xr.DataArray( _np.stack((iY, iY+1), 1), 
                         coords={'mooring': mooring, 'yp1': yp1},
                         dims=('mooring', 'yp1'))
    ixp1 = _xr.DataArray( _np.stack((iX, iX+1), 1), 
                         coords={'mooring': mooring, 'xp1': xp1},
                         dims=('mooring', 'xp1'))
    
    # Initialize new dataset
    new_ds = _xr.Dataset({'mooring': mooring, 
                          'Y': y.rename(y = 'Y'), 'Yp1': yp1.rename(yp1 = 'Yp1'),
                          'X': x.rename(x = 'X'), 'Xp1': xp1.rename(xp1 = 'Xp1')},
                          attrs = ds.attrs)
    
    # Loop and take out (looping is faster than apply to the whole dataset)
    for var in ds.variables:
        if var in ['X', 'Y', 'Xp1', 'Yp1']:
            new_ds[var].attrs.update({attr: ds[var].attrs[attr] for attr in ds[var].attrs if attr not in ['units', 'long_name']})
            continue
        elif not any(dim in ds[var].dims for dim in ['X', 'Y', 'Xp1', 'Yp1']):   
            da = ds[var]
        else:
            for this_dims in [['Y', 'X'], ['Yp1', 'Xp1'], ['Y', 'Xp1'], ['Yp1', 'X']]:
                if set(this_dims).issubset(ds[var].dims):
                    eval('iy', {}, {'iy': iy, 'ix': ix, 'iyp1': iyp1, 'ixp1': ixp1})
                    da = ds[var].isel({dim : eval('i'+dim.lower(),
                                                  {},
                                                  {'iy': iy, 'ix': ix, 'iyp1': iyp1, 'ixp1': ixp1}) for dim in this_dims})
                    da = da.drop(this_dims).rename({dim.lower():dim for dim in this_dims})

        # Merge
        new_ds = _xr.merge([new_ds, da.reset_coords()])
        
    # Merge removes the attributes: put them back!    
    new_ds.attrs = ds.attrs

    # Add distance
    dists = _np.zeros(near_Y.shape)
    for i in range(1,len(dists)):
        coord1   = (near_Y[i-1], near_X[i-1])
        coord2   = (near_Y[i],   near_X[i])
        
        if R is not None:
            # SPHERICAL
            dists[i] = _great_circle(coord1, coord2, radius = R).km
        else:
            # CARTESIAN
            dists[i] = _np.sqrt((coord2[0]-coord1[0])**2+(coord2[1]-coord1[1])**2)
    
    dists = _np.cumsum(dists)
    distance = _xr.DataArray(dists, 
                             coords={'mooring': mooring},
                             dims=('mooring'),
                             attrs={'long_name': 'Distance from first mooring'})
    
    if R is not None:
        # SPHERICAL
        distance.attrs['units'] = 'km'
    else:
        # CARTESIAN
        if 'units' in XC.attrs:
            distance.attrs['units'] = XC.attrs['units']
    new_ds['mooring_dist'] = distance
    
    # Reset coordinates
    new_ds = new_ds.set_coords([coord for coord in ds.coords]+['mooring_dist'])
    
    # Recreate od
    od._ds = new_ds
    od = od.set_grid_coords({'mooring': {'mooring': -0.5}}, add_midp=True, overwrite=False)
    
    # Create dist_midp
    dist_midp = _xr.DataArray(od._grid.interp(od._ds['mooring_dist'], 'mooring'),
                              attrs=od._ds['mooring_dist'].attrs)
    od = od.merge_into_oceandataset(dist_midp.rename('mooring_midp_dist'))
    od._ds = od._ds.set_coords([coord for coord in od._ds.coords]+['mooring_midp_dist'])

    return od

def survey_stations(od, Ysurv, Xsurv, delta=None,
                    xesmf_regridder_kwargs = {'method': 'bilinear'}, **kwargs):
    
    """
    Extract survey stations with regular spacing.
    Trajectories are great circle paths when coordinates are spherical.
    
    Parameters
    ----------
    od: OceanDataset
        od that will be subsampled
    Ysurv: 1D array_like
        Y coordinates of stations. 
    Xsurv: 1D array_like,
        X coordinates of stations.
    delta: scalar, None
        Distance between stations.
        Units are km for spherical coordinate, same units of coordinates for cartesian.
        If None, only (Ysurv, Xsurv) stations are used.
    xesmf_regridder_kwargs: dict
        Keyword arguments for xesmf.regridder, such as `method`.
        Defaul method: `bilinear`.  
        Available methods: {‘bilinear’, ‘conservative’, ‘patch’, ‘nearest_s2d’, ‘nearest_d2s’}
    **kwargs: 
        Keyword arguments for subsample.cutout

    Returns
    -------
    od: OceanDataset
        Subsampled oceandataset
       
    See Also
    --------
    subsample.cutout
    
    References
    ----------
    https://xesmf.readthedocs.io/en/stable/user_api.html#regridder
    
    Notes
    -----
    By default, kwargs['add_Hbdr'] = True. Try to play with add_Hbdr values if zeros/nans are returned.
    ospy.survey_stations interpolates using xesmf.regridder. THIS FUNCTION DOES NOT SUPPORT LAZY COMPUTATION!
    
    xesmf.regridder currently dosen't allow to set the coordinates system (default is spherical).
    Surveys using cartesian coordinates can be made by changing the xesmf source code as explained here: https://github.com/JiaweiZhuang/xESMF/issues/39
    """
    
    # Check
    for wrong_dim in ['mooring', 'station', 'particle']:
        if wrong_dim in od._ds.dims:
            raise ValueError('`survey_stations` cannot subsample oceandatasets with dimension [{}]'.format(wrong_dim))
            
    # Convert variables to numpy arrays and make some check
    Ysurv = _np.asarray(Ysurv, dtype=od._ds['YC'].dtype)
    if   Ysurv.ndim == 0: Ysurv = Ysurv.reshape(1)
    elif Ysurv.ndim>1: raise TypeError('Invalid `Y`')

    Xsurv = _np.asarray(Xsurv, dtype=od._ds['XC'].dtype)
    if   Xsurv.ndim == 0: Xsurv = Xsurv.reshape(1)
    elif Xsurv.ndim >1: raise TypeError('Invalid `X`')
        
    if not isinstance(xesmf_regridder_kwargs, dict):
        raise TypeError('`xesmf_regridder_kwargs` must be dict')
        
    # Earth Radius
    R = od.parameters['rSphere']
    if R is None:
        _warnings.warn("\nospy.survey_stations interpolates using xesmf.regridder."
                       "\nxesmf.regridder currently dosen't allow to set the coordinates system (default is spherical)."
                       "\nSurveys using cartesian coordinates can be made by changing the xesmf source code as explained here:"
                       "https://github.com/JiaweiZhuang/xESMF/issues/39", stacklevel=2)
    
    # Compute trajectory
    for i, (lat0, lon0, lat1, lon1) in enumerate(zip(Ysurv[:-1], Xsurv[:-1], Ysurv[1:], Xsurv[1:])):
        if R is not None:
            # SPHERICAL: follow great circle path
            this_Y, this_X, this_dists = _utils.great_circle_path(lat0,lon0,lat1,lon1,delta, R=R)
        else:
            # CARTESIAN: just a simple interpolation
            this_Y, this_X, this_dists = _utils.cartesian_path(lat0,lon0,lat1,lon1,delta)
        if i==0: 
            Y_surv  = this_Y
            X_surv  = this_X
            dists_surv = this_dists
        else:
            this_Y  = _np.delete(this_Y,  0, axis=None)
            this_X  = _np.delete(this_X,  0, axis=None) 
            this_dists = _np.delete(this_dists, 0, axis=None) 
            if len(this_dists)==0: continue
            this_dists = this_dists + dists_surv[-1]
            Y_surv  = _np.concatenate((Y_surv,  this_Y))
            X_surv  = _np.concatenate((X_surv,  this_X))
            dists_surv = _np.concatenate((dists_surv, this_dists)) 

    # Cutout
    if "YRange"  not in kwargs: kwargs['YRange']    = Y_surv
    if "XRange"  not in kwargs: kwargs['XRange']    = X_surv
    if "add_Hbdr" not in kwargs: kwargs['add_Hbdr'] = True
    od = od.subsample.cutout(**kwargs)
    
    # Message
    print('Carrying out survey.') 
    
    # Unpack ds and grid
    ds   = od._ds
    grid = od._grid
    
    # TODO: This is probably slowing everything down, and perhaps adding some extra error.
    #       I think we should have separate xesmf interpolations for different grids, then merge.
    # Move all variables on same spatial grid
    for var in [var for var in ds.variables if var not in ds.dims]:
        for dim in ['Xp1', 'Yp1']:
            if dim in ds[var].dims and var!=dim:
                attrs   = ds[var].attrs
                ds[var] = grid.interp(ds[var], axis=dim[0], to='center', boundary='fill', fill_value=_np.nan)
                ds[var].attrs = attrs
                
    # Create xesmf datsets    
    ds_in  = ds
    coords_in = ds_in.coords
    ds_in = ds_in.reset_coords()
    ds_in['lat'] = ds_in['YC']
    ds_in['lon'] = ds_in['XC']
    ds = _xr.Dataset({'lat':  (['lat'], Y_surv),
                      'lon':  (['lon'], X_surv)},
                     attrs = ds.attrs)

    # Interpolate
    regridder = _xe.Regridder(ds_in, ds, **xesmf_regridder_kwargs) 
    interp_vars = [var for var in ds_in.variables if var not in ['lon', 'lat', 'X', 'Y']]
    print('Variables to interpolate: {}.'.format([var for var in interp_vars if set(['X', 'Y']).issubset(ds_in[var].dims)]))
    for var in interp_vars:
        if set(['X', 'Y']).issubset(ds_in[var].dims):
            print('Interpolating [{}].'.format(var))
            attrs   = ds_in[var].attrs
            ds[var] = regridder(ds_in[var])
            ds[var].attrs = attrs
        elif var not in ['Xp1', 'Yp1']: 
            ds[var] = ds_in[var].reset_coords(drop=True)
    regridder.clean_weight_file()    
    
    # Extract transect 
    ds = ds.isel(lat=_xr.DataArray(_np.arange(len(Y_surv)), dims='station'),
                 lon=_xr.DataArray(_np.arange(len(X_surv)), dims='station'))
    
    # Add station dimension
    ds['station'] = _xr.DataArray(_np.arange(len(X_surv)), 
                                  dims=('station'),
                                  attrs={'long_name': 'index of survey station',
                                         'units': 'none'})    
    
    # Add distance
    ds['station_dist'] = _xr.DataArray(dists_surv,
                                       dims=('station'),
                                       attrs={'long_name': 'Distance from first station'})
    if R is not None:
        # SPHERICAL
        ds['station_dist'].attrs['units'] = 'km'
    else:
        # CARTESIAN
        if 'units' in ds['lat'].attrs:
            ds['station_dist'].attrs['units'] = ds['lat'].attrs['units']
    ds = ds.set_coords('station_dist')
    
    # Return od
    od._ds = ds
    od = od.set_grid_coords({'Z': od.grid_coords.pop('Z', None), 'time': od.grid_coords.pop('time', None), 'station': {'station': -0.5}}, add_midp=True, overwrite=True)
    
    # Create dist_midp
    dist_midp = _xr.DataArray(od._grid.interp(od._ds['station_dist'], 'station'),
                              attrs=od._ds['station_dist'].attrs)
    od = od.merge_into_oceandataset(dist_midp.rename('station_midp_dist'))
    od._ds = od._ds.set_coords([coord for coord in od._ds.coords]+['station_midp_dist'])
    
    return od





def particle_properties(od, times, Ypart, Xpart, Zpart, **kwargs):
    
    """
    Extract Eulerian properties of particles using nearest-neighbor interpolation.
    
    Parameters
    ----------
    od: OceanDataset
        od that will be subsampled.
    times: 1D array_like or scalar
        time of particles.
    Ypart: 2D array_like or 1D array_like if times is scalar
        Y coordinates of particles. Dimensions order: (time, particle)
    Xpart: 2D array_like or 1D array_like if times is scalar
        X coordinates of particles. Dimensions order: (time, particle)
    Zpart: 2D array_like or 1D array_like if times is scalar
        Z of particles. Dimensions order: (time, particle)
    **kwargs: 
        Keyword arguments for subsample.cutout

    Returns
    -------
    od: OceanDataset
        Subsampled oceandataset
        
    See Also
    --------
    subsample.cutout
    OceanDataset.create_tree
    """
    # TODO: add different interpolations. Renske already has some code.
    
    # Check
    for wrong_dim in ['mooring', 'station', 'particle']:
        if wrong_dim in od._ds.dims:
            raise ValueError('`particle_properties` cannot subsample oceandatasets with dimension [{}]'.format(wrong_dim))
            
    # Convert variables to numpy arrays and make some check
    times  = _np.asarray(times, dtype = od._ds['time'].dtype)
    if times.ndim == 0: times = times.reshape(1)
    elif times.ndim >1: raise TypeError('Invalid `times`')
    
    Ypart  = _np.asarray(Ypart)
    if Ypart.ndim <2 and Ypart.size==1: Ypart = Ypart.reshape((1, Ypart.size))
    elif Ypart.ndim >2: raise TypeError('Invalid `Ypart`')
        
    Xpart  = _np.asarray(Xpart)
    if Xpart.ndim <2 and Xpart.size==1: Xpart = Xpart.reshape((1, Xpart.size))
    elif Xpart.ndim >2: raise TypeError('Invalid `Xpart`')
      
    Zpart  = _np.asarray(Zpart)
    if Zpart.ndim <2 and Zpart.size==1: Zpart = Zpart.reshape((1, Zpart.size))
    elif Zpart.ndim >2: raise TypeError('Invalid `Zpart`')

    if (not Ypart.ndim==Xpart.ndim==Zpart.ndim or not times.size==Ypart.shape[0]):
        TypeError('`times`, `Xpart`, `Ypart`, and `Zpart` have inconsistent shape')
    
    # Cutout
    if "timeRange" not in kwargs: kwargs['timeRange'] = times
    if "YRange"    not in kwargs: kwargs['YRange']    = [_np.min(Ypart), _np.max(Ypart)]
    if "XRange"    not in kwargs: kwargs['XRange']    = [_np.min(Xpart), _np.max(Xpart)]
    if "ZRange"    not in kwargs: kwargs['ZRange']    = [_np.min(Zpart), _np.max(Zpart)]
    if "add_Hbdr"  not in kwargs: kwargs['add_Hbdr']  = True
    if "add_Vbdr"  not in kwargs: kwargs['add_Vbdr']  = True    
    od = od.subsample.cutout(**kwargs)
    
    # Message
    print('Extracting Eulerian properties of particles.')
    
    # Unpack ds and info
    ds = od._ds
    R  = od.parameters['rSphere']

    # Remove time_midp and warn
    vars2drop = [var for var in ds.variables if 'time_midp' in ds[var].dims]
    if vars2drop:
        _warnings.warn("\nParticle properties extraction drops variables on `time_midp` dimension. \nDropped variables: {}.".format(vars2drop), stacklevel=2)
        ds = ds.drop(vars2drop)
    if 'time_midp' in ds.dims: ds = ds.drop('time_midp')
                
    # New dimensions
    time = _xr.DataArray(times,
                         dims = ('time'),
                         attrs = ds['time'].attrs)
    particle = _xr.DataArray(_np.arange(Ypart.shape[1]), 
                             dims = ('particle'),
                             attrs={'long_name': 'index of particle',
                                    'units': 'none'})
    i_ds = _xr.Dataset({'time': time, 'particle': particle})
    
    # Find vertical and time indexes
    for dim in ds.dims:
        if dim=='time':
            tmp = _xr.DataArray(times,
                                dims = ('time'))
        elif dim[0]=='Z':
            tmp = _xr.DataArray(Zpart,
                                dims = ('time', 'particle'))
        else: continue
        itmp = _xr.DataArray(_np.arange(len(ds[dim])),
                             coords = {dim: ds[dim].values},
                             dims   = {dim})
        itmp = itmp.sel({dim: tmp}, method='nearest')
        i_ds['i'+dim] = itmp
    
    # Convert 2 cartesian
    if R is not None:
        x, y, z = _utils.spherical2cartesian(Y = Ypart, X = Xpart, R = R)
    else:
        x = Xpart; y = Ypart; z = _np.zeros(Y.shape)
    
    # Find horizontal indexes
    for grid_pos in ['C', 'U', 'V', 'G']:
        
        # Don't create tree if no variables
        var_grid_pos = [var for var     in ds.variables 
                            if  var not in ds.coords and set(['X'+grid_pos, 'Y'+grid_pos]).issubset(ds[var].coords)]
        if not var_grid_pos: continue
        
        # Useful variables
        Y = od._ds['Y' + grid_pos]
        X = od._ds['X' + grid_pos]
        shape  = X.shape
        Yname  = [dim for dim in Y.dims if dim[0]=='Y'][0]
        Xname  = [dim for dim in X.dims if dim[0]=='X'][0]
        Yindex = X.dims.index(Yname)
        Xindex = X.dims.index(Xname)
        
        # Create tree
        tree = od.create_tree(grid_pos = grid_pos)

        # Indexes of nearest grid points
        _, indexes = tree.query(_np.column_stack((x.flatten(), y.flatten(), z.flatten())))
        indexes    = _np.unravel_index(indexes, shape)
        iY = _xr.DataArray(_np.reshape(indexes[Yindex], y.shape),
                           dims = ('time', 'particle'))
        iX = _xr.DataArray(_np.reshape(indexes[Xindex], x.shape),
                           dims = ('time', 'particle'))        
                           
        # Transform indexes in DataArray and add to dictionary
        i_ds['i'+Yname] = iY
        i_ds['i'+Xname] = iX
        
        # Subsample (looping is faster)
        for var in var_grid_pos:
            i_ds[var] = ds[var].isel({dim: i_ds['i'+dim] for dim in ds[var].dims})
    
    # Remove indexes
    i_ds = i_ds.drop([var for var in i_ds.variables if var not in ds.variables and var not in i_ds.dims])
    
    # Recreate od
    od._ds = i_ds
    
    # Add time midp
    od = od.set_grid_coords({'time' : {'time': -0.5}}, add_midp=True, overwrite=True)
    
    return od


class _subsampleMethdos(object):
    """
    Enables use of oceanspy.subsample functions as attributes on a OceanDataset.
    For example, OceanDataset.subsample.cutout
    """
    
    def __init__(self, od):
        self._od = od

    @_functools.wraps(cutout)
    def cutout(self, **kwargs):
        return cutout(self._od, **kwargs)
    
    @_functools.wraps(mooring_array)
    def mooring_array(self, **kwargs):
        return mooring_array(self._od, **kwargs)
    
    @_functools.wraps(survey_stations)
    def survey_stations(self, **kwargs):
        return survey_stations(self._od, **kwargs)
    
    @_functools.wraps(particle_properties)
    def particle_properties(self, **kwargs):
        return particle_properties(self._od, **kwargs)

