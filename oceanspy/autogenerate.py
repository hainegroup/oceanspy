"""
Generate a ``Dataset`` from model output stored on SciServer.
"""

# Start aliases with _: I don't wanna see them using TAB.
import xarray as _xr
import time as _time

def exp_ASR(cropped = False):
    """
    Same configuration as Almansi et al., 2017 [1]_.
    However, the atmospheric forcing is the Arctic System Reanalysis (ASR).
     
    Parameters
    ----------
    cropped: bool
        If True include diagnostics to close the heat/salt budget. 
        Since these variable have been cropped, return a Dataset on a smaller domain: 
        [ 72N , 69N] [-22E , -13E]

    Returns
    -------
    ds: xarray.Dataset
        Dataset with all available diagnostics
    
    REFERENCES
    ----------
    .. [1] Almansi et al., 2017 http://doi.org/10.1175/JPO-D-17-0129.1
    """
    
    # Check parameters
    if not isinstance(cropped, bool) : raise RuntimeError("'cropped' must be a boolean")
          
    # Hello
    start_time = _time.time()
    print('Opening dataset',end=': ')
    
    # Import grid and fields separately, then merge
    gridpath = '/home/idies/workspace/OceanCirculation/exp_ASR/grid_glued.nc'
    fldspath = '/home/idies/workspace/OceanCirculation/exp_ASR/result_*/output_glued/*.*_glued.nc'
    croppath = '/home/idies/workspace/OceanCirculation/exp_ASR/result_*/output_glued/cropped/*.*_glued.nc'
    gridset = _xr.open_dataset(gridpath,
                              drop_variables = ['XU','YU','XV','YV','RC','RF','RU','RL'])
    fldsset = _xr.open_mfdataset(fldspath,
                                concat_dim     = 'T',
                                drop_variables = ['diag_levels','iter'])
    ds = _xr.merge([gridset, fldsset])
    
    # Create horizontal vectors (remove zeros due to exch2)
    ds['X'].values   = ds.XC.where((ds.XC!=0) & (ds.YC!=0)).mean(dim='Y',   skipna=True)
    ds['Xp1'].values = ds.XG.where((ds.XG!=0) & (ds.YG!=0)).mean(dim='Yp1', skipna=True)
    ds['Y'].values   = ds.YC.where((ds.XC!=0) & (ds.YC!=0)).mean(dim='X',   skipna=True)
    ds['Yp1'].values = ds.YG.where((ds.XG!=0) & (ds.YG!=0)).mean(dim='Xp1', skipna=True)
    ds = ds.drop(['XC','YC','XG','YG'])
    
    # Negative dr in order to be consistent with upward z axis
    ds['drC'].values = -ds['drC'].values
    ds['drF'].values = -ds['drF'].values
       
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
        
    # Adjust dimensions creating conflicts
    ds = ds.rename({'Z': 'Ztmp'})
    ds = ds.rename({'T': 'time', 'Ztmp': 'Z', 'Zmd000216': 'Z'})
    ds = ds.squeeze('Zd000001')
    
    for dim in ['Z','Zp1', 'Zu','Zl']:
        ds[dim].values   = ds[dim].values
        ds[dim].attrs.update({'positive': 'up'}) 
    
    # Add xgcm info
    for dim in ['Z', 'X', 'Y']: ds[dim].attrs.update({'axis': dim})  
    for dim in ['Zp1','Zu','Zl','Xp1','Yp1']: 
        if min(ds[dim].values)<min(ds[dim[0]].values):
            ds[dim].attrs.update({'axis': dim[0], 'c_grid_axis_shift': -0.5})
        elif min(ds[dim].values)>min(ds[dim[0]].values):
            ds[dim].attrs.update({'axis': dim[0], 'c_grid_axis_shift': +0.5})
    
    # ByeBye
    elapsed_time = _time.time() - start_time
    print(_time.strftime('done in %H:%M:%S', _time.gmtime(elapsed_time)))
        
    return ds

