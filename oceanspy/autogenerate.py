import numpy as np
import pandas as pd
import xarray as xr
import xgcm as xgcm
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def generate_ds_grid(cropped   = False,
                     dispTable = False,
                     plotMap   = False):
    """
    Generate a Dataset and its Grid from NetCDFs.
    This function cointains hard-coded parameters specific for SciServer.
    
    Parameters
    ----------
    cropped: bool
            If True include variables to close the heat/salt budget. Budget's variables
            have been cropped and have a smaller domain (lonRange:[69, 72]; latRange:[-13, -22]). 
            Thus, also the regular variables will be cropped and only one Dataset is created.
    dispTable: bool
              If True prints a table with the available variables and their description
    plotMap: bool
            If True plot a map of the model domain and its resolution
               
    Returns
    -------
    ds: xarray.Dataset
       Dataset with all the available variables
    grid: xgcm.Grid
         Grid with all the staggered grids
    """
    
    # Check parameters
    if not isinstance(cropped, bool)  : raise RuntimeError("'cropped' needs to be a Boolean")
    if not isinstance(dispTable, bool): raise RuntimeError("'dispTable' needs to be a Boolean")
    if not isinstance(plotMap, bool)  : raise RuntimeError("'plotMap' needs to be a Boolean")
          
    # Hello
    start_time = time.time()
    print('Opening dataset:',end=' ')
    
    # Import grid and fields separately, then merge
    gridpath = '/home/idies/workspace/OceanCirculation/exp_ASR/grid_glued.nc'
    fldspath = '/home/idies/workspace/OceanCirculation/exp_ASR/result_*/output_glued/*.*_glued.nc'
    croppath = '/home/idies/workspace/OceanCirculation/exp_ASR/result_*/output_glued/cropped/*.*_glued.nc'
    gridset = xr.open_dataset(gridpath,
                              drop_variables = ['XU','YU','XV','YV','RC','RF','RU','RL'])
    fldsset = xr.open_mfdataset(fldspath,
                                concat_dim     = 'T',
                                drop_variables = ['diag_levels','iter'])
    ds = xr.merge([gridset, fldsset])
        
    # Create horizontal vectors (remove zeros due to exch2)
    ds['X'].values   = ds.XC.where((ds.XC!=0) & (ds.YC!=0)).mean(dim='Y',   skipna=True)
    ds['Xp1'].values = ds.XG.where((ds.XG!=0) & (ds.YG!=0)).mean(dim='Yp1', skipna=True)
    ds['Y'].values   = ds.YC.where((ds.XC!=0) & (ds.YC!=0)).mean(dim='X',   skipna=True)
    ds['Yp1'].values = ds.YG.where((ds.XG!=0) & (ds.YG!=0)).mean(dim='Xp1', skipna=True)
    ds = ds.drop(['XC','YC','XG','YG'])
       
    # Read cropped files and crop ds
    if cropped:
        cropset = xr.open_mfdataset(croppath,
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
        ds = xr.merge([ds, cropset])
        
    # Adjust dimensions creating conflicts
    ds = ds.rename({'Z': 'Ztmp'})
    ds = ds.rename({'T': 'time', 'Ztmp': 'Z', 'Zmd000216': 'Z'})
    ds = ds.squeeze('Zd000001')
    for dim in ['Z','Zp1', 'Zu','Zl']:
        ds[dim].values   = np.fabs(ds[dim].values)
        ds[dim].attrs.update({'positive': 'down'}) 
    
    # Create xgcm grid
    for dim in ['Z', 'X', 'Y']: ds[dim].attrs.update({'axis': dim})  
    for dim in ['Zp1','Zu','Zl','Xp1','Yp1']:
        if min(ds[dim].values)<min(ds[dim[0]].values):
            ds[dim].attrs.update({'axis': dim[0], 'c_grid_axis_shift': -0.5})
        elif min(ds[dim].values)>min(ds[dim[0]].values):
            ds[dim].attrs.update({'axis': dim[0], 'c_grid_axis_shift': +0.5})
    grid = xgcm.Grid(ds,periodic=False)
    
    # ByeBye
    elapsed_time = time.time() - start_time
    print(time.strftime('done in %H:%M:%S', time.gmtime(elapsed_time)))
    
    # Display available variables
    if dispTable:
        from IPython.core.display import HTML, display
        name        = ds.variables
        description = []
        units       = []
        for varName in name:
            this_desc  = ds[varName].attrs.get('long_name')
            this_units = ds[varName].attrs.get('units')
            if this_desc is None: 
                this_desc = ds[varName].attrs.get('description')
                if this_desc is None: this_desc = ' '
            if this_units is None: this_units = ' '
            description.append(this_desc)
            units.append(this_units)
        table = {'Name': name,'Description': description, 'Units':units}    
        table = pd.DataFrame(table)
        display(HTML(table[['Name','Description','Units']].to_html()))
        
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

