"""
Visualize: plot dataset
"""

import xarray as _xr
import numpy as _np
import xgcm as _xgcm

def interactive(dsORda,
                info,
                hvplot_kwargs = {}):
    """
    GUI for plotting data interactively using hvPlot.
    Plot maps using Orthographic projection, 
    and recognize vertical sections/profiles

    Parameters
    ----------
    dsORda: xarray.Dataset or xarray.DataArray
    info: oceanspy.open_dataset._info
    hvplot_kwargs: dict
        Keyword arguments for hvPlot: https://hvplot.pyviz.org/user_guide/index.html

    Returns
    -------
    GUI
    """

    import cartopy.crs as _ccrs
    import copy as _copy

    # Import modules
    from ipywidgets import interact
    import hvplot.xarray
    
    # Define global variables to pass to plot_da
    global ds, hv_kwargs 
    if isinstance(dsORda,_xr.DataArray): ds = dsORda.to_dataset()
    elif isinstance(dsORda,_xr.Dataset): ds = dsORda
    hv_kwargs = hvplot_kwargs
    
    # Choose variable interactively
    def plot_da(varname):
        # Make a copy of kwargs
        this_hv_kwargs = _copy.deepcopy(hv_kwargs)
        
        # Extract datarray and print info
        da   = ds[varname]
        for attr in da.attrs: print(attr,':',da.attrs[attr])
        print()
        for dim in da.dims:
            print('Dimension :',dim)
            for attr in da[dim].attrs: print('     ',attr,':',da[dim].attrs[attr])
        
        # Find axis
        axis = {}
        for dim in da.dims:
            if 'axis' in da[dim].attrs:
                axis[da[dim].attrs['axis']] = dim

        # MAP
        if 'X' in axis and 'Y' in axis:
            # Default 
            if 'x' not in this_hv_kwargs: this_hv_kwargs['x'] = axis['X']
            if 'y' not in this_hv_kwargs: this_hv_kwargs['y'] = axis['Y']
            if 'crs' not in this_hv_kwargs:
                crs = _ccrs.PlateCarree()
                this_hv_kwargs['crs'] = crs
            if 'projection' not in this_hv_kwargs: 
                lon = ds[axis['X']].values
                lat = ds[axis['Y']].values
                proj = _ccrs.Orthographic(central_longitude = lon.mean(),
                                          central_latitude  = lat.mean())
                this_hv_kwargs['projection'] = proj
            if 'groupby' not in this_hv_kwargs: 
                this_hv_kwargs['groupby'] = [dim for dim in da.dims if dim not in [axis['X'], axis['Y']]]
            if 'global_extent' not in this_hv_kwargs: this_hv_kwargs['global_extent'] = False
            if 'rasterize' not in this_hv_kwargs: this_hv_kwargs['rasterize'] = True
            if 'cmap' not in this_hv_kwargs: this_hv_kwargs['cmap'] = 'viridis'
            if 'kind' not in this_hv_kwargs: this_hv_kwargs['kind'] = 'quadmesh'
     
        # VERTICAL PROFILE   
        elif 'Z' in axis and len(da.dims)==1:
            # Default
            if 'invert' not in this_hv_kwargs: this_hv_kwargs['invert'] = True
            if 'kind'   not in this_hv_kwargs: this_hv_kwargs['kind'] = 'line'  
        
        # VERTICAL SECTION
        elif ('dist_VS' in info.var_names) and (info.var_names['dist_VS'] in da.dims):
            # 1D
            if len(da.dims)==1:
                # Default
                if 'kind'   not in this_hv_kwargs: this_hv_kwargs['kind'] = 'line'
                if 'groupby' not in this_hv_kwargs: 
                    this_hv_kwargs['groupby'] = []
                    
            # 2D
            if 'Z' in axis and info.var_names['dist_VS'] in da.dims:
                # Default
                if 'x' not in this_hv_kwargs: this_hv_kwargs['x'] = info.var_names['dist_VS']
                if 'y' not in this_hv_kwargs: this_hv_kwargs['y'] = axis['Z']
                if 'groupby' not in this_hv_kwargs: 
                    this_hv_kwargs['groupby'] = [dim for dim in da.dims if dim not in [axis['Z'], info.var_names['dist_VS']]]
                if 'cmap' not in this_hv_kwargs: this_hv_kwargs['cmap'] = 'viridis'
                if 'kind' not in this_hv_kwargs: this_hv_kwargs['kind'] = 'quadmesh'
            
        return da.hvplot(**this_hv_kwargs)
        
    return interact(plot_da,varname=[var for var in ds.variables if 
                                     (var not in ds.dims and var not in ds.coords)])
