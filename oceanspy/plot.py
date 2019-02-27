"""
Plot using OceanDataset objects.
"""

import xarray   as _xr
import oceanspy as _ospy
import numpy    as _np
import warnings as _warnings
import copy     as _copy
from . import compute as _compute

def TS_diagram(od, 
               Tlim           = None,
               Slim           = None,
               colorName      = None,
               ax             = None,
               cmap_kwargs    = None,
               contour_kwargs = None,
               clabel_kwargs  = None,
               **kwargs):
    
    # TODO: check anddoc
    
    # Handle kwargs
    if cmap_kwargs is None:
        cmap_kwargs = {}
    if contour_kwargs is None:
        contour_kwargs = {}
    if clabel_kwargs is None:
        clabel_kwargs = {}
    # Check and extract T and S
    varList = ['Temp', 'S']
    od = _compute._add_missing_variables(od, varList)
    T = od._ds['Temp']
    S = od._ds['S']
    
    # Extract color field, and interpolate if needed
    if colorName is not None:
        
        # Use dataset if available, otherwise try to compute from _ds
        if colorName in od.dataset.variables:
            color = od.dataset[colorName]
            grid  = od.grid
            # If dimension have aliases, take care of the aliases
            # TODO: need to double check this!
            if od.aliases is not None:
                Tdims = [od.aliases[dim] if dim in od.aliases else dim for dim in T.dims]
            else:
                Tdims = T.dims
            dims2interp = [dim for dim in color.dims if dim not in Tdims]
        else:
            # TODO: add a warning?
            od = _compute._add_missing_variables(od, [colorName])
            grid = od._grid
            dims2interp = [dim for dim in color.dims if dim not in T.dims]
        
        # Interpolation 
        for dim in dims2interp:
            for axis in od.grid.axes.keys():
                if dim in [od.grid.axes[axis].coords[k].name for k in od.grid.axes[axis].coords.keys()]: 
                    print('Interpolating [{}] along [{}]-axis'.format(colorName, axis))
                    attrs = color.attrs
                    color = grid.interp(color, axis, to='center', boundary='fill', fill_value=_np.nan)
                    color.attrs = attrs
                        
        # Broadcast, in case color has different dimensions
        T, S, color = _xr.broadcast(T, S, color)
            
    # Compute density
    if Tlim is None:
        Tlim = [_np.floor(T.min().values), _np.ceil(T.max().values)]
    if Slim is None:
        Slim = [_np.floor(S.min().values), _np.ceil(S.max().values)]
    t, s = _xr.broadcast(_xr.DataArray(_np.linspace(Tlim[0], Tlim[-1], 100), dims= ('t')),
                         _xr.DataArray(_np.linspace(Slim[0], Slim[-1], 100), dims= ('s')))
    odSigma0 = _ospy.OceanDataset(_xr.Dataset({'Temp': t, 'S': s})).set_parameters(od.parameters)
    odSigma0 = odSigma0.merge_potential_density_anomaly()
    
    # Extract density variables
    t = odSigma0._ds['Temp']
    s = odSigma0._ds['S']
    d = odSigma0._ds['Sigma0']   
    
    # Create axis
    import matplotlib.pyplot as _plt
    if ax is None: ax = _plt.gca()
        
    # Use plot if colorless (faster!), otherwise use scatter
    if colorName is None:
        default_kwargs = {'color': 'k', 'linestyle': 'None', 'marker': '.'}
        kwargs = {**default_kwargs, **kwargs}
        ax.plot(S.values.flatten(), T.values.flatten(), **kwargs)
    else:
        # Mask points out of axes
        color = color.where(_np.logical_and(T>min(Tlim), T<max(Tlim)))
        color = color.where(_np.logical_and(S>min(Slim), T<max(Slim)))
        color = color.stack(all_dims=color.dims)
        c = color.values
        # Create colorbar (stolen from xarray)
        cmap_kwargs['plot_data'] = c
        cmap_params = _xr.plot.utils._determine_cmap_params(**cmap_kwargs)
        extend = cmap_params.pop('extend')
        _ = cmap_params.pop('levels')
        kwargs = {**cmap_params, **kwargs}
        # Scatter
        sc   = ax.scatter(S.values.flatten(), T.values.flatten(), c=c, **kwargs)
        cbar = _plt.colorbar(sc, label=_xr.plot.utils.label_from_attrs(color), extend=extend)
    
    # Plot isopycnals
    default_contour_kwargs = {'colors': 'gray'}
    contour_kwargs = {**default_contour_kwargs, **contour_kwargs}
    CS = ax.contour(s.values, t.values, d.values, **contour_kwargs)
    ax.clabel(CS, **clabel_kwargs)
    
    # Set labels and limits
    ax.set_xlabel(_xr.plot.utils.label_from_attrs(S))
    ax.set_ylabel(_xr.plot.utils.label_from_attrs(T))
    ax.set_xlim(Slim)
    ax.set_ylim(Tlim)
    ax.set_title('TS diagram')
    
    return ax