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
               cutout_kwargs  = None,
               **kwargs):
    
    """
    Plot temperature-salinity diagram.
    
    Parameters
    ----------
    od: OceanDataset
        oceandataset to check for missing variables
    Tlim: array_like with 2 elements
        Temperature limits on the y axis.
        If None, uses the min and max value.
    Slim: array_like with 2 elements
        Salinity limits on the x axis.
        If None, uses the min and max value.
    colorName: str, None
        string of the variable to use to color (e.g., Temp).
        If None, uses plot insted of scatter (much faster)
    ax: matplotlib.pyplot.axes
        If None, uses the current axis.
    cmap_kwargs: dict
        Keyword arguments for the colormap (same used by xarray)
    contour_kwargs: dict
        Keyword arguments for matplotlib.pytplot.contour (isopycnals)
    clabel_kwargs: dict
        Keyword arguments for matplotlib.pytplot.clabel (isopycnals)  
    cutout_kwargs: dict
        Keyword arguments for subsample.cutout
    **kwargs:
        If colorName is None: Kewyword arguments for matplotlib.pytplot.plot()
        Otherwise, kewyword arguments for matplotlib.pytplot.scatter()
        
    Returns
    -------
    ax: matplotlib.pyplot.Axes
    
    References
    ----------
    http://xarray.pydata.org/en/stable/plotting.html#introduction
    """
    
    # TODO: implement Faceting?
    
    import matplotlib.pyplot as _plt
    
    # Check parameters
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    if Tlim is not None:
        Tlim  = _np.asarray(Tlim)
        if Tlim.size!=2:  raise TypeError('`Tlim` must contain 2 elements')
        Tlim = Tlim.reshape(2)
            
    if Slim is not None:
        Slim  = _np.asarray(Slim)
        if Slim.size!=2:  raise TypeError('`Slim` must contain 2 elements')
        Slim = Slim.reshape(2)
       
    if not isinstance(colorName, (type(None), str)):
        raise TypeError('`colorName` must be str')
    
    if not isinstance(ax, (type(None), _plt.Axes)):
        raise TypeError('`ax` must be matplotlib.pyplot.Axes')
            
    if not isinstance(cmap_kwargs, (type(None), dict)):
        raise TypeError('`cmap_kwargs` must be None or dict')
       
    if not isinstance(contour_kwargs, (type(None), dict)):
        raise TypeError('`contour_kwargs` must be None or dict')
        
    if not isinstance(clabel_kwargs, (type(None), dict)):
        raise TypeError('`clabel_kwargs` must be None or dict')
        
    if not isinstance(cutout_kwargs, (type(None), dict)):
        raise TypeError('`cutout_kwargs` must be None or dict')
        
    # Handle kwargs
    if cmap_kwargs is None:    cmap_kwargs = {}
    if contour_kwargs is None: contour_kwargs = {}
    if clabel_kwargs is None:  clabel_kwargs = {}
    if cutout_kwargs is None:  cutout_kwargs = {}
        
    # Cutout first
    od = od.cutout(**cutout_kwargs)
    
    # Check and extract T and S
    varList = ['Temp', 'S']
    od = _compute._add_missing_variables(od, varList)
    T = od._ds['Temp']
    S = od._ds['S']
    
    # Extract color field, and interpolate if needed
    if colorName is not None:
        
        # Add missing variables (use private)
        _colorName =  _compute._rename_aliased(od, colorName)
        od = _compute._add_missing_variables(od, _colorName)

        # Extract color (use public)
        color = od.dataset[colorName]
        grid  = od.grid
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
    
    # Set title
    title = ''
    for dim in T.dims:
        dim2rem = [d for d in T.dims if len(T[d])==1 and d!=dim]
        tit0 = T.squeeze(dim2rem).drop(dim2rem).isel({dim: 0})._title_for_slice()
        tit1 = T.squeeze(dim2rem).drop(dim2rem).isel({dim: -1})._title_for_slice()
        
        
        if tit0==tit1:
            tit = tit0
        else:
            tit0 = tit0.replace(dim+' = ', dim+': from ')
            tit1 = tit1.replace(dim+' = ', ' to ')
            tit = tit0 + tit1
        if title=='':
            title = title + tit
        else:
            title = title + '\n' + tit
    ax.set_title(title)
    
    return ax