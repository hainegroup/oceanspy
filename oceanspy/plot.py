"""
Plot using OceanDataset objects.
"""
# TODO: add test that check squeezing!

import xarray    as _xr
import oceanspy  as _ospy
import numpy     as _np
import warnings  as _warnings
import copy      as _copy
import functools as _functools

from . import compute as _compute

try: 
    import matplotlib as _matplotlib
    _matplotlib.use('agg')
    import matplotlib.pyplot as _plt
except ImportError: pass
try:
    import cartopy.crs as _ccrs
except ImportError: pass

def TS_diagram(od, 
               Tlim           = None,
               Slim           = None,
               dens           = None,
               meanAxes       = None,
               colorName      = None,
               plotFreez      = True,
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
    meanAxes: 1D array_like, str, or None
        List of axes over which to apply weighted mean.
        If None, don't average. 
    dens: xarray.DataArray
        DataArray corresponding to density used for isopycnals.
        Must contain coordinates (Temp, S)
        In None, it will be inferred.
    colorName: str, None
        Name of the variable to use to color (e.g., Temp).
        If None, uses plot insted of scatter (much faster)
    plotFreez: bool
        If True, plot freezing line in blue.
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
    Axes object
    
    See also
    --------
    subsample.coutout
    animate.TS_diagram
    
    References
    ----------
    http://xarray.pydata.org/en/stable/plotting.html#introduction
    """
    
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
        
    if dens is not None:
        if not isinstance(dens, _xr.DataArray):
            raise TypeError('`dens` must be xarray.DataArray')
        elif not set(['Temp', 'S']).issubset(dens.coords):
            raise ValueError('`dens` must have coordinates (Temp, S)') 
            
    if not isinstance(colorName, (type(None), str)):
        raise TypeError('`colorName` must be str or None')
    
    if not isinstance(plotFreez, bool):
        raise TypeError('`plotFreez` must be bool')
        
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
    if cmap_kwargs    is None: cmap_kwargs    = {}
    if contour_kwargs is None: contour_kwargs = {}
    if clabel_kwargs  is None: clabel_kwargs  = {}
    if cutout_kwargs  is None: cutout_kwargs  = {}
        
    # Cutout first
    od = od.subsample.cutout(**cutout_kwargs)
        
    # Check and extract T and S
    varList = ['Temp', 'S']
    od = _compute._add_missing_variables(od, varList)
    
    # Compute mean
    if meanAxes is not None:
        mean_ds = _compute.weighted_mean(od, varNameList= ['Temp', 'S'], axesList=meanAxes, storeWeights=False, aliased=False)
        T = mean_ds['w_mean_Temp'].rename('Temp')
        S = mean_ds['w_mean_S'].rename('S')
        lost_dims = list(set(od._ds['Temp'].dims)-set(T.dims))
    else:
        T = od._ds['Temp']
        S = od._ds['S']
        lost_dims = []
    
    # Extract color field, and interpolate if needed
    if colorName is not None:
        
        # Add missing variables (use private)
        _colorName =  _compute._rename_aliased(od, colorName)
        od = _compute._add_missing_variables(od, _colorName)

        # Extract color (use public)
        color = od.dataset[colorName]
        if meanAxes is not None:
            mean_ds = _compute.weighted_mean(od, varNameList= [_colorName], axesList=meanAxes, storeWeights=False, aliased=False)
            color = mean_ds['w_mean_'+_colorName].rename(_colorName)
        else:
            color = od.dataset[colorName]
        grid  = od.grid
        dims2interp = [dim for dim in color.dims if dim not in T.dims]
        
        # Interpolation 
        for dim in dims2interp:
            for axis in od.grid.axes.keys():
                if dim in [od.grid.axes[axis].coords[k].name for k in od.grid.axes[axis].coords.keys()]: 
                    print('Interpolating [{}] along [{}]-axis.'.format(colorName, axis))
                    attrs = color.attrs
                    color = grid.interp(color, axis, to='center', boundary='fill', fill_value=_np.nan)
                    color.attrs = attrs
                        
        # Broadcast, in case color has different dimensions
        T, S, color = _xr.broadcast(T, S, color)
    
    # Compute density
    T = T.persist()
    S = S.persist()
    if Tlim is None:
        Tlim = [T.min().values, T.max().values]
    if Slim is None:
        Slim = [S.min().values, S.max().values]
        
    if dens is None:
        print('Isopycnals: ', end='')
        t, s = _xr.broadcast(_xr.DataArray(_np.linspace(Tlim[0], Tlim[-1], 100), dims= ('t')),
                             _xr.DataArray(_np.linspace(Slim[0], Slim[-1], 100), dims= ('s')))
        odSigma0 = _ospy.OceanDataset(_xr.Dataset({'Temp': t, 'S': s})).set_parameters(od.parameters)
        odSigma0 = odSigma0.compute.potential_density_anomaly()
        odSigma0._ds = odSigma0._ds.set_coords(['Temp', 'S'])
        
        # Freezing point
        paramsList = ['tempFrz0', 'dTempFrz_dS']
        params2use = {par:od.parameters[par] for par in od.parameters if par in paramsList}
        tempFrz0    = params2use['tempFrz0']
        dTempFrz_dS = params2use['dTempFrz_dS']
        freez_point = tempFrz0 + odSigma0._ds['S']*dTempFrz_dS
        
        # Extract Density
        dens = odSigma0._ds['Sigma0'].where(odSigma0._ds['Temp']>freez_point)    

    # Extract temp and salinity
    t = dens['Temp']
    s = dens['S']
    
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
    CS = ax.contour(s.values, t.values, dens.values, **contour_kwargs)
    ax.clabel(CS, **clabel_kwargs)
    
    # Plot freezing point
    if plotFreez:
        paramsList = ['tempFrz0', 'dTempFrz_dS']
        params2use = {par:od.parameters[par] for par in od.parameters if par in paramsList}
        tempFrz0    = params2use['tempFrz0']
        dTempFrz_dS = params2use['dTempFrz_dS']
        s = _np.unique(s.values.flatten())
        ax.plot(s, tempFrz0 +s*dTempFrz_dS, 'b')
    
    # Set labels and limits
    ax.set_xlabel(_xr.plot.utils.label_from_attrs(S))
    ax.set_ylabel(_xr.plot.utils.label_from_attrs(T))
    ax.set_xlim(Slim)
    ax.set_ylim(Tlim)
    
    # Set title
    title = ''
    T = od._ds['Temp']
    for dim in list(T.dims):
        dim2rem = [d for d in T.dims if len(T[d])==1 and d!=dim]
        tit0 = T.squeeze(dim2rem).drop(dim2rem).isel({dim: 0})._title_for_slice()
        tit1 = T.squeeze(dim2rem).drop(dim2rem).isel({dim: -1})._title_for_slice()
        
        if tit0==tit1:
            tit = tit0
        else:
            if dim in lost_dims:
                tit0 = tit0.replace(dim+' = ', dim+': mean from ')
            else:
                tit0 = tit0.replace(dim+' = ', dim+': from ')
            tit1 = tit1.replace(dim+' = ', ' to ')
            tit = tit0 + tit1
        if title=='':
            title = title + tit
        else:
            title = title + '\n' + tit
    ax.set_title(title)
    _plt.tight_layout()
    
    return ax


def time_series(od, 
                varName, 
                meanAxes      = False, 
                intAxes       = False,
                cutout_kwargs = None,
                **kwargs):
        
    """
    Plot time series.
    
    Parameters
    ----------
    od: OceanDataset
        oceandataset to check for missing variables
    varName: str, None
        Name of the variable to plot.
    meanAxes: 1D array_like, str, or bool
        List of axes over which to apply mean.
        If True, set meanAxes=od.grid_coords (excluding time, mooring, station).
        If False, does not apply mean.
    intAxes: 1D array_like, str, or bool
        List of axes over which to integrate.
        Integration is performed after mean.
        If True, set intAxes=od.grid_coords (excluding time, mooring, station).
        If False, does not apply int.
    cutout_kwargs: dict
        Keyword arguments for subsample.cutout
    **kwargs:
        Kewyword arguments for xarray.plot.line
        
    Returns
    -------
    Axes object
    
    See also
    --------
    subsample.coutout
    
    References
    ----------
    http://xarray.pydata.org/en/stable/generated/xarray.plot.line.html#xarray.plot.line
    """
        
    
    # Check parameters
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    if not isinstance(varName, str):
        raise TypeError('`varName` must be str')
    
    meanAxes, intAxes = _check_mean_and_int_axes(od=od, 
                                                 meanAxes=meanAxes, intAxes=intAxes, 
                                                 exclude=['time', 'mooring', 'station'])
        
    if not isinstance(cutout_kwargs, (dict, type(None))):
        raise ValueError('`cutout_kwargs` must be dict or None')
        
    # Handle kwargs
    if cutout_kwargs is None:  cutout_kwargs = {}
    
    # Cutout first
    od = od.subsample.cutout(**cutout_kwargs)
    
    # Variable name 
    _varName =  _compute._rename_aliased(od, varName)
    od = _compute._add_missing_variables(od, _varName)
    
    # Get time name
    time_name = [dim for dim in od.grid_coords['time'] if dim in od.dataset[varName].dims][0]
    
    # Mean and sum
    da, varName = _compute_mean_and_int(od, varName, meanAxes, intAxes)

    # Check
    if len(da.shape)>2:
        dims = list(da.dims)
        dims.remove(time_name)
        raise ValueError('Timeseries containing multiple dimension other than time: {}'.format(dims))
        
    # Plot
    _ = da.plot.line(**{'x': time_name, **kwargs})
    _plt.tight_layout()
    
    return _plt.gca()



def horizontal_section(od, 
                       varName, 
                       plotType       = 'pcolormesh',
                       use_coords     = True,
                       contourName    = None,
                       meanAxes       = False, 
                       intAxes        = False,
                       contour_kwargs = None,
                       clabel_kwargs  = None,
                       cutout_kwargs  = None,
                       **kwargs):
        
    """
    Plot horizontal sections.
    
    Parameters
    ----------
    od: OceanDataset
        oceandataset to check for missing variables
    varName: str, None
        Name of the variable to plot.
    plotType: str 
        2D plot type: {'contourf', 'contour', 'imshow', 'pcolormesh'}
    use_coords: bool
        If True, use coordinates for x and y axis (e.g., XC and YC).
        If False, use dimensions for x and y axis (e.g., X and Y)
    contourName: str, None
        Name of the variable to contour on top.
    meanAxes: 1D array_like, str, or bool
        List of axes over which to apply mean.
        If True, set meanAxes=od.grid_coords (excluding X, Y).
        If False, does not apply mean.
    intAxes: 1D array_like, str, or bool
        List of axes over which to integrate.
        Integration is performed after mean.
        If True, set intAxes=od.grid_coords (excluding X, Y).
        If False, does not apply int.
    contour_kwargs: dict
        Keyword arguments for xarray.plot.contour
    clabel_kwargs: dict
        Keyword arguments for matplotlib.pyplot.clabel
    cutout_kwargs: dict
        Keyword arguments for subsample.cutout
    **kwargs:
        Kewyword arguments for xarray.plot.['plotType']
        
    Returns
    -------
    Axes or FacetGrid object
    
    See also
    --------
    subsample.coutout
    animate.horizontal_section
    
    References
    ----------
    http://xarray.pydata.org/en/stable/plotting.html
    """
            
    # Check parameters
    if 'mooring' in od.grid_coords or 'station' in od.grid_coords:
        raise ValueError('`od` can not be a mooring or a survey')
    
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    if not isinstance(varName, str):
        raise TypeError('`varName` must be str')
        
    plotTypes = ['contourf', 'contour', 'imshow', 'pcolormesh']
    if not isinstance(plotType, str):
        raise TypeError('`plotType` must be str')
    elif plotType not in plotTypes:
        raise TypeError('plotType [{}] not available. Options are: {}'.format(plotType, plotTypes))
    
    if not isinstance(use_coords, bool):
        raise TypeError('`use_coords` must be bool')
        
    if not isinstance(contourName, (type(None), str)):
        raise TypeError('`contourName` must be str or None')
        
    meanAxes, intAxes = _check_mean_and_int_axes(od=od, 
                                                 meanAxes=meanAxes, intAxes=intAxes, 
                                                 exclude=['X', 'Y'])
        
    if not isinstance(contour_kwargs, (type(None), dict)):
        raise TypeError('`contour_kwargs` must be None or dict')
       
    if not isinstance(clabel_kwargs, (type(None), dict)):
        raise TypeError('`clabel_kwargs` must be None or dict')
        
    if not isinstance(cutout_kwargs, (type(None), dict)):
        raise TypeError('`cutout_kwargs` must be None or dict')
        
    # Handle kwargs
    if contour_kwargs  is None: contour_kwargs = {}
    if clabel_kwargs   is None: clabel_kwargs  = {}
    if cutout_kwargs   is None: cutout_kwargs  = {}
        
    # Cutout first
    od = od.subsample.cutout(**cutout_kwargs)
    
    # Check variables and add
    listName = [varName]
    if contourName is not None: listName = listName + [contourName]
    _listName =  _compute._rename_aliased(od, listName)
    od = _compute._add_missing_variables(od, _listName)
    
    # Apply mean and sum
    da, varName = _compute_mean_and_int(od, varName, meanAxes, intAxes)
    
    # SQUEEZE! Otherwise animation don't show up because xarray make a faceted plot
    da = da.squeeze()
    
    # Get dimension names
    X_name = [dim for dim in od.grid_coords['X'] if dim in da.dims][0]
    Y_name = [dim for dim in od.grid_coords['Y'] if dim in da.dims][0]
    
    # CONTOURNAME
    if contourName is not None: 

        # Apply mean and sum
        da_contour, contourName = _compute_mean_and_int(od, contourName, meanAxes, intAxes)
        
        # SQUEEZE! Otherwise animation don't show up because xarray make a faceted plot
        da_contour = da_contour.squeeze()
        
        # Get dimension names
        X_name_cont = [dim for dim in od.grid_coords['X'] if dim in da_contour.dims][0]
        Y_name_cont = [dim for dim in od.grid_coords['Y'] if dim in da_contour.dims][0]   
        
    # Check dimensions
    dims = list(da.dims)
    dims.remove(X_name)
    dims.remove(Y_name)
    
    # Use coordinates
    if use_coords:
        if X_name=='X' and Y_name=='Y': 
            point = 'C'
        elif X_name=='Xp1' and Y_name=='Y':
            point = 'U'
        elif X_name=='X' and Y_name=='Yp1':
            point = 'V'
        elif X_name=='Xp1' and Y_name=='Yp1':
            point = 'G'
        X_name = 'X'+point
        Y_name = 'Y'+point

        if contourName is not None:
            if X_name_cont=='X' and Y_name_cont=='Y': 
                point_cont = 'C'
            elif X_name_cont=='Xp1' and Y_name_cont=='Y':
                point_cont = 'U'
            elif X_name_cont=='X' and Y_name_cont=='Yp1':
                point_cont = 'V'
            elif X_name_cont=='Xp1' and Y_name_cont=='Yp1':
                point_cont = 'G'
            X_name_cont = 'X'+point_cont
            Y_name_cont = 'Y'+point_cont
                
    # Pop from kwargs
    ax  = kwargs.pop('ax', None)
    col = kwargs.pop('col', None)
    col_wrap    = kwargs.pop('col_wrap', None)
    subplot_kws = kwargs.pop('subplot_kws', None)
    transform   = kwargs.pop('transform', None)
    
    if len(dims)==0:
        # Single plot:    
        # Add ax
        if ax is None:
            ax = _plt.axes(projection=od.projection);
        kwargs['ax'] = ax
            
    elif len(dims)==1:
        
        # Multiple plots:
        extra_name = dims[0]
        
        # TODO: For some reason, faceting and cartopy are not working very nice with our configurations
        #       Drop it for now, but we need to explore it more
        if od.projection is not None:
            _warnings.warn("\nSwitch projection off."
                           " This function currently does not support faceting for projected plots.", stacklevel=2)
            od = od.set_projection(None)
            transform = None
        
        # Add col
        if col is None:
            col = extra_name
        kwargs['col'] = col
        kwargs['col_wrap'] = col_wrap
        
        # Add projection
        if isinstance(subplot_kws, dict):
            projection = subplot_kws.pop('projection', None)
            if projection is None:
                projection = od.projection
            subplot_kws['projection'] = projection
        else:
            subplot_kws = {'projection': od.projection}
        kwargs['subplot_kws'] = subplot_kws
            
    # Add transform
    if transform is None and od.projection is not None:
        kwargs['transform'] = _ccrs.PlateCarree()
        
    # Plot
    args = {'x': X_name, 'y': Y_name, **kwargs}
    plotfunc = eval('_xr.plot.'+plotType)
    p = plotfunc(da, **args)

    # Contour
    if contourName is not None: 
        ax          = args.pop('ax', None)
        transform   = args.pop('transform', None)
        subplot_kws = args.pop('subplot_kws', None)
        args        = {'x': X_name_cont, 'y': Y_name_cont, 'ax': ax, 'transform': transform, 'subplot_kws': subplot_kws, 'colors': 'gray', 'add_labels': False, **contour_kwargs}
        if ax is not None:
            cont = da_contour.plot.contour(**args, **clabel_kwargs)
            _plt.clabel(cont)
        else:
            for i, thisax in enumerate(p.axes.flat):
                if extra_name in da_contour.dims:
                    da_contour_i = da_contour.isel({extra_name: i}).squeeze()
                else:
                    da_contour_i = da_contour
                cont = da_contour_i.plot.contour(**{**args, 'ax': thisax})
                _plt.clabel(cont, **clabel_kwargs)
    
    # Labels and return
    add_labels = kwargs.pop('add_labels', None)
    if ax is not None: 
        if add_labels is not False:
            try:
                gl = ax.gridlines(crs=transform, draw_labels=True)
                gl.xlabels_top   = False
                gl.ylabels_right = False
            except: pass
        if od.projection is None:
            _plt.tight_layout()
        return ax
    else: 
        return p



def vertical_section(od, 
                     varName, 
                     plotType       = 'pcolormesh',
                     use_dist       = True,
                     subsampMethod  = None,
                     contourName    = None,
                     meanAxes       = False, 
                     intAxes        = False,
                     contour_kwargs = None,
                     clabel_kwargs  = None,
                     subsamp_kwargs = None,
                     cutout_kwargs  = None, 
                     **kwargs):
    """
    Plot vertical section.
    
    Parameters
    ----------
    od: OceanDataset
        oceandataset to check for missing variables
    varName: str, None
        Name of the variable to plot.
    plotType: str 
        2D plot type: {'contourf', 'contour', 'imshow', 'pcolormesh'}
    use_dist: bool
        If True, use distances for x axis.
        If False, use mooring or station.
    subsampMethod: str, None
        Subsample methods: {'mooring_array', 'survey_station'}
    contourName: str, None
        Name of the variable to contour on top.
    meanAxes: 1D array_like, str, or bool
        List of axes over which to apply mean.
        If True, set meanAxes=od.grid_coords (time only).
        If False, does not apply mean.
    intAxes: 1D array_like, str, or bool
        List of axes over which to integrate.
        Integration is performed after mean.
        If True, set intAxes=od.grid_coords (time only).
        If False, does not apply int.
    contour_kwargs: dict
        Keyword arguments for xarray.plot.contour
    clabel_kwargs: dict
        Keyword arguments for matplotlib.pyplot.clabel
    subsamp_kwargs: dict
        Keyword arguments for subsample.mooring_array or subsample.survey_stations
    cutout_kwargs: dict
        Keyword arguments for subsample.cutout
    **kwargs:
        Kewyword arguments for xarray.plot.['plotType']
        
    Returns
    -------
    Axes or FacetGrid object
    
    See also
    --------
    subsample.coutout
    subsample.mooring_array
    subsample.survey_stations
    animate.vertical_section
    
    References
    ----------
    http://xarray.pydata.org/en/stable/plotting.html
    """
    
    # Check parameters
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    if not isinstance(varName, str):
        raise TypeError('`varName` must be str')
    
    plotTypes = ['contourf', 'contour', 'imshow', 'pcolormesh']
    if not isinstance(plotType, str):
        raise TypeError('`plotType` must be str')
    elif plotType not in plotTypes:
        raise TypeError('plotType [{}] not available. Options are: {}'.format(plotType, plotTypes))
        
    if not isinstance(use_dist, bool):
        raise TypeError('`use_dist` must be bool')
        
    if subsampMethod is not None:
        subsampMethods = ['mooring_array', 'survey_stations']
        if not isinstance(subsampMethod, str):
            raise TypeError('`subsampMethod` must be str or None')
        elif subsampMethod not in subsampMethods:
            raise TypeError('subsampMethod [{}] not available. Options are: {}'.format(subsampMethod, subsampMethods))
        
    if not isinstance(contourName, (type(None), str)):
        raise TypeError('`contourName` must be str or None')
        
    meanAxes, intAxes = _check_mean_and_int_axes(od=od, 
                                                 meanAxes=meanAxes, intAxes=intAxes, 
                                                 exclude=['mooring', 'station', 'X', 'Y', 'Z'])     
        
    if not isinstance(contour_kwargs, (type(None), dict)):
        raise TypeError('`contour_kwargs` must be None or dict')
       
    if not isinstance(clabel_kwargs, (type(None), dict)):
        raise TypeError('`clabel_kwargs` must be None or dict')
        
    if not isinstance(subsamp_kwargs, (type(None), dict)):
        raise TypeError('`subsamp_kwargs` must be None or dict')
        
    # Handle kwargs
    if contour_kwargs  is None: contour_kwargs = {}
    if clabel_kwargs   is None: clabel_kwargs  = {}
    
    # For animation purposes.
    # TODO: take out useless variables?
    if cutout_kwargs is not None:     
        # Cutout first (needed)
        od = od.subsample.cutout(**cutout_kwargs)
    
    # Subsample first
    if subsamp_kwargs is not None:
        if subsampMethod=='mooring_array':
            od = od.subsample.mooring_array(**subsamp_kwargs)
        elif subsampMethod=='survey_stations':
            od = od.subsample.survey_stations(**subsamp_kwargs)
        
    # Check variables and add
    listName = [varName]
    if contourName is not None: listName = listName + [contourName]
    _listName =  _compute._rename_aliased(od, listName)
    od = _compute._add_missing_variables(od, _listName)
    
    # Apply mean and sum
    da = od.dataset[varName]
    if 'time' in da.dims or 'time_midp' in da.dims:
        da, varName = _compute_mean_and_int(od, varName, meanAxes, intAxes)
        
    # SQUEEZE! Otherwise animation don't show up because xarray make a faceted plot
    da = da.squeeze()
    
    # Get dimension names
    # TODO: make interpolation work with aliases
    
    if 'mooring' in od.grid_coords:
        if 'Xp1' in da.dims:
            print('Regridding [{}] along [{}]-axis.'.format(varName, 'X'))
            da = od.grid.interp(da, 'X')
        if 'Yp1' in da.dims:
            print('Regridding [{}] along [{}]-axis.'.format(varName, 'Y'))
            da = od.grid.interp(da, 'Y')
        da = da.squeeze()
        hor_name = [dim for dim in od.grid_coords['mooring'] if dim in da.dims][0]
    elif 'station' in od.grid_coords:
        hor_name = [dim for dim in od.grid_coords['station'] if dim in da.dims][0]
    else: 
        raise ValueError('The oceandataset must be subsampled using mooring or survey')
    ver_name = [dim for dim in od.grid_coords['Z'] if dim in da.dims][0]

    
    # CONTOURNAME
    if contourName is not None: 
        
        # Apply mean and sum
        da_contour = od.dataset[contourName]
        if 'time' in da_contour.dims or 'time_midp' in da_contour.dims:
            da_contour, contourName = _compute_mean_and_int(od, contourName, meanAxes, intAxes)
        
        # SQUEEZE! Otherwise animation don't show up because xarray make a faceted plot
        da_contour = da_contour.squeeze()

        # Get dimension names
        # TODO: make interpolation work with aliases
        if 'mooring' in od.grid_coords:
            if 'Xp1' in da_contour.dims:
                print('Regridding [{}] along [{}]-axis.'.format(contourName, 'X'))
                da_contour = od.grid.interp(da_contour, 'X')
            if 'Yp1' in da.dims:
                print('Regridding [{}] along [{}]-axis.'.format(contourName, 'Y'))
                da_contour = od.grid.interp(da_contour, 'Y')
            da_contour = da_contour.squeeze()
            hor_name_cont = [dim for dim in od.grid_coords['mooring'] if dim in da_contour.dims][0]
        elif 'station' in od.grid_coords:
            hor_name_cont = [dim for dim in od.grid_coords['station'] if dim in da_contour.dims][0]
        ver_name_cont = [dim for dim in od.grid_coords['Z'] if dim in da_contour.dims][0]
                
    # Check dimensions
    dims = list(da.dims)
    dims.remove(hor_name)
    dims.remove(ver_name)
    
    # Use distances
    if use_dist:
        if contourName is None:
            if hor_name+'_dist' in da.coords: 
                hor_name=hor_name+'_dist'
        else:
            if hor_name+'_dist' in da.coords and hor_name_cont+'_dist' in da_contour.coords:
                hor_name      =hor_name+'_dist'
                hor_name_cont =hor_name_cont+'_dist'
    
    # Pop from kwargs
    ax  = kwargs.pop('ax', None)
    col = kwargs.pop('col', None)
    
    if len(dims)==0:
        # Single plot:    
        # Add ax
        if ax is None:
            ax = _plt.axes();
        kwargs['ax'] = ax
        
    elif len(dims)==1:
        
        # Multiple plots:
        extra_name = dims[0]
        
        # Add col
        if col is None:
            col = extra_name
        kwargs['col'] = col
    
    # Plot
    args = {'x': hor_name, 'y': ver_name, **kwargs}
    plotfunc = eval('_xr.plot.'+plotType)
    p = plotfunc(da, **args)
        
    # Contour
    if contourName is not None: 
        ax          = args.pop('ax', None)
        args        = {'x': hor_name_cont, 'y': ver_name_cont, 'ax': ax, 'colors': 'gray', 'add_labels': False, **contour_kwargs}
        if ax is not None:
            cont = da_contour.plot.contour(**args, **clabel_kwargs)
            _plt.clabel(cont)
        else:
            for i, thisax in enumerate(p.axes.flat):
                if extra_name in da_contour.dims:
                    da_contour_i = da_contour.isel({extra_name: i}).squeeze()
                else:
                    da_contour_i = da_contour
                cont = da_contour_i.plot.contour(**{**args, 'ax': thisax})
                _plt.clabel(cont, **clabel_kwargs)
          
    
    # Return
    if ax is not None: 
        _plt.tight_layout()
        return ax
    else: 
        return p

    
    
class _plotMethdos(object):
    """
    Enables use of oceanspy.plot functions as attributes on a OceanDataset.
    For example, OceanDataset.plot.TS_diagram
    """
    
    def __init__(self, od):
        self._od = od

    @_functools.wraps(TS_diagram)
    def TS_diagram(self, **kwargs):
        return TS_diagram(self._od, **kwargs)
    
    @_functools.wraps(time_series)
    def time_series(self, **kwargs):
        return time_series(self._od, **kwargs)
    
    @_functools.wraps(horizontal_section)
    def horizontal_section(self, **kwargs):
        return horizontal_section(self._od, **kwargs)
    
    @_functools.wraps(vertical_section)
    def vertical_section(self, **kwargs):
        return vertical_section(self._od, **kwargs)

    
# TODO: document this private functions!    
    
def _check_mean_and_int_axes(od, meanAxes, intAxes, exclude):

    if (meanAxes is True and intAxes is not False) or (intAxes is True and meanAxes is not False):
        raise ValueError('If one between `meanAxes` and `intAxes` is True, the other must be False')

    if not isinstance(meanAxes, bool):
        meanAxes = _np.asarray(meanAxes, dtype='str')
        if meanAxes.ndim == 0: meanAxes = meanAxes.reshape(1)
        elif meanAxes.ndim >1: raise TypeError('Invalid `meanAxes`')
        axis_error = [axis for axis in meanAxes if axis not in od.grid_coords]
        if len(axis_error)!=0:
            raise ValueError('{} are not in od.grid_coords and can not be averaged'.format(axis_error))
        elif 'time' in meanAxes:
            raise ValueError('`time` can not be in `meanAxes`')
    elif meanAxes is True:
        meanAxes = [coord for coord in od.grid_coords if coord not in exclude]
    else:
        meanAxes = []

    if not isinstance(intAxes, bool):
        intAxes = _np.asarray(intAxes, dtype='str')
        if intAxes.ndim == 0: intAxes = intAxes.reshape(1)
        elif intAxes.ndim >1: raise TypeError('Invalid `meanAxes`')
        axis_error = [axis for axis in intAxes if axis not in od.grid_coords]
        if len(axis_error)!=0:
            raise ValueError('{} are not in od.grid_coords and can not be averaged'.format(axis_error))
        elif 'time' in intAxes:
            raise ValueError('`time` can not be in `intAxes`')
    elif intAxes is True:
        intAxes = [coord for coord in od.grid_coords if coord not in exclude]
    else:
        intAxes = []    
            
    if len(intAxes)>0 and len(meanAxes)>0:
        if set(meanAxes).issubset(intAxes) or set(intAxes).issubset(meanAxes):
            raise ValueError('`meanAxes` and `intAxes` can not contain the same Axes')
        
    return meanAxes, intAxes


def _compute_mean_and_int(od, varName, meanAxes, intAxes):
    # Mean and sum
    if len(meanAxes)!=0:
        ds = _compute.weighted_mean(od, varNameList= [varName], axesList=meanAxes, storeWeights=False)
        for var in ds.data_vars:
            varName = var
        od = od.merge_into_oceandataset(ds)
        
    if len(intAxes)!=0:
        ds = _compute.integral(od, varNameList= [varName], axesList=intAxes)
        for var in ds.data_vars:
            varName = var
        od = od.merge_into_oceandataset(ds)
    
    # Extract da
    da = od.dataset[varName]
    return da, varName