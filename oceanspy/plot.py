"""
Plot using OceanDataset objects.
"""
# TODO: add test that check squeezing!

# Required dependencies
import xarray as _xr
import oceanspy as _ospy
import numpy as _np
import warnings as _warnings
import functools as _functools
import pandas as _pd

# From oceanspy
from . import compute as _compute
from ._ospy_utils import (_rename_aliased, _check_instance,
                          _check_mean_and_int_axes)
from .compute import (_add_missing_variables)
from .compute import weighted_mean as _weighted_mean
from .compute import integral as _integral

# Additional dependencies
try:
    import matplotlib as _matplotlib
    _matplotlib.use('agg')
    import matplotlib.pyplot as _plt
except ImportError:  # pragma: no cover
    pass
try:
    import cartopy.crs as _ccrs
except ImportError:  # pragma: no cover
    pass


def TS_diagram(od,
               Tlim=None,
               Slim=None,
               dens=None,
               meanAxes=None,
               colorName=None,
               plotFreez=True,
               ax=None,
               cmap_kwargs=None,
               contour_kwargs=None,
               clabel_kwargs=None,
               cutout_kwargs=None,
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
    _check_instance({'od': od,
                     'colorName': colorName,
                     'plotFreez': plotFreez,
                     'ax': ax,
                     'cmap_kwargs': cmap_kwargs,
                     'contour_kwargs': contour_kwargs,
                     'clabel_kwargs': clabel_kwargs,
                     'cutout_kwargs': cutout_kwargs,
                     'dens': dens},
                    {'od': 'oceanspy.OceanDataset',
                     'colorName': ['type(None)', 'str'],
                     'plotFreez': 'bool',
                     'ax': ['type(None)', 'matplotlib.pyplot.Axes'],
                     'cmap_kwargs': ['type(None)', 'dict'],
                     'contour_kwargs': ['type(None)', 'dict'],
                     'clabel_kwargs': ['type(None)', 'dict'],
                     'cutout_kwargs': ['type(None)', 'dict'],
                     'dens': ['type(None)', 'xarray.DataArray']})

    if Tlim is not None:
        Tlim = _np.asarray(Tlim)
        if Tlim.size != 2:
            raise ValueError('`Tlim` must contain 2 elements')
        Tlim = Tlim.reshape(2)

    if Slim is not None:
        Slim = _np.asarray(Slim)
        if Slim.size != 2:
            raise ValueError('`Slim` must contain 2 elements')
        Slim = Slim.reshape(2)

    if dens is not None and not set(['Temp', 'S']).issubset(dens.coords):
        raise ValueError('`dens` must have coordinates (Temp, S)')

    # Change None in empty dict
    if cmap_kwargs is None:
        cmap_kwargs = {}
    if contour_kwargs is None:
        contour_kwargs = {}
    if clabel_kwargs is None:
        clabel_kwargs = {}
    if cutout_kwargs is None:
        cutout_kwargs = {}

    # Cutout first
    if len(cutout_kwargs) != 0:
        od = od.subsample.cutout(**cutout_kwargs)

    # Check and extract T and S
    varList = ['Temp', 'S']
    od = _add_missing_variables(od, varList)

    # Compute mean
    if meanAxes is not None:
        mean_ds = _compute.weighted_mean(od, varNameList=['Temp', 'S'],
                                         axesList=meanAxes,
                                         storeWeights=False, aliased=False)
        T = mean_ds['w_mean_Temp'].rename('Temp')
        S = mean_ds['w_mean_S'].rename('S')
        lost_coords = list(set(od._ds['Temp'].dims)-set(T.coords))
    else:
        T = od._ds['Temp']
        S = od._ds['S']
        lost_coords = []

    # Extract color field, and interpolate if needed
    if colorName is not None:

        # Add missing variables (use private)
        _colorName = _rename_aliased(od, colorName)
        od = _add_missing_variables(od, _colorName)

        # Extract color (use public)
        color = od.dataset[colorName]
        if meanAxes is not None:
            mean_ds = _compute.weighted_mean(od, varNameList=_colorName,
                                             axesList=meanAxes,
                                             storeWeights=False,
                                             aliased=False)
            color = mean_ds['w_mean_'+_colorName].rename(_colorName)
        else:
            color = od.dataset[colorName]
        grid = od.grid
        dims2interp = [dim for dim in color.dims if dim not in T.dims]

        # Interpolation
        for dim in dims2interp:
            for axis in od.grid.axes.keys():
                if dim in [od.grid.axes[axis].coords[k]
                           for k in od.grid.axes[axis].coords.keys()]:
                    print('Interpolating [{}] along [{}]-axis.'
                          ''.format(colorName, axis))
                    attrs = color.attrs
                    color = grid.interp(color, axis, to='center',
                                        boundary='fill', fill_value=_np.nan)
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
        tlin = _xr.DataArray(_np.linspace(Tlim[0], Tlim[-1], 100), dims=('t'))
        slin = _xr.DataArray(_np.linspace(Slim[0], Slim[-1], 100), dims=('s'))
        t, s = _xr.broadcast(tlin, slin)
        odSigma0 = _ospy.OceanDataset(_xr.Dataset({'Temp': t, 'S': s}))
        odSigma0 = odSigma0.set_parameters(od.parameters)
        odSigma0 = odSigma0.compute.potential_density_anomaly()
        odSigma0._ds = odSigma0._ds.set_coords(['Temp', 'S'])

        # Freezing point
        paramsList = ['tempFrz0', 'dTempFrz_dS']
        params2use = {par: od.parameters[par]
                      for par in od.parameters
                      if par in paramsList}
        tempFrz0 = params2use['tempFrz0']
        dTempFrz_dS = params2use['dTempFrz_dS']
        freez_point = tempFrz0 + odSigma0._ds['S']*dTempFrz_dS

        # Extract Density
        dens = odSigma0._ds['Sigma0'].where(odSigma0._ds['Temp'] > freez_point)

    # Create axis
    if ax is None:
        ax = _plt.gca()

    # Use plot if colorless (faster!), otherwise use scatter
    if colorName is None:
        default_kwargs = {'color': 'k', 'linestyle': 'None', 'marker': '.'}
        kwargs = {**default_kwargs, **kwargs}
        ax.plot(S.values.flatten(), T.values.flatten(), **kwargs)
    else:
        # Mask points out of axes
        color = color.where(_np.logical_and(T > min(Tlim), T < max(Tlim)))
        color = color.where(_np.logical_and(S > min(Slim), T < max(Slim)))
        color = color.stack(all_dims=color.dims)
        c = color.values

        # Create colorbar (stolen from xarray)
        cmap_kwargs['plot_data'] = c
        cmap_params = _xr.plot.utils._determine_cmap_params(**cmap_kwargs)
        extend = cmap_params.pop('extend')
        _ = cmap_params.pop('levels')
        kwargs = {**cmap_params, **kwargs}
        # Scatter
        sc = ax.scatter(S.values.flatten(), T.values.flatten(), c=c, **kwargs)
        _plt.colorbar(sc, label=_xr.plot.utils.label_from_attrs(color),
                      extend=extend)

    # Plot isopycnals
    t = dens['Temp']
    s = dens['S']
    default_contour_kwargs = {'colors': 'gray'}
    contour_kwargs = {**default_contour_kwargs, **contour_kwargs}
    CS = ax.contour(s.values, t.values, dens.values, **contour_kwargs)
    ax.clabel(CS, **clabel_kwargs)

    # Plot freezing point
    if plotFreez:
        paramsList = ['tempFrz0', 'dTempFrz_dS']
        params2use = {par: od.parameters[par]
                      for par in od.parameters
                      if par in paramsList}
        tempFrz0 = params2use['tempFrz0']
        dTempFrz_dS = params2use['dTempFrz_dS']
        s = _np.unique(s.values.flatten())
        ax.plot(s, tempFrz0 + s*dTempFrz_dS, 'b')

    # Set labels and limits
    ax.set_xlabel(_xr.plot.utils.label_from_attrs(S))
    ax.set_ylabel(_xr.plot.utils.label_from_attrs(T))
    ax.set_xlim(Slim)
    ax.set_ylim(Tlim)

    # Set title
    title = []
    all_coords = list(lost_coords) + list(T.coords)
    skip_coords = ['X', 'Y', 'Xp1', 'Yp1']
    if any([dim in od._ds.dims for dim in ['mooring', 'station', 'particle']]):
        skip_coords = [coord
                       for coord in od._ds.coords
                       if 'X' in coord
                       or 'Y' in coord]
    for coord in all_coords:
        if coord not in skip_coords:
            if coord in list(lost_coords):
                da = od._ds['Temp']
                pref = '<'
                suf = '>'
            else:
                da = T
                pref = ''
                suf = ''
            rng = [da[coord].min().values, da[coord].max().values]
            units = da[coord].attrs.pop('units', '')
            if units.lower() == 'none':
                units = ''
            if 'time' in coord:
                for i, v in enumerate(rng):
                    ts = _pd.to_datetime(str(v))
                    rng[i] = ts.strftime('%Y-%m-%d %r')

            if rng[0] == rng[-1]:
                rng = '{}'.format(rng[0])
            else:
                rng = 'from {} to {}'.format(rng[0], rng[1])
            title = title + ['{}{}{}: {} {}'
                             ''.format(pref, coord, suf, rng, units)]

    ax.set_title('\n'.join(title))
    _plt.tight_layout()

    return ax


def time_series(od,
                varName,
                meanAxes=False,
                intAxes=False,
                cutout_kwargs=None,
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
        If True,
         set meanAxes=od.grid_coords.
        If False, does not apply mean.
    intAxes: 1D array_like, str, or bool
        List of axes over which to integrate.
        Integration is performed after mean.
        If True, set intAxes=od.grid_coords.
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
    _check_instance({'od': od,
                     'varName': varName,
                     'cutout_kwargs': cutout_kwargs},
                    {'od': 'oceanspy.OceanDataset',
                     'varName': 'str',
                     'cutout_kwargs': ['type(None)', 'dict']})

    # Check mean and int axes
    meanAxes, intAxes = _check_mean_and_int_axes(od=od,
                                                 meanAxes=meanAxes,
                                                 intAxes=intAxes,
                                                 exclude=['time'])

    # Handle kwargs
    if cutout_kwargs is None:
        cutout_kwargs = {}

    # Cutout first
    if len(cutout_kwargs) != 0:
        od = od.subsample.cutout(**cutout_kwargs)

    # Variable name
    _varName = _rename_aliased(od, varName)
    od = _add_missing_variables(od, _varName)

    # Mean and sum
    da, varName = _compute_mean_and_int(od, varName, meanAxes, intAxes)

    # Get time name
    time_name = [dim
                 for dim in od.grid_coords['time']
                 if dim in da.dims]
    if len(time_name) != 1:
        raise ValueError("Couldn't find time dimension")
    else:
        time_name = time_name[0]

    # Check
    if len(da.shape) > 2:
        dims = list(da.dims)
        dims.remove(time_name)
        raise ValueError('Timeseries containing multiple'
                         ' dimension other than time: {}'.format(dims))

    # Plot
    _ = da.plot.line(**{'x': time_name, **kwargs})
    _plt.tight_layout()

    return _plt.gca()


def horizontal_section(od, varName,
                       plotType='pcolormesh',
                       use_coords=True,
                       contourName=None,
                       meanAxes=False,
                       intAxes=False,
                       contour_kwargs=None,
                       clabel_kwargs=None,
                       cutout_kwargs=None,
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
        2D plot type:{'contourf', 'contour', 'imshow', 'pcolormesh'}
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
        raise TypeError('plotType [{}] not available.'
                        ' Options are: {}'.format(plotType, plotTypes))

    if not isinstance(use_coords, bool):
        raise TypeError('`use_coords` must be bool')

    if not isinstance(contourName, (type(None), str)):
        raise TypeError('`contourName` must be str or None')

    meanAxes, intAxes = _check_mean_and_int_axes(od=od,
                                                 meanAxes=meanAxes,
                                                 intAxes=intAxes,
                                                 exclude=['X', 'Y'])

    if not isinstance(contour_kwargs, (type(None), dict)):
        raise TypeError('`contour_kwargs` must be None or dict')

    if not isinstance(clabel_kwargs, (type(None), dict)):
        raise TypeError('`clabel_kwargs` must be None or dict')

    if not isinstance(cutout_kwargs, (type(None), dict)):
        raise TypeError('`cutout_kwargs` must be None or dict')

    # Handle kwargs
    if contour_kwargs is None:
        contour_kwargs = {}
    if clabel_kwargs is None:
        clabel_kwargs = {}
    if cutout_kwargs is None:
        cutout_kwargs = {}

    # Cutout first
    if len(cutout_kwargs) != 0:
        od = od.subsample.cutout(**cutout_kwargs)

    # Check variables and add
    listName = [varName]
    if contourName is not None:
        listName = listName + [contourName]
    _listName = _rename_aliased(od, listName)
    od = _add_missing_variables(od, _listName)

    # Apply mean and sum
    da, varName = _compute_mean_and_int(od, varName, meanAxes, intAxes)

    # SQUEEZE! Otherwise animation don't show up
    # because xarray make a faceted plot
    da = da.squeeze()

    # Get dimension names
    X_name = [dim for dim in od.grid_coords['X'] if dim in da.dims][0]
    Y_name = [dim for dim in od.grid_coords['Y'] if dim in da.dims][0]

    # CONTOURNAME
    if contourName is not None:
        # Apply mean and sum
        da_contour, contourName = _compute_mean_and_int(od,
                                                        contourName,
                                                        meanAxes,
                                                        intAxes)

        # SQUEEZE! Otherwise animation don't show up
        # because xarray make a faceted plot
        da_contour = da_contour.squeeze()

        # Get dimension names
        X_name_cont = [dim
                       for dim in od.grid_coords['X']
                       if dim in da_contour.dims][0]
        Y_name_cont = [dim
                       for dim in od.grid_coords['Y']
                       if dim in da_contour.dims][0]

    # Check dimensions
    dims = list(da.dims)
    dims.remove(X_name)
    dims.remove(Y_name)

    # Use coordinates
    if use_coords:
        if X_name == 'X' and Y_name == 'Y':
            point = 'C'
        elif X_name == 'Xp1' and Y_name == 'Y':
            point = 'U'
        elif X_name == 'X' and Y_name == 'Yp1':
            point = 'V'
        elif X_name == 'Xp1' and Y_name == 'Yp1':
            point = 'G'
        X_name = 'X'+point
        Y_name = 'Y'+point

        if contourName is not None:
            if all([X_name_cont == 'X',
                    Y_name_cont == 'Y']):
                point_cont = 'C'
            elif all([X_name_cont == 'Xp1',
                      Y_name_cont == 'Y']):
                point_cont = 'U'
            elif all([X_name_cont == 'X',
                      Y_name_cont == 'Yp1']):
                point_cont = 'V'
            elif all([X_name_cont == 'Xp1',
                      Y_name_cont == 'Yp1']):
                point_cont = 'G'
            X_name_cont = 'X'+point_cont
            Y_name_cont = 'Y'+point_cont

    # Pop from kwargs
    ax = kwargs.pop('ax', None)
    col = kwargs.pop('col', None)
    col_wrap = kwargs.pop('col_wrap', None)
    subplot_kws = kwargs.pop('subplot_kws', None)
    transform = kwargs.pop('transform', None)

    if len(dims) == 0:
        # Single plot
        # Add ax
        if ax is None:
            ax = _plt.axes(projection=od.projection)
        elif od.projection is not None and not hasattr(ax, 'projection'):
            od = od.set_projection(None)
            _warnings.warn("\nSwitching projection off."
                           "If `ax` is passed, it needs"
                           " to be initialiazed with a projection.",
                           stacklevel=2)
        kwargs['ax'] = ax

    elif len(dims) == 1:
        # Multiple plots:
        extra_name = dims[0]

        # TODO: For some reason, faceting and cartopy are not
        #       working very nice with our configurations
        #       Drop it for now, but we need to explore it more
        if od.projection is not None:
            _warnings.warn("\nSwitch projection off."
                           " This function currently"
                           " does not support faceting for projected plots.",
                           stacklevel=2)
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
        ax = args.pop('ax', None)
        transform = args.pop('transform', None)
        subplot_kws = args.pop('subplot_kws', None)
        args = {'x': X_name_cont,
                'y': Y_name_cont,
                'ax': ax,
                'transform': transform,
                'subplot_kws': subplot_kws,
                'colors': 'gray',
                'add_labels': False,
                **contour_kwargs}
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
                gl = ax.gridlines(crs=transform,
                                  draw_labels=True)
                gl.xlabels_top = False
                gl.ylabels_right = False
            except TypeError:
                # Gridlines don't work with all projections
                pass
        if od.projection is None:
            _plt.tight_layout()
        return ax
    else:
        return p


def vertical_section(od,
                     varName,
                     plotType='pcolormesh',
                     use_dist=True,
                     subsampMethod=None,
                     contourName=None,
                     meanAxes=False,
                     intAxes=False,
                     contour_kwargs=None,
                     clabel_kwargs=None,
                     subsamp_kwargs=None,
                     cutout_kwargs=None,
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
        Keyword arguments for subsample.mooring_array
         or subsample.survey_stations
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
        raise TypeError('plotType [{}] not available. Options are:'
                        ' {}'.format(plotType, plotTypes))

    if not isinstance(use_dist, bool):
        raise TypeError('`use_dist` must be bool')

    if subsampMethod is not None:
        subsampMethods = ['mooring_array', 'survey_stations']
        if not isinstance(subsampMethod, str):
            raise TypeError('`subsampMethod` must be str or None')
        elif subsampMethod not in subsampMethods:
            raise TypeError('subsampMethod [{}] not available.'
                            ' Options are: {}'.format(subsampMethod,
                                                      subsampMethods))

    if not isinstance(contourName, (type(None), str)):
        raise TypeError('`contourName` must be str or None')

    meanAxes, intAxes = _check_mean_and_int_axes(od=od,
                                                 meanAxes=meanAxes,
                                                 intAxes=intAxes,
                                                 exclude=['mooring',
                                                          'station',
                                                          'X', 'Y', 'Z'])

    if not isinstance(contour_kwargs, (type(None), dict)):
        raise TypeError('`contour_kwargs` must be None or dict')

    if not isinstance(clabel_kwargs, (type(None), dict)):
        raise TypeError('`clabel_kwargs` must be None or dict')

    if not isinstance(subsamp_kwargs, (type(None), dict)):
        raise TypeError('`subsamp_kwargs` must be None or dict')

    # Handle kwargs
    if contour_kwargs is None:
        contour_kwargs = {}
    if clabel_kwargs is None:
        clabel_kwargs = {}
    if cutout_kwargs is None:
        cutout_kwargs = {}

    # For animation purposes.
    # TODO: take out useless variables?
    if len(cutout_kwargs) != 0:
        od = od.subsample.cutout(**cutout_kwargs)

    # Subsample first
    if subsamp_kwargs is not None:
        if subsampMethod == 'mooring_array':
            od = od.subsample.mooring_array(**subsamp_kwargs)
        elif subsampMethod == 'survey_stations':
            od = od.subsample.survey_stations(**subsamp_kwargs)

    # Check variables and add
    listName = [varName]
    if contourName is not None:
        listName = listName + [contourName]
    _listName = _rename_aliased(od, listName)
    od = _add_missing_variables(od, _listName)

    # Apply mean and sum
    da = od.dataset[varName]
    if 'time' in da.dims or 'time_midp' in da.dims:
        da, varName = _compute_mean_and_int(od, varName, meanAxes, intAxes)

    # SQUEEZE! Otherwise animation don't show up
    # because xarray make a faceted plot
    da = da.squeeze()
    time_coords = {timeName: da[timeName]
                   for timeName in ['time', 'time_midp']
                   if timeName in da.coords}
    if 'mooring' in od.grid_coords:
        if 'Xp1' in da.dims:
            print('Regridding [{}] along [{}]-axis.'.format(varName, 'X'))
            da_attrs = da.attrs
            da = od.grid.interp(da, 'X')
            da.attrs = da_attrs
        if 'Yp1' in da.dims:
            print('Regridding [{}] along [{}]-axis.'.format(varName, 'Y'))
            da_attrs = da.attrs
            da = od.grid.interp(da, 'Y')
            da.attrs = da_attrs
        hor_name = [dim
                    for dim in od.grid_coords['mooring']
                    if dim in da.dims][0]
        da = da.assign_coords(**time_coords)
        if hor_name+'_dist' in od._ds.coords:
            da = da.assign_coords(**{hor_name+'_dist':
                                     od._ds[hor_name+'_dist']})
        for toRem in ['X', 'Y', 'Xp1', 'Yp1']:
            if toRem in da.coords:
                da = da.drop(toRem)
    elif 'station' in od.grid_coords:
        hor_name = [dim
                    for dim in od.grid_coords['station']
                    if dim in da.dims][0]
    else:
        raise ValueError('The oceandataset must be subsampled'
                         ' using mooring or survey')
    ver_name = [dim for dim in od.grid_coords['Z'] if dim in da.dims][0]
    da = da.squeeze()

    # CONTOURNAME
    if contourName is not None:

        # Apply mean and sum
        da_contour = od.dataset[contourName]
        if 'time' in da_contour.dims or 'time_midp' in da_contour.dims:
            da_contour, contourName = _compute_mean_and_int(od, contourName,
                                                            meanAxes, intAxes)

        # SQUEEZE! Otherwise animation don't show up
        # because xarray make a faceted plot
        da_contour = da_contour.squeeze()

        # Get dimension names
        # TODO: make interpolation work with aliases
        if 'mooring' in od.grid_coords:
            if 'Xp1' in da_contour.dims:
                print('Regridding [{}] along [{}]-axis.'
                      ''.format(contourName, 'X'))
                da_contour_attrs = da_contour.attrs
                da_contour = od.grid.interp(da_contour, 'X')
                da_contour.attrs = da_contour_attrs
            if 'Yp1' in da.dims:
                print('Regridding [{}] along [{}]-axis.'
                      ''.format(contourName, 'Y'))
                da_contour_attrs = da_contour.attrs
                da_contour = od.grid.interp(da_contour, 'Y')
                da_contour.attrs = da_contour_attrs
            hor_name_cont = [dim
                             for dim in od.grid_coords['mooring']
                             if dim in da_contour.dims][0]
            if hor_name+'_dist' in od._ds.coords:
                toassign = {hor_name+'_dist': od._ds[hor_name+'_dist']}
                da_contour = da_contour.assign_coords(**toassign)
            for toRem in ['X', 'Y', 'Xp1', 'Yp1']:
                if toRem in da_contour.coords:
                    da_contour = da_contour.drop(toRem)
        elif 'station' in od.grid_coords:
            hor_name_cont = [dim
                             for dim in od.grid_coords['station']
                             if dim in da_contour.dims][0]
        ver_name_cont = [dim
                         for dim in od.grid_coords['Z']
                         if dim in da_contour.dims][0]
        da_contour = da_contour.squeeze()

    # Check dimensions
    dims = list(da.dims)
    dims.remove(hor_name)
    dims.remove(ver_name)

    # Use distances
    if use_dist:
        if contourName is None:
            if hor_name + '_dist' in da.coords:
                hor_name = hor_name + '_dist'
        else:
            check1 = hor_name + '_dist' in da.coords
            if check1 and hor_name_cont + '_dist' in da_contour.coords:
                hor_name = hor_name + '_dist'
                hor_name_cont = hor_name_cont + '_dist'

    # Pop from kwargs
    ax = kwargs.pop('ax', None)
    col = kwargs.pop('col', None)

    if len(dims) == 0:
        # Single plot:
        # Add ax
        if ax is None:
            ax = _plt.axes()
        kwargs['ax'] = ax

    elif len(dims) == 1:
        # Multiple plots:
        extra_name = dims[0]

        # Add col
        if col is None:
            col = extra_name
        kwargs['col'] = col

    # Plot
    args = {'x': hor_name, 'y': ver_name, **kwargs}
    plotfunc = eval('_xr.plot.' + plotType)
    p = plotfunc(da, **args)

    # Contour
    if contourName is not None:
        ax = args.pop('ax', None)
        args = {'x': hor_name_cont,
                'y': ver_name_cont,
                'ax': ax,
                'colors': 'gray',
                'add_labels': False, **contour_kwargs}
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


def _compute_mean_and_int(od, varName, meanAxes, intAxes):

    # Mean and sum
    if meanAxes is not False:
        ds = _weighted_mean(od, varNameList=[varName],
                            axesList=meanAxes, storeWeights=False)
        for var in ds.data_vars:
            varName = var
        od = od.merge_into_oceandataset(ds)

    if intAxes is not False:
        ds = _integral(od, varNameList=[varName], axesList=intAxes)
        for var in ds.data_vars:
            varName = var
        od = od.merge_into_oceandataset(ds)

    # Extract da
    da = od.dataset[varName]
    return da, varName


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
