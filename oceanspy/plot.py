"""
Plot using OceanDataset objects.
"""

# Instructions for developers:
# 1. All funcions must return plt.Axes or xr.plot.FacetGrid objects,
# 2. Functions should use the cutout_kwargs argument at the beginning.
# 3. Make functions compatible with the animate module,
#    and create a twin function under animate.
# 4. Add new functions to _plotMethods
# 5. Add new functions to docs/api.rst

# Required dependencies (private)
import xarray as _xr
import oceanspy as _ospy
import numpy as _np
import warnings as _warnings
import functools as _functools
import pandas as _pd

# From oceanspy (private)
from . import compute as _compute
from ._ospy_utils import (_rename_aliased, _check_instance,
                          _check_mean_and_int_axes, _check_options)
from .compute import (_add_missing_variables)
from .compute import weighted_mean as _weighted_mean
from .compute import integral as _integral

# Additional dependencies (private)
try:
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
        oceandataset used to plot.
    Tlim: array_like with 2 elements
        Temperature limits on the y axis.
        If None, uses min and max values.
    Slim: array_like with 2 elements
        Salinity limits on the x axis.
        If None, uses min and max values.
    dens: xarray.DataArray
        DataArray with densities used for isopycnals.
        Must have coordinates (Temp, S).
        In None, dens is inferred from Temp and S.
    meanAxes: 1D array_like, str, or None
        List of axes over which to apply weighted mean.
        If None, don't average.
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
        Keyword arguments for
        :py:func:`matplotlib.pytplot.contour` (isopycnals)
    clabel_kwargs: dict
        Keyword arguments for
        :py:func:`matplotlib.pytplot.clabel` (isopycnals)
    cutout_kwargs: dict
        Keyword arguments for
        :py:func:`oceanspy.subsample.cutout`
    **kwargs:
        If colorName is None:
        Kewyword arguments for :py:func:`matplotlib.pytplot.plot`
        Otherwise,
        kewyword arguments for :py:func:`matplotlib.pytplot.scatter`

    Returns
    -------
    ax: matplotlib.pyplot.axes
        Axes object.

    References
    ----------
    http://xarray.pydata.org/en/stable/plotting.html#introduction

    See Also
    --------
    oceanspy.animate.TS_diagram
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
        oceandataset used to plot.
    varName: str, None
        Name of the variable to plot.
    meanAxes: 1D array_like, str, or bool
        List of axes over which to apply
        :py:func:`oceanspy.compute.weighted_mean`.
        If True,
        set meanAxes= :py:attr:`oceanspy.OceanDataset.grid_coords`.
        If False, skip weighted mean.
    intAxes: 1D array_like, str, or bool
        List of axes over which to apply
        :py:func:`oceanspy.compute.integral`.
        If True,
        set intAxes= :py:attr:`oceanspy.OceanDataset.grid_coords`.
        If False, skip integral.
    cutout_kwargs: dict
        Keyword arguments for
        :py:func:`oceanspy.subsample.cutout`
    **kwargs:
        Kewyword arguments for :py:func:`xarray.plot.line`

    Returns
    -------
    ax: matplotlib.pyplot.axes
        Axes object.

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
    Plot horizontal section.

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to plot.
    varName: str, None
        Name of the variable to plot.
    plotType: str
        2D plot type.
        Options: {'contourf', 'contour', 'imshow', 'pcolormesh'}
    use_coords: bool
        If True, use coordinates for x and y axis (e.g., XC and YC).
        If False, use dimensions for x and y axis (e.g., X and Y)
    contourName: str, None
        Name of the variable to contour on top.
    meanAxes: 1D array_like, str, or bool
        List of axes over which to apply
        :py:func:`oceanspy.compute.weighted_mean`.
        If True,
        set meanAxes= :py:attr:`oceanspy.OceanDataset.grid_coords`.
        If False, skip weighted mean.
    intAxes: 1D array_like, str, or bool
        List of axes over which to apply
        :py:func:`oceanspy.compute.integral`.
        If True,
        set intAxes= :py:attr:`oceanspy.OceanDataset.grid_coords`.
        If False, skip integral.
    contour_kwargs: dict
        Keyword arguments for :py:func:`xarray.plot.contour`
    clabel_kwargs: dict
        Keyword arguments for :py:func:`matplotlib.pyplot.clabel`
    cutout_kwargs: dict
        Keyword arguments for
        :py:func:`oceanspy.subsample.cutout`
    **kwargs:
        Kewyword arguments for :py:mod:`xarray.plot`.plotType

    Returns
    -------
    matplotlib.pyplot.axes or xarray.plot.FacetGrid

    References
    ----------
    http://xarray.pydata.org/en/stable/plotting.html

    See Also
    --------
    oceanspy.animate.horizontal_section
    """

    # Check parameters
    _check_instance({'od': od,
                     'varName': varName,
                     'plotType': plotType,
                     'use_coords': use_coords,
                     'contourName': contourName,
                     'contour_kwargs': contour_kwargs,
                     'clabel_kwargs': clabel_kwargs,
                     'cutout_kwargs': cutout_kwargs},
                    {'od': 'oceanspy.OceanDataset',
                     'varName': 'str',
                     'plotType': 'str',
                     'use_coords': 'bool',
                     'contourName': ['type(None)', 'str'],
                     'contour_kwargs': ['type(None)', 'dict'],
                     'clabel_kwargs': ['type(None)', 'dict'],
                     'cutout_kwargs': ['type(None)', 'dict']})

    # Check oceandataset
    wrong_dims = ['mooring', 'station', 'particle']
    if any([dim in od._ds.dims for dim in wrong_dims]):
        raise ValueError('`plot.vertical_section` does not support'
                         ' `od` with the following dimensions: '
                         '{}'.format(wrong_dims))

    # Check plot
    _check_options(name='plotType',
                   selected=plotType,
                   options=['contourf', 'contour', 'imshow', 'pcolormesh'])

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

    # Check mean and int axes
    meanAxes, intAxes = _check_mean_and_int_axes(od=od,
                                                 meanAxes=meanAxes,
                                                 intAxes=intAxes,
                                                 exclude=['X', 'Y'])

    # Apply mean and sum
    da, varName = _compute_mean_and_int(od, varName, meanAxes, intAxes)

    # SQUEEZE! Otherwise animation don't show up
    # because xarray makes a faceted plot
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
        # because xarray makes a faceted plot
        da_contour = da_contour.squeeze()

        # Get dimension names
        X_name_cont = [dim
                       for dim in od.grid_coords['X']
                       if dim in da_contour.dims][0]
        Y_name_cont = [dim
                       for dim in od.grid_coords['Y']
                       if dim in da_contour.dims][0]

    # Get dimensions
    dims = list(da.dims)
    dims.remove(X_name)
    dims.remove(Y_name)

    # Use coordinates
    if use_coords:
        al_dim = {}
        for dim in ['X', 'Y', 'Xp1', 'Yp1']:
            al_dim[dim] = _rename_aliased(od, varNameList=dim)

        if X_name == al_dim['X'] and Y_name == al_dim['Y']:
            point = 'C'
        elif X_name == al_dim['Xp1'] and Y_name == al_dim['Y']:
            point = 'U'
        elif X_name == al_dim['X'] and Y_name == al_dim['Yp1']:
            point = 'V'
        else:
            point = 'G'
        X_name = _rename_aliased(od, varNameList='X'+point)
        Y_name = _rename_aliased(od, varNameList='Y'+point)

        if contourName is not None:
            if all([X_name_cont == al_dim['X'],
                    Y_name_cont == al_dim['Y']]):
                point_cont = 'C'
            elif all([X_name_cont == al_dim['Xp1'],
                      Y_name_cont == al_dim['Y']]):
                point_cont = 'U'
            elif all([X_name_cont == al_dim['X'],
                      Y_name_cont == al_dim['Yp1']]):
                point_cont = 'V'
            else:
                point_cont = 'G'
            X_name_cont = _rename_aliased(od, varNameList='X'+point_cont)
            Y_name_cont = _rename_aliased(od, varNameList='Y'+point_cont)

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

    else:
        raise ValueError('There are too many dimensions: {}.'
                         'A maximum of 3 dimensions (including time)'
                         ' are supported.'
                         'Reduce the number of dimensions using'
                         '`meanAxes` and/or `intAxes`'.format(dims))

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
            cont = da_contour.plot.contour(**args)
            _plt.clabel(cont, **clabel_kwargs)
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
        if od.projection is None:
            _plt.tight_layout()
        else:
            if add_labels is not False:
                try:
                    gl = ax.gridlines(crs=transform,
                                      draw_labels=True)
                    gl.xlabels_top = False
                    gl.ylabels_right = False
                except TypeError:
                    # Gridlines don't work with all projections
                    pass
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
        oceandataset used to plot.
    varName: str, None
        Name of the variable to plot.
    plotType: str
        2D plot type.
        Options: {'contourf', 'contour', 'imshow', 'pcolormesh'}
    use_dist: bool
        If True, use distances for x axis.
        If False, use mooring or station.
    subsampMethod: str, None
        Subsample method.
        Options: {'mooring_array', 'survey_station'}
    contourName: str, None
        Name of the variable to contour on top.
    meanAxes: 1D array_like, str, or bool
        List of axes over which to apply
        :py:func:`oceanspy.compute.weighted_mean`.
        If True,
        set meanAxes= :py:attr:`oceanspy.OceanDataset.grid_coords`.
        If False, skip weighted mean.
    intAxes: 1D array_like, str, or bool
        List of axes over which to apply
        :py:func:`oceanspy.compute.integral`.
        If True,
        set intAxes= :py:attr:`oceanspy.OceanDataset.grid_coords`.
        If False, skip integral.
    contour_kwargs: dict
        Keyword arguments for :py:func:`xarray.plot.contour`
    clabel_kwargs: dict
        Keyword arguments for :py:func:`matplotlib.pyplot.clabel`
    subsamp_kwargs: dict
        Keyword arguments for
        :py:func:`oceanspy.subsample.mooring_array`
        or :py:func:`oceanspy.subsample.survey_stations`
    cutout_kwargs: dict
        Keyword arguments for
        :py:func:`oceanspy.subsample.cutout`
    **kwargs:
        Kewyword arguments for :py:mod:`xarray.plot`.plotType

    Returns
    -------
    matplotlib.pyplot.axes or xarray.plot.FacetGrid

    References
    ----------
    http://xarray.pydata.org/en/stable/plotting.html

    See Also
    --------
    oceanspy.animate.vertical_section
    """

    # Check parameters
    _check_instance({'od': od,
                     'varName': varName,
                     'plotType': plotType,
                     'use_dist': use_dist,
                     'subsampMethod': subsampMethod,
                     'contourName': contourName,
                     'contour_kwargs': contour_kwargs,
                     'clabel_kwargs': clabel_kwargs,
                     'subsamp_kwargs': subsamp_kwargs},
                    {'od': 'oceanspy.OceanDataset',
                     'varName': 'str',
                     'plotType': 'str',
                     'use_dist': 'bool',
                     'subsampMethod': ['type(None)', 'str'],
                     'contourName': ['type(None)', 'str'],
                     'contour_kwargs': ['type(None)', 'dict'],
                     'clabel_kwargs': ['type(None)', 'dict'],
                     'subsamp_kwargs': ['type(None)', 'dict']})

    # Check plot
    _check_options(name='plotType',
                   selected=plotType,
                   options=['contourf', 'contour', 'imshow', 'pcolormesh'])

    # Check subsample
    if subsampMethod is not None:
        # Check plot
        _check_options(name='subsampMethod',
                       selected=subsampMethod,
                       options=['mooring_array', 'survey_stations'])

    # Handle kwargs
    if contour_kwargs is None:
        contour_kwargs = {}
    if clabel_kwargs is None:
        clabel_kwargs = {}
    if cutout_kwargs is None:
        cutout_kwargs = {}

    # For animation purposes.
    if len(cutout_kwargs) != 0:
        od = od.subsample.cutout(**cutout_kwargs)

    # Subsample first
    if subsamp_kwargs is not None and subsampMethod is not None:
        if subsampMethod == 'mooring_array':
            od = od.subsample.mooring_array(**subsamp_kwargs)
        else:
            # survey_stations
            od = od.subsample.survey_stations(**subsamp_kwargs)

    # Check oceandataset
    needed_dims = ['mooring', 'station']
    if not any([dim in od.grid_coords.keys() for dim in needed_dims]):
        raise ValueError('`plot.vertical_section` only supports'
                         ' `od` with one of the following grid coordinates: '
                         '{}'.format(needed_dims))

    # Check variables and add
    listName = [varName]
    if contourName is not None:
        listName = listName + [contourName]
    _listName = _rename_aliased(od, listName)
    od = _add_missing_variables(od, _listName)

    # Check mean and int axes
    meanAxes, intAxes = _check_mean_and_int_axes(od=od,
                                                 meanAxes=meanAxes,
                                                 intAxes=intAxes,
                                                 exclude=['mooring',
                                                          'station',
                                                          'X', 'Y', 'Z'])

    # Apply mean and sum
    da, varName = _compute_mean_and_int(od, varName, meanAxes, intAxes)

    # SQUEEZE! Otherwise animation don't show up
    # because xarray makes a faceted plot
    da = da.squeeze()
    da, hor_name = _Vsection_regrid(od, da, varName)
    ver_name = [dim for dim in od.grid_coords['Z'] if dim in da.dims][0]
    da = da.squeeze()

    # CONTOURNAME
    if contourName is not None:

        # Apply mean and sum
        da_contour = od.dataset[contourName]
        da_contour, contourName = _compute_mean_and_int(od, contourName,
                                                        meanAxes, intAxes)

        # SQUEEZE! Otherwise animation don't show up
        # because xarray makes a faceted plot
        da_contour = da_contour.squeeze()

        # Get dimension names
        da_contour, hor_name_cont = _Vsection_regrid(od,
                                                     da_contour, contourName)
        ver_name_cont = [dim
                         for dim in od.grid_coords['Z']
                         if dim in da_contour.dims]
        if len(ver_name_cont) != 1:
            raise ValueError("Couldn't find Z dimension of [{}]"
                             "".format(contourName))
        else:
            ver_name_cont = ver_name_cont[0]
        da_contour = da_contour.squeeze()

    # Check dimensions
    dims = list(da.dims)
    dims.remove(hor_name)
    dims.remove(ver_name)

    # Use distances
    if use_dist:
        if hor_name + '_dist' in da.coords:
            hor_name = hor_name + '_dist'
            hor_name_cont = hor_name

    # Pop from kwargs
    ax = kwargs.pop('ax', None)
    col = kwargs.pop('col', None)

    if len(dims) == 0:
        # Single plot:
        # Add ax
        if ax is None:
            ax = _plt.axes()
        kwargs['ax'] = ax
    else:
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
            cont = da_contour.plot.contour(**args)
            _plt.clabel(cont, **clabel_kwargs)
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


def _Vsection_regrid(od, da, varName):

    if 'mooring' in od.grid_coords:
        # Time coordinates
        if 'time' in od.grid_coords.keys():
            time_coords = {timeName: da[timeName]
                           for timeName in od.grid_coords['time'].keys()
                           if timeName in da.coords}
        else:
            time_coords = {}

        # Regrid to center dim
        for axis in ['X', 'Y']:
            dim2regrid = [dim
                          for dim in od.grid_coords[axis]
                          if (od.grid_coords[axis][dim] is not None
                              and dim in da.dims)]
            if len(dim2regrid) != 0:
                print('Regridding [{}] along [{}]-axis.'
                      ''.format(varName, axis))
                da_attrs = da.attrs
                da = od.grid.interp(da, axis)
                da.attrs = da_attrs
            hor_name = [dim
                        for dim in od.grid_coords['mooring']
                        if dim in da.dims]
            if len(hor_name) != 1:
                raise ValueError("Couldn't find `mooring` dimension of [{}]"
                                 "".format(varName))
            else:
                hor_name = hor_name[0]
            da = da.assign_coords(**time_coords)
            if hor_name+'_dist' in od._ds.coords:
                da = da.assign_coords(**{hor_name+'_dist':
                                         od._ds[hor_name+'_dist']})
            for toRem in ['X', 'Y', 'Xp1', 'Yp1']:
                toRem = _rename_aliased(od, varNameList=toRem)
                if toRem in da.coords:
                    da = da.drop(toRem)
    else:
        # Station
        hor_name = [dim
                    for dim in od.grid_coords['station']
                    if dim in da.dims]
        if len(hor_name) != 1:
            raise ValueError("Couldn't find `station` dimension of [{}]"
                             "".format(varName))
        else:
            hor_name = hor_name[0]

    return da, hor_name


class _plotMethods(object):
    """
    Enables use of functions as OceanDataset attributes.
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
