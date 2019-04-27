"""
Animate using oceanspy plot functions.
"""

# Instructions for developers:
# 1. All funcions must return matplotlib.animation.FuncAnimation objects
# 2. Animate functions are twins of functions under plot
# 3. Add new functions to _animateMethods
# 4. Add new functions to docs/api.rst

# Required dependencies (private)
import xarray as _xr
import numpy as _np
import oceanspy as _ospy
import functools as _functools

# From OceanSpy (private)
from xarray.plot.utils import _determine_cmap_params
from . import compute as _compute
from . import plot as _plot  # noqa: F401
from ._ospy_utils import (_check_instance, _rename_aliased,
                          _ax_warning, _check_options)

# Recommended dependencies (private)
try:
    import matplotlib.pyplot as _plt
    from matplotlib.animation import FuncAnimation as _FuncAnimation
except ImportError:  # pragma: no cover
    pass

try:
    from IPython.utils import io as _io
    from IPython.display import HTML as _HTML
    from IPython.display import display as _display
except ImportError:  # pragma: no cover
    pass

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    pass


def _create_animation(od, time, plot_func, func_kwargs, display,
                      **kwargs):
    """
    Create animation using oceanspy plot functions.

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to plot.
    time: DataArray
        DataArray corresponding to time.
    plot_func: function
        Alias referring to the plot function.
    func_kwargs:
        Keyword arguments for plot_func.
    display: bool
        If True, display the animation.
    **kwargs:
        Keyword arguments for py:class:`matplotlib.animation.FuncAnimation`

    Returns
    -------
    anim: matplotlib.animation.FuncAnimation
        Animation object

    References
    ----------
    https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html
    """

    # Check parameters
    _check_instance({'od': od,
                     'time': time,
                     'func_kwargs': func_kwargs,
                     'display': display},
                    {'od': 'oceanspy.OceanDataset',
                     'time': 'xarray.DataArray',
                     'func_kwargs': ['type(None)', 'dict'],
                     'display': ['bool']})

    # Handle kwargs
    if func_kwargs is None:
        func_kwargs = {}

    # Animate function
    def animate(i):
        _plt.clf()
        func_kwargs['cutout_kwargs'] = {'timeRange':
                                        time.isel({time.dims[0]: i}).values,
                                        'dropAxes': 'time'}
        with _io.capture_output() as captured:
            plot_func(od, **func_kwargs)
        if 'pbar' in locals():
            pbar.update(1)

    # Create animation object
    anim = _FuncAnimation(**{'fig': _plt.gcf(),
                             'func': animate,
                             'frames': len(time),
                             **kwargs})

    # Display
    if display is True:
        pbar = _tqdm(total=len(time))
        _display(_HTML(anim.to_html5_video()))
        pbar.close()
        del pbar

    return anim


def vertical_section(od, display=True, FuncAnimation_kwargs=None,
                     **kwargs):
    """
    Animate vertical section plots.

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to plot.
    display: bool
        If True, display the animation.
    FuncAnimation_kwargs: dict
        Keyword arguments from :py:func:`matplotlib.animation.FuncAnimation`
    **kwargs:
        Keyword arguments from :py:func:`oceanspy.plot.vertical_section`

    Returns
    -------
    anim: matplotlib.animation.FuncAnimation
        Animation object

    See also
    --------
    oceanspy.plot.vertical_section
    """

    # Check parameters
    _check_instance({'od': od,
                     'display': display,
                     'FuncAnimation_kwargs': FuncAnimation_kwargs},
                    {'od': 'oceanspy.OceanDataset',
                     'display': 'bool',
                     'FuncAnimation_kwargs': ['type(None)', 'dict']})

    # Handle kwargs
    if FuncAnimation_kwargs is None:
        FuncAnimation_kwargs = {}

    # Name of the plot_functions
    plot_func = eval('_plot.vertical_section')

    # First cutout and get time
    subsamp_kwargs = kwargs.pop('subsamp_kwargs', None)
    subsampMethod = kwargs.pop('subsampMethod', None)

    if subsampMethod is not None:
        # Check plot
        _check_options(name='subsampMethod',
                       selected=subsampMethod,
                       options=['mooring_array', 'survey_stations'])

    if subsamp_kwargs is not None:
        # Subsample first
        if subsampMethod == 'mooring_array':
            od = od.subsample.mooring_array(**subsamp_kwargs)
        else:
            # 'survey_stations'
            od = od.subsample.survey_stations(**subsamp_kwargs)

    time = od._ds['time']

    # Fix colorbar
    varName = kwargs.pop('varName', None)

    # Add missing variables (use private)
    _varName = _rename_aliased(od, varName)
    od = _compute._add_missing_variables(od, _varName)

    # Extract color (use public)
    color = od.dataset[varName]

    # Create colorbar (stolen from xarray)
    cmap_kwargs = {}
    for par in ['vmin', 'vmax', 'cmap', 'center',
                'robust', 'extend', 'levels', 'filled', 'norm']:
        cmap_kwargs[par] = kwargs.pop(par, None)
    cmap_kwargs['plot_data'] = color.values
    cmap_kwargs = _xr.plot.utils._determine_cmap_params(**cmap_kwargs)
    kwargs = {'varName': varName, **kwargs, **cmap_kwargs}

    # Remove ax
    _ax_warning(kwargs)

    # Animation
    anim = _create_animation(od=od,
                             time=time,
                             plot_func=plot_func,
                             func_kwargs=kwargs,
                             display=display,
                             **FuncAnimation_kwargs)
    return anim


def horizontal_section(od, display=True, FuncAnimation_kwargs=None,
                       **kwargs):
    """
    Animate horizontal section plots.

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to plot.
    display: bool
        If True, display the animation.
    FuncAnimation_kwargs: dict
        Keyword arguments from :py:func:`matplotlib.animation.FuncAnimation`
    **kwargs:
        Keyword arguments from :py:func:`oceanspy.plot.horizontal_section`

    Returns
    -------
    anim: matplotlib.animation.FuncAnimation
        Animation object

    See also
    --------
    oceanspy.plot.horizontal_section
    """

    # Check parameters
    _check_instance({'od': od,
                     'display': display,
                     'FuncAnimation_kwargs': FuncAnimation_kwargs},
                    {'od': 'oceanspy.OceanDataset',
                     'display': 'bool',
                     'FuncAnimation_kwargs': ['type(None)', 'dict']})

    # Handle kwargs
    if FuncAnimation_kwargs is None:
        FuncAnimation_kwargs = {}

    # Name of the plot_functions
    plot_func = eval('_plot.horizontal_section')

    # First cutout and get time
    cutout_kwargs = kwargs.pop('cutout_kwargs', None)
    if cutout_kwargs is not None:
        od = od.subsample.cutout(**cutout_kwargs)
    time = od._ds['time']

    #  colorbar
    varName = kwargs.pop('varName', None)

    # Add missing variables (use private)
    _varName = _rename_aliased(od, varName)
    od = _compute._add_missing_variables(od, _varName)

    # Extract color (use public)
    color = od.dataset[varName]

    # Create colorbar (stolen from xarray)
    cmap_kwargs = {}
    for par in ['vmin', 'vmax', 'cmap',
                'center', 'robust', 'extend',
                'levels', 'filled', 'norm']:
        cmap_kwargs[par] = kwargs.pop(par, None)
    cmap_kwargs['plot_data'] = color.values
    cmap_kwargs = _xr.plot.utils._determine_cmap_params(**cmap_kwargs)
    kwargs = {'varName': varName, **kwargs, **cmap_kwargs}

    # Remove ax
    _ax_warning(kwargs)

    # Animation
    anim = _create_animation(od=od,
                             time=time,
                             plot_func=plot_func,
                             func_kwargs=kwargs,
                             display=display,
                             **FuncAnimation_kwargs)
    return anim


def TS_diagram(od, display=True, FuncAnimation_kwargs=None,
               **kwargs):
    """
    Animate TS diagrams.

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to plot.
    display: bool
        If True, display the animation.
    FuncAnimation_kwargs: dict
        Keyword arguments from :py:func:`matplotlib.animation.FuncAnimation`
    **kwargs:
        Keyword arguments from :py:func:`oceanspy.plot.TS_diagram`

    Returns
    -------
    anim: matplotlib.animation.FuncAnimation
        Animation object

    See also
    --------
    oceanspy.plot.TS_diagram
    """

    # Check parameters
    _check_instance({'od': od,
                     'display': display,
                     'FuncAnimation_kwargs': FuncAnimation_kwargs},
                    {'od': 'oceanspy.OceanDataset',
                     'display': 'bool',
                     'FuncAnimation_kwargs': ['type(None)', 'dict']})

    # Handle kwargs
    if FuncAnimation_kwargs is None:
        FuncAnimation_kwargs = {}

    # Name of the plot_functions
    plot_func = eval('_plot.TS_diagram')

    # First cutout and get time
    cutout_kwargs = kwargs.pop('cutout_kwargs', None)
    if cutout_kwargs is not None:
        od = od.subsample.cutout(**cutout_kwargs)
    time = od._ds['time']

    # Check Temp and S
    varList = ['Temp', 'S']
    od = _compute._add_missing_variables(od, varList)

    # Fix T and S axes
    Tlim = kwargs.pop('Tlim', None)
    Slim = kwargs.pop('Slim', None)
    if Tlim is None:
        cmap_params = _determine_cmap_params(od._ds['Temp'].values,
                                             center=False)
        Tlim = [cmap_params['vmin'], cmap_params['vmax']]
    if Slim is None:
        cmap_params = _determine_cmap_params(od._ds['S'].values,
                                             center=False)
        Slim = [cmap_params['vmin'], cmap_params['vmax']]
    kwargs['Tlim'] = Tlim
    kwargs['Slim'] = Slim

    # Fix density
    dens = kwargs.pop('dens', None)
    if dens is None:
        t, s = _xr.broadcast(_xr.DataArray(_np.linspace(Tlim[0],
                                                        Tlim[-1],
                                                        100), dims=('t')),
                             _xr.DataArray(_np.linspace(Slim[0],
                                                        Slim[-1],
                                                        100), dims=('s')))
        odSigma0 = _ospy.OceanDataset(_xr.Dataset({'Temp': t,
                                                   'S': s}))
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
        dens = odSigma0._ds['Sigma0']
        dens = dens.where(odSigma0._ds['Temp'] > freez_point)
    kwargs['dens'] = dens

    # Fix colorbar
    colorName = kwargs.pop('colorName', None)
    if colorName is not None:

        # Add missing variables (use private)
        _colorName = _rename_aliased(od, colorName)
        od = _compute._add_missing_variables(od, _colorName)

        # Extract color (use public)
        color = od.dataset[colorName]

        # Create colorbar (stolen from xarray)
        cmap_kwargs = kwargs.pop('cmap_kwargs', None)
        if cmap_kwargs is None:
            cmap_kwargs = {}
        cmap_kwargs['plot_data'] = color.values
        kwargs['cmap_kwargs'] = _determine_cmap_params(**cmap_kwargs)
    kwargs['colorName'] = colorName

    # Remove ax
    _ax_warning(kwargs)

    # Animation
    anim = _create_animation(od=od,
                             time=time,
                             plot_func=plot_func,
                             func_kwargs=kwargs,
                             display=display,
                             **FuncAnimation_kwargs)
    return anim


class _animateMethods(object):
    """
    Enables use of functions as OceanDataset attributes.
    """
    def __init__(self, od):
        self._od = od

    @_functools.wraps(TS_diagram)
    def TS_diagram(self, **kwargs):
        return TS_diagram(self._od, **kwargs)

    @_functools.wraps(horizontal_section)
    def horizontal_section(self, **kwargs):
        return horizontal_section(self._od, **kwargs)

    @_functools.wraps(vertical_section)
    def vertical_section(self, **kwargs):
        return vertical_section(self._od, **kwargs)
