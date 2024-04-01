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

import functools as _functools
import warnings as _warnings

import numpy as _np
import pandas as _pd
import xarray as _xr
import xoak as _xoak
from xarray import DataArray

import oceanspy as _ospy

# From oceanspy (private)
from . import compute as _compute
from ._ospy_utils import (
    _check_instance,
    _check_mean_and_int_axes,
    _check_options,
    _rename_aliased,
)
from .compute import _add_missing_variables
from .compute import integral as _integral
from .compute import weighted_mean as _weighted_mean
from .llc_rearrange import Dims, face_direction, fill_path, splitter
from .utils import circle_path_array, connector

# Required dependencies (private)


# Additional dependencies (private)
try:
    import matplotlib.pyplot as _plt
except ImportError:  # pragma: no cover
    pass
try:
    import cartopy.crs as _ccrs
except ImportError:  # pragma: no cover
    pass


def TS_diagram(
    od,
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
    **kwargs,
):
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
    _check_instance(
        {
            "od": od,
            "colorName": colorName,
            "plotFreez": plotFreez,
            "ax": ax,
            "cmap_kwargs": cmap_kwargs,
            "contour_kwargs": contour_kwargs,
            "clabel_kwargs": clabel_kwargs,
            "cutout_kwargs": cutout_kwargs,
            "dens": dens,
        },
        {
            "od": "oceanspy.OceanDataset",
            "colorName": ["type(None)", "str"],
            "plotFreez": "bool",
            "ax": ["type(None)", "matplotlib.pyplot.Axes"],
            "cmap_kwargs": ["type(None)", "dict"],
            "contour_kwargs": ["type(None)", "dict"],
            "clabel_kwargs": ["type(None)", "dict"],
            "cutout_kwargs": ["type(None)", "dict"],
            "dens": ["type(None)", "xarray.DataArray"],
        },
    )

    if Tlim is not None:
        Tlim = _np.asarray(Tlim)
        if Tlim.size != 2:
            raise ValueError("`Tlim` must contain 2 elements")
        Tlim = Tlim.reshape(2)

    if Slim is not None:
        Slim = _np.asarray(Slim)
        if Slim.size != 2:
            raise ValueError("`Slim` must contain 2 elements")
        Slim = Slim.reshape(2)

    if dens is not None and not set(["Temp", "S"]).issubset(dens.coords):
        raise ValueError("`dens` must have coordinates (Temp, S)")

    # Change None in empty dict
    if cmap_kwargs is None:
        cmap_kwargs = {}
    cmap_kwargs = dict(cmap_kwargs)
    if contour_kwargs is None:
        contour_kwargs = {}
    contour_kwargs = dict(contour_kwargs)
    if clabel_kwargs is None:
        clabel_kwargs = {}
    clabel_kwargs = dict(clabel_kwargs)
    if cutout_kwargs is None:
        cutout_kwargs = {}
    cutout_kwargs = dict(cutout_kwargs)

    # Cutout first
    if len(cutout_kwargs) != 0:
        od = od.subsample.cutout(**cutout_kwargs)

    # Check and extract T and S
    varList = ["Temp", "S"]
    od = _add_missing_variables(od, varList)

    # Compute mean
    if meanAxes is not None:
        mean_ds = _compute.weighted_mean(
            od,
            varNameList=["Temp", "S"],
            axesList=meanAxes,
            storeWeights=False,
            aliased=False,
        )
        T = mean_ds["w_mean_Temp"].rename("Temp")
        S = mean_ds["w_mean_S"].rename("S")
        lost_coords = list(set(od._ds["Temp"].dims) - set(T.coords))
    else:
        T = od._ds["Temp"]
        S = od._ds["S"]
        lost_coords = []

    # Extract color field, and interpolate if needed
    if colorName is not None:
        # Add missing variables (use private)
        _colorName = _rename_aliased(od, colorName)
        od = _add_missing_variables(od, _colorName)

        # Extract color (use public)
        color = od.dataset[colorName]
        if meanAxes is not None:
            mean_ds = _compute.weighted_mean(
                od,
                varNameList=_colorName,
                axesList=meanAxes,
                storeWeights=False,
                aliased=False,
            )
            color = mean_ds["w_mean_" + _colorName].rename(_colorName)
        else:
            color = od.dataset[colorName]
        grid = od.grid
        dims2interp = [dim for dim in color.dims if dim not in T.dims]

        # Interpolation
        for dim in dims2interp:
            for axis in od.grid.axes.keys():
                if dim in [
                    od.grid.axes[axis].coords[k]
                    for k in od.grid.axes[axis].coords.keys()
                ]:
                    print(
                        "Interpolating [{}] along [{}]-axis." "".format(colorName, axis)
                    )
                    attrs = color.attrs
                    color = grid.interp(
                        color, axis, to="center", boundary="fill", fill_value=_np.nan
                    )
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
        print("Isopycnals: ", end="")
        tlin = _xr.DataArray(_np.linspace(Tlim[0], Tlim[-1], 100), dims=("t"))
        slin = _xr.DataArray(_np.linspace(Slim[0], Slim[-1], 100), dims=("s"))
        t, s = _xr.broadcast(tlin, slin)
        odSigma0 = _ospy.OceanDataset(_xr.Dataset({"Temp": t, "S": s}))
        odSigma0 = odSigma0.set_parameters(od.parameters)
        odSigma0 = odSigma0.compute.potential_density_anomaly()
        odSigma0._ds = odSigma0._ds.set_coords(["Temp", "S"])

        # Freezing point
        paramsList = ["tempFrz0", "dTempFrz_dS"]
        params2use = {
            par: od.parameters[par] for par in od.parameters if par in paramsList
        }
        tempFrz0 = params2use["tempFrz0"]
        dTempFrz_dS = params2use["dTempFrz_dS"]
        freez_point = tempFrz0 + odSigma0._ds["S"] * dTempFrz_dS

        # Extract Density
        dens = odSigma0._ds["Sigma0"].where(odSigma0._ds["Temp"] > freez_point)

    # Create axis
    if ax is None:
        ax = _plt.gca()

    # Use plot if colorless (faster!), otherwise use scatter
    if colorName is None:
        default_kwargs = {"color": "k", "linestyle": "None", "marker": "."}
        kwargs = {**default_kwargs, **kwargs}
        ax.plot(S.values.flatten(), T.values.flatten(), **kwargs)
    else:
        # Mask points out of axes
        color = color.where(_np.logical_and(T > min(Tlim), T < max(Tlim)))
        color = color.where(_np.logical_and(S > min(Slim), T < max(Slim)))
        color = color.stack(all_dims=color.dims)
        c = color.values

        # Create colorbar (stolen from xarray)
        cmap_kwargs["plot_data"] = c
        cmap_params = _xr.plot.utils._determine_cmap_params(**cmap_kwargs)
        extend = cmap_params.pop("extend")
        _ = cmap_params.pop("levels")
        kwargs = {**cmap_params, **kwargs}
        # Scatter
        sc = ax.scatter(S.values.flatten(), T.values.flatten(), c=c, **kwargs)
        _plt.colorbar(sc, label=_xr.plot.utils.label_from_attrs(color), extend=extend)

    # Plot isopycnals
    t = dens["Temp"]
    s = dens["S"]
    col_keys = ["colors", "cmap"]
    default_contour_kwargs = {key: contour_kwargs.pop(key, None) for key in col_keys}
    if all(default_contour_kwargs[key] is None for key in col_keys):
        default_contour_kwargs["colors"] = "gray"
    contour_kwargs = {**default_contour_kwargs, **contour_kwargs}
    CS = ax.contour(s.values, t.values, dens.values, **contour_kwargs)
    ax.clabel(CS, **clabel_kwargs)

    # Plot freezing point
    if plotFreez:
        paramsList = ["tempFrz0", "dTempFrz_dS"]
        params2use = {
            par: od.parameters[par] for par in od.parameters if par in paramsList
        }
        tempFrz0 = params2use["tempFrz0"]
        dTempFrz_dS = params2use["dTempFrz_dS"]
        s = _np.unique(s.values.flatten())
        ax.plot(s, tempFrz0 + s * dTempFrz_dS, "b")

    # Set labels and limits
    ax.set_xlabel(_xr.plot.utils.label_from_attrs(S))
    ax.set_ylabel(_xr.plot.utils.label_from_attrs(T))
    ax.set_xlim(Slim)
    ax.set_ylim(Tlim)

    # Set title
    title = []
    all_coords = list(lost_coords) + list(T.coords)
    skip_coords = ["X", "Y", "Xp1", "Yp1"]
    if any([dim in od._ds.dims for dim in ["mooring", "station", "particle"]]):
        skip_coords = [coord for coord in od._ds.coords if "X" in coord or "Y" in coord]
    for coord in all_coords:
        if coord not in skip_coords:
            if coord in list(lost_coords):
                da = od._ds["Temp"]
                pref = "<"
                suf = ">"
            else:
                da = T
                pref = ""
                suf = ""
            rng = [da[coord].min().values, da[coord].max().values]
            units = da[coord].attrs.pop("units", "")
            if units.lower() == "none":
                units = ""
            if "time" in coord:
                for i, v in enumerate(rng):
                    ts = _pd.to_datetime(str(v))
                    rng[i] = ts.strftime("%Y-%m-%d %r")

            if rng[0] == rng[-1]:
                rng = "{}".format(rng[0])
            else:
                rng = "from {} to {}".format(rng[0], rng[1])
            title = title + ["{}{}{}: {} {}" "".format(pref, coord, suf, rng, units)]

    ax.set_title("\n".join(title))

    return ax


def time_series(
    od, varName, meanAxes=False, intAxes=False, cutout_kwargs=None, **kwargs
):
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
    _check_instance(
        {"od": od, "varName": varName, "cutout_kwargs": cutout_kwargs},
        {
            "od": "oceanspy.OceanDataset",
            "varName": "str",
            "cutout_kwargs": ["type(None)", "dict"],
        },
    )

    # Check mean and int axes
    meanAxes, intAxes = _check_mean_and_int_axes(
        od=od, meanAxes=meanAxes, intAxes=intAxes, exclude=["time"]
    )

    # Handle kwargs
    if cutout_kwargs is None:
        cutout_kwargs = {}
    cutout_kwargs = dict(cutout_kwargs)

    # Cutout first
    if len(cutout_kwargs) != 0:
        od = od.subsample.cutout(**cutout_kwargs)

    # Variable name
    _varName = _rename_aliased(od, varName)
    od = _add_missing_variables(od, _varName)

    # Mean and sum
    da, varName = _compute_mean_and_int(od, varName, meanAxes, intAxes)

    # Get time name
    time_name = [dim for dim in od.grid_coords["time"] if dim in da.dims]
    if len(time_name) != 1:
        raise ValueError("Couldn't find time dimension")
    else:
        time_name = time_name[0]

    # Check
    if len(da.shape) > 2:
        dims = list(da.dims)
        dims.remove(time_name)
        raise ValueError(
            "Timeseries containing multiple"
            " dimension other than time: {}".format(dims)
        )

    # Plot
    _ = da.plot.line(**{"x": time_name, **kwargs})
    _plt.tight_layout()

    return _plt.gca()


def horizontal_section(
    od,
    varName,
    plotType="pcolormesh",
    use_coords=True,
    contourName=None,
    meanAxes=False,
    intAxes=False,
    contour_kwargs=None,
    clabel_kwargs=None,
    cutout_kwargs=None,
    **kwargs,
):
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
    _check_instance(
        {
            "od": od,
            "varName": varName,
            "plotType": plotType,
            "use_coords": use_coords,
            "contourName": contourName,
            "contour_kwargs": contour_kwargs,
            "clabel_kwargs": clabel_kwargs,
            "cutout_kwargs": cutout_kwargs,
        },
        {
            "od": "oceanspy.OceanDataset",
            "varName": "str",
            "plotType": "str",
            "use_coords": "bool",
            "contourName": ["type(None)", "str"],
            "contour_kwargs": ["type(None)", "dict"],
            "clabel_kwargs": ["type(None)", "dict"],
            "cutout_kwargs": ["type(None)", "dict"],
        },
    )

    # Check oceandataset
    wrong_dims = ["mooring", "station", "particle"]
    if any([dim in od._ds.dims for dim in wrong_dims]):
        raise ValueError(
            "`plot.vertical_section` does not support"
            " `od` with the following dimensions: "
            "{}".format(wrong_dims)
        )

    # Check plot
    _check_options(
        name="plotType",
        selected=plotType,
        options=["contourf", "contour", "imshow", "pcolormesh"],
    )

    # Handle kwargs
    if contour_kwargs is None:
        contour_kwargs = {}
    contour_kwargs = dict(contour_kwargs)
    if clabel_kwargs is None:
        clabel_kwargs = {}
    clabel_kwargs = dict(clabel_kwargs)
    if cutout_kwargs is None:
        cutout_kwargs = {}
    cutout_kwargs = dict(cutout_kwargs)

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
    meanAxes, intAxes = _check_mean_and_int_axes(
        od=od, meanAxes=meanAxes, intAxes=intAxes, exclude=["X", "Y"]
    )

    # Apply mean and sum
    da, varName = _compute_mean_and_int(od, varName, meanAxes, intAxes)

    # SQUEEZE! Otherwise animation don't show up
    # because xarray makes a faceted plot
    da = da.squeeze()

    # Get dimension names
    X_name = [dim for dim in od.grid_coords["X"] if dim in da.dims][0]
    Y_name = [dim for dim in od.grid_coords["Y"] if dim in da.dims][0]

    # CONTOURNAME
    if contourName is not None:
        # Apply mean and sum
        da_contour, contourName = _compute_mean_and_int(
            od, contourName, meanAxes, intAxes
        )

        # SQUEEZE! Otherwise animation don't show up
        # because xarray makes a faceted plot
        da_contour = da_contour.squeeze()

        # Get dimension names
        X_name_cont = [dim for dim in od.grid_coords["X"] if dim in da_contour.dims][0]
        Y_name_cont = [dim for dim in od.grid_coords["Y"] if dim in da_contour.dims][0]

    # Get dimensions
    dims = list(da.dims)
    dims.remove(X_name)
    dims.remove(Y_name)

    # Use coordinates
    if use_coords:
        al_dim = {}
        for dim in ["X", "Y", "Xp1", "Yp1"]:
            al_dim[dim] = _rename_aliased(od, varNameList=dim)

        if X_name == al_dim["X"] and Y_name == al_dim["Y"]:
            point = "C"
        elif X_name == al_dim["Xp1"] and Y_name == al_dim["Y"]:
            point = "U"
        elif X_name == al_dim["X"] and Y_name == al_dim["Yp1"]:
            point = "V"
        else:
            point = "G"
        X_name = _rename_aliased(od, varNameList="X" + point)
        Y_name = _rename_aliased(od, varNameList="Y" + point)

        if contourName is not None:
            if all([X_name_cont == al_dim["X"], Y_name_cont == al_dim["Y"]]):
                point_cont = "C"
            elif all([X_name_cont == al_dim["Xp1"], Y_name_cont == al_dim["Y"]]):
                point_cont = "U"
            elif all([X_name_cont == al_dim["X"], Y_name_cont == al_dim["Yp1"]]):
                point_cont = "V"
            else:
                point_cont = "G"
            X_name_cont = _rename_aliased(od, varNameList="X" + point_cont)
            Y_name_cont = _rename_aliased(od, varNameList="Y" + point_cont)

    # Pop from kwargs
    ax = kwargs.pop("ax", None)
    col = kwargs.pop("col", None)
    col_wrap = kwargs.pop("col_wrap", None)
    subplot_kws = kwargs.pop("subplot_kws", None)
    transform = kwargs.pop("transform", None)
    xstep, ystep = kwargs.pop("xstep", None), kwargs.pop("ystep", None)

    DIMS = [dim for dim in da.dims if dim[0] in ["X", "Y"]]
    dims_var = Dims(DIMS[::-1])

    if xstep is not None and ystep is not None:
        xslice = slice(0, len(da[dims_var.X]), xstep)
        yslice = slice(0, len(da[dims_var.Y]), ystep)
    else:
        xslice = slice(0, len(da[dims_var.X]))
        yslice = slice(0, len(da[dims_var.Y]))

    sargs = {dims_var.X: xslice, dims_var.Y: yslice}
    da = da.isel(**sargs)

    # Projection
    if ax is None:
        if subplot_kws is None:
            subplot_kws = dict(projection=od.projection)
        elif "projection" not in subplot_kws.keys():
            subplot_kws["projection"] = od.projection
    elif ax and od.projection is not None and not hasattr(ax, "projection"):
        od = od.set_projection(None)
        _warnings.warn(
            "\nSwitching projection off."
            "If `ax` is passed, it needs"
            " to be initialiazed with a projection.",
            stacklevel=2,
        )
    kwargs["ax"] = ax
    kwargs["subplot_kws"] = subplot_kws

    # Multiple plots:
    if len(dims) == 1:
        extra_name = dims[0]

        # TODO: For some reason, faceting and cartopy are not
        #       working very nice with our configurations
        #       Drop it for now, but we need to explore it more
        sbp_kws_proj = kwargs["subplot_kws"].pop("projection", None)
        if od.projection is not None or sbp_kws_proj is not None:
            _warnings.warn(
                "\nSwitch projection off."
                " This function currently"
                " does not support faceting for projected plots.",
                stacklevel=2,
            )
            od = od.set_projection(None)
            transform = None
            sbp_kws_proj = None
        kwargs["subplot_kws"]["projection"] = sbp_kws_proj

        # Add col
        if col is None:
            col = extra_name
        kwargs["col"] = col
        kwargs["col_wrap"] = col_wrap

    elif len(dims) != 0:
        raise ValueError(
            "There are too many dimensions: {}."
            "A maximum of 3 dimensions (including time)"
            " are supported."
            "Reduce the number of dimensions using"
            "`meanAxes` and/or `intAxes`".format(dims)
        )

    # Add transform
    if transform is None and od.projection is not None:
        kwargs["transform"] = _ccrs.PlateCarree()

    # Plot
    args = {"x": X_name, "y": Y_name, **kwargs}
    plotfunc = eval("_xr.plot." + plotType)
    p = plotfunc(da, **args)

    # Contour
    if contourName is not None:
        ax = args.pop("ax", None)
        transform = args.pop("transform", None)
        col_keys = ["colors", "cmap"]
        default_contour_kwargs = {
            key: contour_kwargs.pop(key, None) for key in col_keys
        }
        if all(default_contour_kwargs[key] is None for key in col_keys):
            default_contour_kwargs["colors"] = "gray"
        contour_kwargs = {**default_contour_kwargs, **contour_kwargs}
        args = {
            "x": X_name_cont,
            "y": Y_name_cont,
            "ax": ax,
            "transform": transform,
            "add_labels": False,
            **contour_kwargs,
        }
        if len(dims) == 0:
            cont = da_contour.plot.contour(**args)
            _plt.clabel(cont, **clabel_kwargs)
        else:
            for i, thisax in enumerate(p.axes.flat):
                if extra_name in da_contour.dims:
                    da_contour_i = da_contour.isel({extra_name: i}).squeeze()
                else:
                    da_contour_i = da_contour
                cont = da_contour_i.plot.contour(**{**args, "ax": thisax})
                _plt.clabel(cont, **clabel_kwargs)

    # Labels and return
    add_labels = kwargs.pop("add_labels", True)
    if len(dims) == 0:
        ax = _plt.gca()
        if od.projection is None:
            _plt.tight_layout()
        else:
            if add_labels is not False:
                try:
                    gl = ax.gridlines(crs=transform, draw_labels=True)
                    gl.top_labels = False
                    gl.right_labels = False
                except TypeError:
                    # Gridlines don't work with all projections
                    pass
        return ax
    else:
        return p


def vertical_section(
    od,
    varName,
    plotType="pcolormesh",
    use_dist=True,
    subsampMethod=None,
    contourName=None,
    meanAxes=False,
    intAxes=False,
    contour_kwargs=None,
    clabel_kwargs=None,
    subsamp_kwargs=None,
    cutout_kwargs=None,
    **kwargs,
):
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
    _check_instance(
        {
            "od": od,
            "varName": varName,
            "plotType": plotType,
            "use_dist": use_dist,
            "subsampMethod": subsampMethod,
            "contourName": contourName,
            "contour_kwargs": contour_kwargs,
            "clabel_kwargs": clabel_kwargs,
            "subsamp_kwargs": subsamp_kwargs,
        },
        {
            "od": "oceanspy.OceanDataset",
            "varName": "str",
            "plotType": "str",
            "use_dist": "bool",
            "subsampMethod": ["type(None)", "str"],
            "contourName": ["type(None)", "str"],
            "contour_kwargs": ["type(None)", "dict"],
            "clabel_kwargs": ["type(None)", "dict"],
            "subsamp_kwargs": ["type(None)", "dict"],
        },
    )

    # Check plot
    _check_options(
        name="plotType",
        selected=plotType,
        options=["contourf", "contour", "imshow", "pcolormesh"],
    )

    # Check subsample
    if subsampMethod is not None:
        # Check plot
        _check_options(
            name="subsampMethod",
            selected=subsampMethod,
            options=["mooring_array", "survey_stations"],
        )

    # Handle kwargs
    if contour_kwargs is None:
        contour_kwargs = {}
    contour_kwargs = dict(contour_kwargs)
    if clabel_kwargs is None:
        clabel_kwargs = {}
    clabel_kwargs = dict(clabel_kwargs)
    if cutout_kwargs is None:
        cutout_kwargs = {}
    cutout_kwargs = dict(cutout_kwargs)

    # For animation purposes.
    if len(cutout_kwargs) != 0:
        od = od.subsample.cutout(**cutout_kwargs)

    # Subsample first
    if subsamp_kwargs is not None and subsampMethod is not None:
        if subsampMethod == "mooring_array":
            od = od.subsample.mooring_array(**subsamp_kwargs)
        else:
            # survey_stations
            od = od.subsample.survey_stations(**subsamp_kwargs)

    # Check oceandataset
    needed_dims = ["mooring", "station"]
    if not any([dim in od.grid_coords.keys() for dim in needed_dims]):
        raise ValueError(
            "`plot.vertical_section` only supports"
            " `od` with one of the following grid coordinates: "
            "{}".format(needed_dims)
        )

    # Check variables and add
    listName = [varName]
    if contourName is not None:
        listName = listName + [contourName]
    _listName = _rename_aliased(od, listName)
    od = _add_missing_variables(od, _listName)

    # Check mean and int axes
    meanAxes, intAxes = _check_mean_and_int_axes(
        od=od,
        meanAxes=meanAxes,
        intAxes=intAxes,
        exclude=["mooring", "station", "X", "Y", "Z"],
    )

    # Apply mean and sum
    da, varName = _compute_mean_and_int(od, varName, meanAxes, intAxes)

    # SQUEEZE! Otherwise animation don't show up
    # because xarray makes a faceted plot
    da = da.squeeze()
    da, hor_name = _Vsection_regrid(od, da, varName)
    ver_name = [dim for dim in od.grid_coords["Z"] if dim in da.dims][0]
    da = da.squeeze()

    # slicing along section
    step = kwargs.pop("step", None)
    DIMS = [dim for dim in da.dims]
    dims_var = Dims(DIMS[::-1])
    if step is not None and dims_var.X in ["mooring", "station"]:
        xslice = slice(0, len(da[dims_var.X]), step)
    else:
        xslice = slice(0, len(da[dims_var.X]))

    sargs = {dims_var.X: xslice}
    da = da.isel(**sargs)

    # CONTOURNAME
    if contourName is not None:
        # Apply mean and sum
        da_contour = od.dataset[contourName]
        da_contour, contourName = _compute_mean_and_int(
            od, contourName, meanAxes, intAxes
        )

        # SQUEEZE! Otherwise animation don't show up
        # because xarray makes a faceted plot
        da_contour = da_contour.squeeze()

        # Get dimension names
        da_contour, hor_name_cont = _Vsection_regrid(od, da_contour, contourName)
        ver_name_cont = [dim for dim in od.grid_coords["Z"] if dim in da_contour.dims]
        if len(ver_name_cont) != 1:
            raise ValueError("Couldn't find Z dimension of [{}]" "".format(contourName))
        else:
            ver_name_cont = ver_name_cont[0]
        da_contour = da_contour.squeeze()

    # Check dimensions
    dims = list(da.dims)
    dims.remove(hor_name)
    dims.remove(ver_name)

    # Use distances
    if use_dist:
        if hor_name + "_dist" in da.coords:
            hor_name = hor_name + "_dist"
            hor_name_cont = hor_name

    # Pop from kwargs
    ax = kwargs.pop("ax", None)
    col = kwargs.pop("col", None)

    if len(dims) == 0:
        # Single plot:
        # Add ax
        if ax is None:
            ax = _plt.axes()
        kwargs["ax"] = ax
    else:
        # Multiple plots:
        extra_name = dims[0]

        # Add col
        if col is None:
            col = extra_name
        kwargs["col"] = col

    # Plot
    args = {"x": hor_name, "y": ver_name, **kwargs}
    plotfunc = eval("_xr.plot." + plotType)
    p = plotfunc(da, **args)

    # Contour
    if contourName is not None:
        ax = args.pop("ax", None)
        col_keys = ["colors", "cmap"]
        default_contour_kwargs = {
            key: contour_kwargs.pop(key, None) for key in col_keys
        }
        if all(default_contour_kwargs[key] is None for key in col_keys):
            default_contour_kwargs["colors"] = "gray"
        contour_kwargs = {**default_contour_kwargs, **contour_kwargs}
        args = {
            "x": hor_name_cont,
            "y": ver_name_cont,
            "ax": ax,
            "add_labels": False,
            **contour_kwargs,
        }
        if ax is not None:
            cont = da_contour.plot.contour(**args)
            _plt.clabel(cont, **clabel_kwargs)
        else:
            for i, thisax in enumerate(p.axes.flat):
                if extra_name in da_contour.dims:
                    da_contour_i = da_contour.isel({extra_name: i}).squeeze()
                else:
                    da_contour_i = da_contour
                cont = da_contour_i.plot.contour(**{**args, "ax": thisax})
                _plt.clabel(cont, **clabel_kwargs)

    # Return
    if ax is not None:
        _plt.tight_layout()
        return ax
    else:
        return p


def faces_array(
    od,
    Ymoor,
    Xmoor,
    varName="Depth",
    xoak_index="scipy_kdtree",
    face2axis=None,
    **kwargs,
):
    """
    Plots a variable defined on the LLC4320 grid by joining and rotating
    each face next to each other on a flat plane. No projection is needed.

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to plot.

    Ymoor, Xmoor: array-like data.
        degrees lat and lons. equal length
    varName: str
        name of (2D) variable to plot. default = `Depth`
    xoak_index: str
        default `scipy_kdtree`.
    face2axis: dict, default=None
        {face_count: (face index, subplot_rows, subplot_cols)}.
        face_count = range(len(faces with data)).
        face_index: range(0, 12).

        face_count, face_index must be consistent with data. otherwise ValueError.

    kargs: matplotlib.pyplot

    Returns
    -------
    matplotlib.pyplot.axes or xarray.plot.FacetGrid

    See Also
    --------
    oceanspy.subsample.mooring_array

    """
    transpose = [k for k in range(7, 13)]
    data = od._ds[varName]  # assert 2D, otherwise pick z=0, time=0
    # =========================================================
    # repeated code from mooring array
    ds_grid = od._ds[["XC", "YC"]]  # center points
    R = od.parameters["rSphere"]
    face_connections = od.face_connections["face"]

    if R is not None:
        Ymoor, Xmoor = circle_path_array(Ymoor, Xmoor, R)
    for key, value in ds_grid.sizes.items():
        ds_grid["i" + f"{key}"] = DataArray(range(value), dims=key)
    if xoak_index not in _xoak.IndexRegistry():
        raise ValueError(
            "`xoak_index` [{}] is not supported."
            "\nAvailable options: {}"
            "".format(xoak_index, _xoak.IndexRegistry())
        )
    ds_grid.xoak.set_index(["XC", "YC"], xoak_index)
    cdata = {"XC": ("mooring", Xmoor), "YC": ("mooring", Ymoor)}
    ds_data = _xr.Dataset(cdata)  # mooring data

    # find nearest points to given data.
    nds = ds_grid.xoak.sel(XC=ds_data["XC"], YC=ds_data["YC"])
    iX, iY, iface = (nds[f"{i}"].data for i in ("X", "Y", "face"))
    _dat = nds.face.values
    ll = _np.where(abs(_np.diff(_dat)))[0]
    _faces = [_dat[i] for i in ll] + [_dat[-1]]
    print(_faces)
    Niter = len(_faces)
    Niter = len(set(_faces))
    # ===========================================================

    # ========================================================
    # initialize
    nrows, ncols = None, None
    if Niter == 1:
        inX, inY = connector(iX, iY)
        inface = [_faces[0]] * len(inX)
        nrows, ncols = 1, 1
        face2axis = {0: (_faces[0],) + (0, 0)}
    elif Niter == 2:
        nX0, nY0 = splitter(iX, iY, iface)
        inX, inY, inface = [], [], []
        # for ii in range(Niter): #
        for ii in range(
            len(_faces)
        ):  # Niter count total faces. _faces can have repeated faces
            nix, niy = fill_path(nX0, nY0, _faces, ii, face_connections)
            inface += [_faces[ii]] * len(list(nix))
            inX += list(nix)
            inY += list(niy)
        if face2axis is None:
            # no dict defined, figure out one from face topo
            # there are 2 posibilities.
            nrot = [k for k in set(_faces) if k in range(6)]
            if len(nrot) > 0:
                # 1) There is a non-rotated face. Use this as anchor
                i0 = _faces.index(nrot[0])  # pick one / inrot
                i1 = _faces.index(list(set(_faces) - set([_faces[i0]]))[0])
                # below, face direction from non-rot to rot
                fdir = face_direction(_faces[i0], _faces[i1], face_connections)
                if fdir in [0, 1]:
                    # horizontal orientation
                    nrows, ncols = 1, 2
                    if fdir == 0:  # to left
                        fi0, fi1 = (0, 1), (0, 0)
                    else:  # to right
                        fi0, fi1 = (0, 0), (0, 1)
                else:
                    # vertical orientation
                    nrows, ncols = 2, 1
                    if fdir == 2:
                        fi0, fi1 = (0, 0), (1, 0)
                    elif fdir == 3:
                        fi0, fi1 = (1, 0), (0, 0)
            else:
                # case 2) There is no non-rotated face
                # two posibilities: 1) no arctic, 2) arctic
                if _faces[0] != 6:
                    i0 = 0
                    i1 = 1
                else:
                    i1 = 0
                    i0 = 1
                fdir = face_direction(_faces[i0], _faces[i1], face_connections)
                if fdir in [0, 1]:
                    # vertical orientation
                    nrows, ncols = 2, 1
                    if fdir == 0:
                        # i0 on top
                        fi0, fi1 = (1, 0), (0, 0)
                    else:
                        # i0 on bottom
                        fi0, fi1 = (0, 0), (0, 1)
                else:
                    # horizontal orientation
                    nrows, ncols = 1, 2
                    if fdir == 2:
                        # i0 on left
                        fi0, fi1 = (0, 1), (0, 0)
                    else:
                        # i0 on right
                        fi0, fi1 = (0, 0), (0, 1)

            face2axis = {0: (_faces[i0],) + fi0, 1: (_faces[i1],) + fi1}
        print("face2axis: ", face2axis)

    elif Niter > 2:
        # it is possible to only have two faces but len(Niter)>2
        # for example re entry. Need to make it so that it
        # len(Niter)>2 means 3 or more faces.
        nX0, nY0 = splitter(iX, iY, iface)
        inX, inY, inface = [], [], []
        # for ii in range(Niter):
        for ii in range(len(_faces)):
            nix, niy = fill_path(nX0, nY0, _faces, ii, face_connections)
            inface += [_faces[ii]] * len(list(nix))
            inX += list(nix)
            inY += list(niy)

        if face2axis is None:
            nrows = 4
            ncols = 4
            # need to do this smartly.
            face2axis = {
                0: (3, 3, 0),
                1: (4, 2, 0),
                2: (5, 1, 0),
                3: (6, 0, 2),
                4: (7, 1, 1),
                5: (8, 2, 1),
                6: (9, 3, 1),
                7: (10, 1, 2),
                8: (11, 2, 2),
                9: (12, 3, 2),
                10: (None, 0, 0),
                11: (None, 0, 1),
                12: (None, 0, 3),
                13: (0, 3, 3),
                14: (1, 2, 3),
                15: (2, 1, 3),
                16: (None, 0, 3),
            }
            # todo: improve arctic face representation
            # to expand domain towards right.

    # ========================================================
    ndata = {"iX": ("X", inX), "iY": ("Y", inY), "iface": ("face", inface)}
    nds = _xr.Dataset(ndata)  # mooring data

    spkwargs = {}  # dict for initializing subplots

    gridspec_kw = dict(left=0, bottom=0, right=1, top=1)

    params = ["wspace", "hspace"]
    for param in params:
        if param in kwargs.keys():
            gridspec_kw[param] = kwargs.pop(param)
        else:
            gridspec_kw[param] = 0.001
    spkwargs["gridspec_kw"] = gridspec_kw
    if "figsize" in kwargs.keys():
        spkwargs["figsize"] = kwargs.pop("figsize")  # should I pop it?

    if nrows is None and ncols is None:
        nrows, ncols = 0, 0
        for i in face2axis.keys():
            nrows = max(nrows, face2axis[i][1])
            ncols = max(ncols, face2axis[i][2])
        nrows += 1
        ncols += 1

    fig, axes = _plt.subplots(nrows=nrows, ncols=ncols, **spkwargs)

    plt_params = ["ls", "color", "marker", "markersize", "alpha"]
    plt_preset = ["", "#FF8000", ".", "2", 1]
    pkwargs = {}  # plot kwargs

    for param, val in zip(plt_params, plt_preset):
        if param in kwargs.keys():
            pkwargs[param] = kwargs.pop(param, None)
        else:
            pkwargs[param] = val

    kw_params = ["vmin", "vmax", "cmap", "levels"]
    kw_preset = [0, 1000, "Greys_r", 10]
    for pm, val in zip(kw_params, kw_preset):
        if pm not in kwargs.keys():
            kwargs[pm] = val
    kwargs["add_colorbar"] = False  # fixed

    for count, (face, j, i) in face2axis.items():
        kwargs["xincrease"] = True
        kwargs["yincrease"] = True
        if ncols * nrows == 1:
            ax = axes
        elif nrows > 1 and ncols == 1:
            ax = axes[j]
        elif nrows == 1 and ncols > 1:
            # vertical, allow arctic
            ax = axes[i]
        elif nrows > 1 and ncols > 1:
            ax = axes[j, i]
        # here the plotting begins.
        if face is None:
            ax.axis("off")
        else:
            data_ax = data.isel(face=face)

            # 2d plotting / contourf
            if face == 6:
                # faces present that connect with arctic
                _exch_fs = len(set([2, 5, 7, 10]) & set(_faces))
                if Niter >= 2:
                    if 2 in _faces and _exch_fs == 1:
                        kwargs.pop("xincrease")
                        data_ax, kwargs["xincrease"] = data_ax.transpose(), False
                    elif 5 in _faces and _exch_fs == 1:
                        pass
                    elif 7 in _faces and _exch_fs == 1:
                        kwargs.pop("yincrease")
                        data_ax, kwargs["yincrease"] = data_ax.transpose(), False
                    elif 10 in _faces and _exch_fs == 1:
                        [kwargs.pop(var) for var in ["xincrease", "yincrease"]]
                        kwargs["xincrease"], kwargs["yincrease"] = False, False
                    else:
                        # face = 6 atop face = 7
                        [kwargs.pop(var) for var in ["xincrease", "yincrease"]]
                        kwargs["xincrease"], kwargs["yincrease"] = False, False
            if face in transpose:
                data_ax = data_ax.transpose()
                kwargs["yincrease"] = False
            data_ax.plot(ax=ax, **kwargs)

            # line plotting of arrays
            if face in nds.iface.values:
                ind = _np.argwhere(nds["iface"].data == face)
                xvals, yvals = nds["iX"].values[ind], nds["iY"].values[ind]
                if face == 6 and Niter == 1:
                    # face 6 atop face 7
                    xvals, yvals = xvals[::-1], yvals[::-1]
                elif face == 6 and Niter >= 2:
                    if 2 in _faces and _exch_fs == 1:
                        xvals, yvals = yvals, xvals
                    elif 5 in _faces and _exch_fs == 1:
                        pass
                    elif 7 in _faces and _exch_fs == 1:
                        xvals, yvals = yvals, xvals
                    elif 10 in _faces and _exch_fs == 1:
                        xvals, yvals = xvals[::-1], yvals[::-1]
                    else:
                        # more that two faces that exchange with arctic
                        # are present in the array
                        print("Warning - 2 or more faces exchanging with arctic")
                        print("is not yet supported. if you would like to")
                        print(" contribute to this - please raise an issue")
                elif face in transpose:
                    xvals, yvals = yvals, xvals
                ax.plot(xvals, yvals, **pkwargs)
            ax.axis("off")
            ax.set_title("")
    return axes


def _compute_mean_and_int(od, varName, meanAxes, intAxes):
    # Mean and sum
    if meanAxes is not False:
        ds = _weighted_mean(
            od, varNameList=[varName], axesList=meanAxes, storeWeights=False
        )
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
    if "mooring" in od.grid_coords:
        # Time coordinates
        if "time" in od.grid_coords.keys():
            time_coords = {
                timeName: da[timeName]
                for timeName in od.grid_coords["time"].keys()
                if timeName in da.coords
            }
        else:
            time_coords = {}

        # Regrid to center dim
        for axis in ["X", "Y"]:
            dim2regrid = [
                dim
                for dim in od.grid_coords[axis]
                if (od.grid_coords[axis][dim] is not None and dim in da.dims)
            ]
            if len(dim2regrid) != 0:
                print("Regridding [{}] along [{}]-axis." "".format(varName, axis))
                da_attrs = da.attrs
                da = od.grid.interp(da, axis)
                da.attrs = da_attrs
            hor_name = [dim for dim in od.grid_coords["mooring"] if dim in da.dims]
            if len(hor_name) != 1:
                raise ValueError(
                    "Couldn't find `mooring` dimension of [{}]" "".format(varName)
                )
            else:
                hor_name = hor_name[0]
            da = da.assign_coords(**time_coords)
            if hor_name + "_dist" in od._ds.coords:
                da = da.assign_coords(
                    **{hor_name + "_dist": od._ds[hor_name + "_dist"]}
                )
            for toRem in ["X", "Y", "Xp1", "Yp1"]:
                toRem = _rename_aliased(od, varNameList=toRem)
                if toRem in da.coords:
                    da = da.drop_vars(toRem)
    else:
        # Station
        hor_name = [dim for dim in od.grid_coords["station"] if dim in da.dims]
        if len(hor_name) != 1:
            raise ValueError(
                "Couldn't find `station` dimension of [{}]" "".format(varName)
            )
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

    @_functools.wraps(faces_array)
    def faces_array(self, **kwargs):
        return faces_array(self._od, **kwargs)
