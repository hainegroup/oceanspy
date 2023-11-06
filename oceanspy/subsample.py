"""
Subsample OceanDataset objects.
"""

# Instructions for developers:
# 1. All funcions must return a OceanDataset.
# 2. All functions must operate on private objects of an od (_ds, _grid),
#    and use OceanSpy reference names.
# 3. All functions should use the cutout_kwargs argument at the beginning.
# 4. Preserve original grid structure if possible.
# 5. Add new functions to _subsampleMethods
# 6. Add new functions to docs/api.rst

import copy as _copy
import functools as _functools
import warnings as _warnings

# import dask
import numpy as _np
import pandas as _pd

# Required dependencies (private)
import xarray as _xr
from packaging.version import parse as _parse_version
from xarray import DataArray

# From OceanSpy (private)
from . import compute as _compute
from . import utils as _utils
from ._ospy_utils import (
    _check_instance,
    _check_list_of_string,
    _check_native_grid,
    _check_part_position,
    _check_range,
    _rename_aliased,
)
from .llc_rearrange import LLCtransformation as _llc_trans
from .llc_rearrange import (
    connector,
    cross_face_diffs,
    eval_dataset,
    fill_path,
    flip_v,
    mates,
    mooring_singleface,
    splitter,
    station_singleface,
)
from .utils import (
    _rel_lon,
    _reset_range,
    circle_path_array,
    diff_and_inds_where_insert,
    get_maskH,
    remove_repeated,
    reset_dim,
)

# Recommended dependencies (private)
try:
    from geopy.distance import great_circle as _great_circle
except ImportError:  # pragma: no cover
    pass
try:
    import xesmf as _xe
except ImportError:  # pragma: no cover
    pass
try:
    import xoak as _xoak
except ImportError:  # pragma: no cover
    pass


def cutout(
    od,
    varList=None,
    YRange=None,
    XRange=None,
    add_Hbdr=False,
    mask_outside=False,
    ZRange=None,
    add_Vbdr=False,
    timeRange=None,
    timeFreq=None,
    sampMethod="snapshot",
    dropAxes=False,
    centered=None,
    persist=False,
):
    """
    Cutout the original dataset in space and time
    preserving the original grid structure.

    Parameters
    ----------
    od: OceanDataset
        oceandataset to subsample
    varList: 1D array_like, str, or None
        List of variables (strings).
    YRange: 1D array_like, scalar, or None
        Y axis limits (e.g., latitudes).
        If len(YRange)>2, max and min values are used.
    XRange: 1D array_like, scalar, or None
        X axis limits (e.g., longitudes).
        If len(XRange)>2, max and min values are used.
    add_Hbdr: bool, scal
        If scalar, add and subtract `add_Hbdr` to the the horizontal range.
        of the horizontal ranges.
        If True, automatically estimate add_Hbdr.
        If False, add_Hbdr is set to zero.
    mask_outside: bool
        If True, set all values in areas outside specified (Y,X)ranges to NaNs.
        (Useful for curvilinear grids).
    ZRange: 1D array_like, scalar, or None
        Z axis limits.
        If len(ZRange)>2, max and min values are used.
    add_Vbdr: bool, scal
        If scalar, add and subtract `add_Vbdr` to the the vertical range.
        If True, automatically estimate add_Vbdr.
        If False, add_Vbdr is set to zero.
    timeRange: 1D array_like, numpy.ScalarType, or None
        time axis limits.
        If len(timeRange)>2, max and min values are used.
    timeFreq: str or None
        Time frequency.
        Available optionts are pandas Offset Aliases (e.g., '6H'):
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    sampMethod: {'snapshot', 'mean'}
        Downsampling method (only if timeFreq is not None).
        "snapshot" means just throw away everything in between.
        "mean" means take a running average with the time freq.
    dropAxes: 1D array_like, str, or bool
        List of axes to remove from Grid object.
        if one point only is in the range.
        If True, set dropAxes=od.grid_coords.
        If False, preserve original grid.
    centered: str or bool.
        Only used when `face` is a dimension. When str, 'Atlantic' or 'Pacific'
        are the only possible choices. Default is `None` and centered is estimated
        during the cutout.
    persist: bool.
        Only used when `face` is a dimension. If `False` (default) the transformation
        is not persisted.

    Returns
    -------
    od: OceanDataset
        Subsampled oceandataset

    Notes
    -----
    If any of the horizontal ranges is not None,
    the horizontal dimensions of the cutout will have
    len(Xp1)>len(X) and len(Yp1)>len(Y)
    even if the original oceandataset had
    len(Xp1)==len(X) or len(Yp1)==len(Y).
    """

    # Checks
    unsupported_dims = ["mooring", "particle", "station"]
    check1 = XRange is not None or YRange is not None
    if check1 and any([dim in unsupported_dims for dim in od._ds.dims]):
        _warnings.warn(
            "\nHorizontal cutout not supported" "for moorings, surveys, and particles",
            stacklevel=2,
        )
        XRange = None
        YRange = None

    _check_instance(
        {
            "od": od,
            "add_Hbdr": add_Hbdr,
            "mask_outside": mask_outside,
            "timeFreq": timeFreq,
        },
        {
            "od": "oceanspy.OceanDataset",
            "add_Hbdr": "(float, int, bool)",
            "mask_outside": "bool",
            "timeFreq": ["type(None)", "str"],
        },
    )
    varList = _check_list_of_string(varList, "varList")
    YRange = _check_range(od, YRange, "YRange")
    XRange = _check_range(od, XRange, "XRange")
    ZRange = _check_range(od, ZRange, "ZRange")
    timeRange = _check_range(od, timeRange, "timeRange")
    sampMethod_list = ["snapshot", "mean"]

    if sampMethod not in sampMethod_list:
        raise ValueError(
            "`sampMethod` [{}] is not supported."
            "\nAvailable options: {}"
            "".format(sampMethod, sampMethod_list)
        )

    if not isinstance(dropAxes, bool):
        dropAxes = _check_list_of_string(dropAxes, "dropAxes")
        axes_warn = [axis for axis in dropAxes if axis not in od.grid_coords]
        if len(axes_warn) != 0:
            _warnings.warn(
                "\n{} are not axes of the oceandataset" "".format(axes_warn),
                stacklevel=2,
            )
            dropAxes = list(set(dropAxes) - set(axes_warn))
        dropAxes = {d: od.grid_coords[d] for d in dropAxes}
    elif dropAxes is True:
        dropAxes = od.grid_coords
        if YRange is None:
            dropAxes.pop("Y", None)
        if XRange is None:
            dropAxes.pop("X", None)
        if ZRange is None:
            dropAxes.pop("Z", None)
        if timeRange is None:
            dropAxes.pop("time", None)
    else:
        dropAxes = {}

    # Message
    print("Cutting out the oceandataset.")

    # Copy
    od = _copy.copy(od)

    # list for coord variables
    co_list = [var for var in od._ds.coords if var not in od._ds.dims]
    # Drop variables
    if varList is not None:
        # Make sure it's a list
        varList = _rename_aliased(od, list(varList) + co_list)

        # Compute missing variables
        od = _compute._add_missing_variables(od, varList)
        # Drop useless
        nvarlist = [v for v in od._ds.data_vars if v not in varList]
        od._ds = od._ds.drop_vars(nvarlist)
    else:  # this way, if applicable, llc_transf gets applied to all vars
        varList = [var for var in od._ds.reset_coords().data_vars]

    # Unpack
    ds = od._ds
    periodic = od.grid_periodic

    # ---------------------------
    # Time CUTOUT
    # ---------------------------
    # Initialize vertical mask
    maskT = _xr.ones_like(ds["time"]).astype("int")

    if timeRange is not None:
        # Use arrays
        timeRange = _np.asarray([_np.min(timeRange), _np.max(timeRange)]).astype(
            ds["time"].dtype
        )

        # Get the closest
        for i, time in enumerate(timeRange):
            if _np.issubdtype(ds["time"].dtype, _np.datetime64):
                diff = _np.fabs(ds["time"].astype("float64") - time.astype("float64"))
            else:
                diff = _np.fabs(ds["time"] - time)
            timeRange[i] = ds["time"].where(diff == diff.min(), drop=True).min().values
        maskT = maskT.where(
            _np.logical_and(ds["time"] >= timeRange[0], ds["time"] <= timeRange[-1]), 0
        )

        # Find time indexes
        maskT = maskT.assign_coords(time=_np.arange(len(maskT["time"])))
        dmaskT = maskT.where(maskT.compute(), drop=True)
        dtime = dmaskT["time"].values
        iT = [min(dtime), max(dtime)]
        maskT["time"] = ds["time"]

        # Indexis
        if iT[0] == iT[1]:
            if "time" not in dropAxes:
                if iT[0] > 0:
                    iT[0] = iT[0] - 1
                else:
                    iT[1] = iT[1] + 1
        else:
            dropAxes.pop("time", None)

        # Cutout
        ds = ds.isel(time=slice(iT[0], iT[1] + 1))
        if "time_midp" in ds.dims:
            if "time" in dropAxes:
                if iT[0] == len(ds["time_midp"]):
                    iT[0] = iT[0] - 1
                    iT[1] = iT[1] - 1
                ds = ds.isel(time_midp=slice(iT[0], iT[1] + 1))
            else:
                ds = ds.isel(time_midp=slice(iT[0], iT[1]))

    # ---------------------------
    # Vertical CUTOUT
    # ---------------------------
    # Initialize vertical mask
    maskV = _xr.ones_like(ds["Zp1"])

    if add_Vbdr is True:
        add_Vbdr = _np.fabs(od._ds["Zp1"].diff("Zp1")).max().values
    elif add_Vbdr is False:
        add_Vbdr = 0

    if ZRange is not None:
        # Use arrays
        ZRange = _np.asarray([_np.min(ZRange) - add_Vbdr, _np.max(ZRange) + add_Vbdr])
        ZRange = ZRange.astype(ds["Zp1"].dtype)

        # Get the closest
        for i, Z in enumerate(ZRange):
            diff = _np.fabs(ds["Zp1"] - Z)
            ZRange[i] = ds["Zp1"].where(diff == diff.min()).min().values
        maskV = maskV.where(
            _np.logical_and(ds["Zp1"] >= ZRange[0], ds["Zp1"] <= ZRange[-1]), 0
        )

        # Find vertical indexes
        maskV = maskV.assign_coords(Zp1=_np.arange(len(maskV["Zp1"])))
        dmaskV = maskV.where(maskV.compute(), drop=True)
        dZp1 = dmaskV["Zp1"].values
        iZ = [_np.min(dZp1), _np.max(dZp1)]
        maskV["Zp1"] = ds["Zp1"]

        # Indexis
        if iZ[0] == iZ[1]:
            if "Z" not in dropAxes:
                if iZ[0] > 0:
                    iZ[0] = iZ[0] - 1
                else:
                    iZ[1] = iZ[1] + 1
        else:
            dropAxes.pop("Z", None)

        # Cutout
        ds = ds.isel(Zp1=slice(iZ[0], iZ[1] + 1))
        if "Z" in dropAxes:
            if iZ[0] == len(ds["Z"]):
                iZ[0] = iZ[0] - 1
                iZ[1] = iZ[1] - 1
            ds = ds.isel(Z=slice(iZ[0], iZ[1] + 1))
        else:
            ds = ds.isel(Z=slice(iZ[0], iZ[1]))

        if len(ds["Zp1"]) == 1:
            if "Zu" in ds.dims and len(ds["Zu"]) > 1:
                ds = ds.sel(Zu=ds["Zp1"].values, method="nearest")
            if "Zl" in ds.dims and len(ds["Zl"]) > 1:
                ds = ds.sel(Zl=ds["Zp1"].values, method="nearest")
        else:
            if "Zu" in ds.dims and len(ds["Zu"]) > 1:
                ds = ds.isel(Zu=slice(iZ[0], iZ[1]))
            if "Zl" in ds.dims and len(ds["Zl"]) > 1:
                ds = ds.isel(Zl=slice(iZ[0], iZ[1]))

    # ---------------------------
    # Horizontal CUTOUT (part I, split into two to avoid repeated code)
    # ---------------------------
    if add_Hbdr is True:
        add_Hbdr = _np.mean(
            [
                _np.fabs(od._ds["XG"].max() - od._ds["XG"].min()),
                _np.fabs(od._ds["YG"].max() - od._ds["YG"].min()),
            ]
        )
        add_Hbdr = 1.5 * add_Hbdr / _np.mean([len(od._ds["X"]), len(od._ds["Y"])])
    elif add_Hbdr is False:
        add_Hbdr = 0

    if "face" in ds.dims:
        arg = {
            "ds": ds,
            "varList": varList,  # vars and grid coords to transform
            "add_Hbdr": add_Hbdr,
            "XRange": XRange,
            "YRange": YRange,
            "centered": centered,
            "persist": persist,
        }
        dsnew = _llc_trans.arctic_crown(**arg)
        dsnew = dsnew.set_coords(co_list)

        grid_coords = {
            "Y": {"Y": None, "Yp1": 0.5},
            "X": {"X": None, "Xp1": 0.5},
            "Z": {"Z": None, "Zp1": 0.5, "Zu": 0.5, "Zl": -0.5},
            "time": {"time": -0.5},
        }
        grid_coords = {"add_midp": True, "overwrite": True, "grid_coords": grid_coords}
        od._ds = dsnew

        # check if XU, YU, XV and YV need to be calculated:
        vel_grid = ["XU", "YU", "XV", "YV"]
        da_list = [var for var in dsnew.reset_coords().data_vars]
        check = all([item in da_list for item in vel_grid])
        if check:  # pragma: no cover
            manipulate_coords = {"coordsUVfromG": False}
        else:  # pragma: no cover
            manipulate_coords = {"coordsUVfromG": True}

        new_face_connections = {"face_connections": {None: {None, None}}}
        od = od.set_face_connections(**new_face_connections)
        od = od.manipulate_coords(**manipulate_coords)
        od = od.set_grid_coords(**grid_coords)
        od._ds.attrs["OceanSpy_description"] = "Cutout of"
        "simulation, with simple topology (face not a dimension)"
        # Unpack the new dataset without face as dimension
        ds = od._ds

    # ---------------------------
    # Horizontal CUTOUT part II (continuation of original code)
    # ---------------------------
    # Initialize horizontal mask
    if XRange is not None or YRange is not None:
        if XRange is not None:
            XRange, ref_lon = _reset_range(XRange)
        else:
            ref_lon = 180
        maskH, dmaskH, XRange, YRange = get_maskH(
            ds, add_Hbdr, XRange, YRange, ref_lon=ref_lon
        )

        dYp1 = dmaskH["Yp1"].values
        dXp1 = dmaskH["Xp1"].values
        iY = [_np.min(dYp1), _np.max(dYp1)]
        iX = [_np.min(dXp1), _np.max(dXp1)]
        maskH["Yp1"] = ds["Yp1"]
        maskH["Xp1"] = ds["Xp1"]

        # Original length
        lenY = len(ds["Yp1"])
        lenX = len(ds["Xp1"])

        # Indexis
        if iY[0] == iY[1]:
            if "Y" not in dropAxes:
                if iY[0] > 0:
                    iY[0] = iY[0] - 1
                else:
                    iY[1] = iY[1] + 1
        else:
            dropAxes.pop("Y", None)

        if iX[0] == iX[1]:
            if "X" not in dropAxes:
                if iX[0] > 0:
                    iX[0] = iX[0] - 1
                else:
                    iX[1] = iX[1] + 1
        else:
            dropAxes.pop("X", None)

        ds = ds.isel(Yp1=slice(iY[0], iY[1] + 1), Xp1=slice(iX[0], iX[1] + 1))

        Xcoords = od._grid.axes["X"].coords
        if "X" in dropAxes:
            if iX[0] == len(ds["X"]):
                iX[0] = iX[0] - 1
                iX[1] = iX[1] - 1
            ds = ds.isel(X=slice(iX[0], iX[1] + 1))
        elif ("outer" in Xcoords and Xcoords["outer"] == "Xp1") or (
            "left" in Xcoords and Xcoords["left"] == "Xp1"
        ):
            ds = ds.isel(X=slice(iX[0], iX[1]))
        elif "right" in Xcoords and Xcoords["right"] == "Xp1":
            ds = ds.isel(X=slice(iX[0] + 1, iX[1] + 1))

        Ycoords = od._grid.axes["Y"].coords
        if "Y" in dropAxes:
            if iY[0] == len(ds["Y"]):
                iY[0] = iY[0] - 1
                iY[1] = iY[1] - 1
            ds = ds.isel(Y=slice(iY[0], iY[1] + 1))
        elif ("outer" in Ycoords and Ycoords["outer"] == "Yp1") or (
            "left" in Ycoords and Ycoords["left"] == "Yp1"
        ):
            ds = ds.isel(Y=slice(iY[0], iY[1]))
        elif "right" in Ycoords and Ycoords["right"] == "Yp1":
            ds = ds.isel(Y=slice(iY[0] + 1, iY[1] + 1))

        # Cut axis can't be periodic
        if (len(ds["Yp1"]) < lenY or "Y" in dropAxes) and "Y" in periodic:
            periodic.remove("Y")
        if (len(ds["Xp1"]) < lenX or "X" in dropAxes) and "X" in periodic:
            periodic.remove("X")

    # ---------------------------
    # Horizontal MASK
    # ---------------------------

    if mask_outside and (YRange is not None or XRange is not None):
        if YRange is not None:
            minY = YRange[0]
            maxY = YRange[1]
        else:
            minY = ds["YG"].min().values
            maxY = ds["YG"].max().values
        if XRange is not None:
            minX = XRange[0]
            maxX = XRange[1]
        else:
            minX = ds["XG"].min().values
            maxX = ds["XG"].max().values

        maskC = _xr.where(
            _np.logical_and(
                _np.logical_and(ds["YC"] >= minY, ds["YC"] <= maxY),
                _np.logical_and(
                    _rel_lon(ds["XC"], ref_lon) >= _rel_lon(minX, ref_lon),
                    _rel_lon(ds["XC"], ref_lon) <= _rel_lon(maxX, ref_lon),
                ),
            ),
            1,
            0,
        ).persist()
        maskG = _xr.where(
            _np.logical_and(
                _np.logical_and(ds["YG"] >= minY, ds["YG"] <= maxY),
                _np.logical_and(
                    _rel_lon(ds["XG"], ref_lon) >= _rel_lon(minX, ref_lon),
                    _rel_lon(ds["XG"], ref_lon) <= _rel_lon(maxX, ref_lon),
                ),
            ),
            1,
            0,
        ).persist()
        maskU = _xr.where(
            _np.logical_and(
                _np.logical_and(ds["YU"] >= minY, ds["YU"] <= maxY),
                _np.logical_and(
                    _rel_lon(ds["XU"], ref_lon) >= _rel_lon(minX, ref_lon),
                    _rel_lon(ds["XU"], ref_lon) <= _rel_lon(maxX, ref_lon),
                ),
            ),
            1,
            0,
        ).persist()
        maskV = _xr.where(
            _np.logical_and(
                _np.logical_and(ds["YV"] >= minY, ds["YV"] <= maxY),
                _np.logical_and(
                    _rel_lon(ds["XV"], ref_lon) >= _rel_lon(minX, ref_lon),
                    _rel_lon(ds["XV"], ref_lon) <= _rel_lon(maxX, ref_lon),
                ),
            ),
            1,
            0,
        ).persist()

        for var in ds.data_vars:
            if set(["X", "Y"]).issubset(ds[var].dims):
                ds[var] = ds[var].where(maskC.compute(), drop=True)
            elif set(["Xp1", "Yp1"]).issubset(ds[var].dims):
                ds[var] = ds[var].where(maskG.compute(), drop=True)
            elif set(["Xp1", "Y"]).issubset(ds[var].dims):
                ds[var] = ds[var].where(maskU.compute(), drop=True)
            elif set(["X", "Yp1"]).issubset(ds[var].dims):
                ds[var] = ds[var].where(maskV.compute(), drop=True)

    # ---------------------------
    # TIME RESAMPLING
    # ---------------------------
    # Resample in time
    if timeFreq:
        # Infer original frequency
        inFreq = _pd.infer_freq(ds.time.values)
        if timeFreq[0].isdigit() and not inFreq[0].isdigit():
            inFreq = "1" + inFreq

        # Same frequency: Skip
        if timeFreq == inFreq:
            _warnings.warn(
                "\nInput time freq:"
                "[{}] = Output time frequency: [{}]:"
                "\nSkip time resampling."
                "".format(inFreq, timeFreq),
                stacklevel=2,
            )

        else:
            # Remove time_midp and warn
            vars2drop = [var for var in ds.variables if "time_midp" in ds[var].dims]
            if vars2drop:
                _warnings.warn(
                    "\nTime resampling drops variables"
                    " on `time_midp` dimension."
                    "\nDropped variables: {}."
                    "".format(vars2drop),
                    stacklevel=2,
                )
                ds = ds.drop_vars(vars2drop)

            # Snapshot
            if sampMethod == "snapshot":
                # Find new times
                time2sel = ds["time"].resample(time=timeFreq).first()
                newtime = ds["time"].sel(time=time2sel)

                # Use slice when possible
                inds = [
                    i for i, t in enumerate(ds["time"].values) if t in newtime.values
                ]
                inds = _xr.DataArray(inds, dims="time")
                ds = ds.isel(time=inds)

            else:
                # Mean
                # Separate time and timeless
                attrs = ds.attrs
                ds_dims = ds.drop_vars(
                    [var for var in ds.variables if var not in ds.dims]
                )
                ds_time = ds.drop_vars(
                    [var for var in ds.variables if "time" not in ds[var].dims]
                )
                ds_timeless = ds.drop_vars(
                    [var for var in ds.variables if "time" in ds[var].dims]
                )

                # Resample
                ds_time = ds_time.resample(time=timeFreq).mean("time")

                # Add all dimensions to ds, and fix attributes
                for dim in ds_time.dims:
                    if dim == "time":
                        ds_time[dim].attrs = ds_dims[dim].attrs
                    else:
                        ds_time[dim] = ds_dims[dim]

                # Merge
                ds = _xr.merge([ds_time, ds_timeless])
                ds.attrs = attrs

    # Update oceandataset
    od._ds = ds

    # Add time midp
    if timeFreq and "time" not in dropAxes:
        od = od.set_grid_coords(
            {**od.grid_coords, "time": {"time": -0.5}}, add_midp=True, overwrite=True
        )

    # Drop axes
    grid_coords = od.grid_coords
    for coord in list(grid_coords):
        if coord in dropAxes:
            grid_coords.pop(coord, None)
    od = od.set_grid_coords(grid_coords, overwrite=True)

    # Cut axis can't be periodic
    od = od.set_grid_periodic(periodic)

    return od


def mooring_array(od, Ymoor, Xmoor, xoak_index="scipy_kdtree", **kwargs):
    """
    Extract a mooring array section following the grid.
    Trajectories are great circle paths if coordinates are spherical.

    Parameters
    ----------
    od: OceanDataset
        od that will be subsampled.
    Ymoor: 1D array_like, scalar
        Y coordinates of moorings.
    Xmoor: 1D array_like, scalar
        X coordinates of moorings.
    xoak_index: str
        xoak index to be used. `scipy_kdtree` by default.
    **kwargs:
        Keyword arguments for :py:func:`oceanspy.subsample.cutout`.

    Returns
    -------
    od: OceanDataset
        Subsampled oceandataset.
    """

    # Check
    _check_native_grid(od, "mooring_array")

    # Useful variable
    R = od.parameters["rSphere"]

    if R is not None:
        # array defines a great circle path.
        Ymoor, Xmoor = circle_path_array(Ymoor, Xmoor, R)

    # Convert variables to numpy arrays and make some check
    Ymoor = _check_range(od, Ymoor, "Ymoor")
    Xmoor = _check_range(od, Xmoor, "Xmoor")

    serial = kwargs.pop("serial", None)

    if serial:
        _diffXYs = True
        varList = kwargs.pop("varList", None)

        args = {
            "varList": varList,
            "Xcoords": Xmoor,
            "Ycoords": Ymoor,
            "dim_name": "mooring",
        }
        od = _copy.deepcopy(od)

        # indexes needed for transport
        Yind, Xind = _xr.broadcast(od._ds["Y"], od._ds["X"])
        Yind = Yind.expand_dims({"face": od._ds["face"]})
        Xind = Xind.expand_dims({"face": od._ds["face"]})
        od._ds["Xind"] = Xind.transpose(*od._ds["XC"].dims)
        od._ds["Yind"] = Yind.transpose(*od._ds["YC"].dims)
        od._ds = od._ds.set_coords(["Yind", "Xind"])

        # when passed, od.subsample.statins returns dataset
        new_ds, diffX, diffY = od.subsample.stations(**args)
        coords = [var for var in new_ds.coords]

        # TODO: need to add Xind, Yind
        # needed for transports (via cutout)

    else:
        _diffXYs = False
        # Cutout
        if "YRange" not in kwargs:  # pragma: no cover
            kwargs["YRange"] = Ymoor
        if "XRange" not in kwargs:  # pragma: no cover
            kwargs["XRange"] = Xmoor
        if "add_Hbdr" not in kwargs:  # pragma: no cover
            kwargs["add_Hbdr"] = True
        od = od.subsample.cutout(**kwargs)

        # Add indexes needed for transports
        Yind, Xind = _xr.broadcast(od._ds["Y"], od._ds["X"])
        od._ds["Xind"] = Xind.transpose(*od._ds["XC"].dims)
        od._ds["Yind"] = Yind.transpose(*od._ds["YC"].dims)
        od._ds = od._ds.set_coords(["Xind", "Yind"])

        # Message
        print("Extracting mooring array.")

        # Unpack ds
        ds = od._ds
        # create list of coordinates.
        coords = [var for var in ds if "time" not in ds[var].dims]

        ds_grid = ds[["XC", "YC"]]  # by convention center point

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

        ix, iy = (nds["i" + f"{i}"].data for i in ("X", "Y"))

        # Remove duplicates that are next to each other.
        mask = _np.argwhere(_np.abs(_np.diff(ix)) + _np.abs(_np.diff(iy)) == 0)
        ix, iy = (_np.delete(ii, mask) for ii in (ix, iy))

        # Initialize variables
        dx, dy, inds = diff_and_inds_where_insert(ix, iy)
        while inds.size:
            dx, dy = (di[inds] for di in (dx, dy))
            mask = _np.abs(dx * dy) == 1
            ix = _np.insert(ix, inds + 1, ix[inds] + (dx / 2).astype(int))
            iy = _np.insert(
                iy, inds + 1, iy[inds] + _np.where(mask, dy, (dy / 2).astype(int))
            )
            # Prepare for next iteration
            dx, dy, inds = diff_and_inds_where_insert(ix, iy)

        # attempt to remove repeated (but not adjacent) coord values
        ix, iy = remove_repeated(ix, iy)

        new_ds = eval_dataset(ds, ix, iy)

    mooring = new_ds.mooring

    near_Y = new_ds["YC"].values
    near_X = new_ds["XC"].values

    # Add distance (0 always first element)
    if R is not None:
        dists = _np.array(
            [0]
            + [
                _great_circle(
                    (near_Y[i + 1], near_X[i + 1]), (near_Y[i], near_X[i]), radius=R
                ).km
                for i in range(len(near_Y) - 1)
            ]
        )
        unit = "km"
    else:
        dists = _np.sqrt(
            (near_Y[1:] - near_Y[:-1]) ** 2 + (near_X[1:] - near_X[:-1]) ** 2
        )
        dists = _np.insert(dists, 0, 0)  # add zero as 1st element
        if "units" in new_ds["XC"].attrs:
            unit = new_ds["XC"].attrs["units"]
        else:
            unit = "None"

    dists = _np.cumsum(dists)
    distance = DataArray(
        dists,
        coords={"mooring": mooring},
        dims=("mooring"),
        attrs={"long_name": "Distance from first mooring", "units": unit},
    )

    new_ds["mooring_dist"] = distance

    # Reset coordinates
    new_ds = new_ds.set_coords(coords + ["mooring_dist"])

    # Recreate od
    od._ds = new_ds

    # remove complex topology from grid
    if od.face_connections is not None:
        new_face_connections = {"face_connections": {None: {None, None}}}
        od = od.set_face_connections(**new_face_connections)
        grid_coords = od.grid_coords
        # remove face from grid coord
        grid_coords.pop("face", None)
        od = od.set_grid_coords(grid_coords, overwrite=True)

    od = od.set_grid_coords(
        {"mooring": {"mooring": -0.5}}, add_midp=True, overwrite=False
    )

    # Create dist_midp
    _grid = od._grid
    dist_midp = _xr.DataArray(
        _grid.interp(od._ds["mooring_dist"], "mooring"),
        attrs=od._ds["mooring_dist"].attrs,
    )
    od = od.merge_into_oceandataset(dist_midp.rename("mooring_midp_dist"))

    if _diffXYs:  # pragma: no cover
        moor_midp = od._ds.mooring_midp.values
        if diffX.size == len(moor_midp):
            # include in dataset
            xr_diffX = DataArray(
                diffX,
                coords={"mooring_midp": moor_midp},
                dims=("mooring_midp"),
                attrs={"long_name": "x-difference between moorings", "units": unit},
            )

            xr_diffY = DataArray(
                diffY,
                coords={"mooring_midp": moor_midp},
                dims=("mooring_midp"),
                attrs={"long_name": "y-difference between moorings", "units": unit},
            )

            od._ds["diffX"] = xr_diffX
            od._ds["diffY"] = xr_diffY
        else:
            print(diffX.size)
            _warnings.warn(
                "diffX and diffY have inconsistent lengths with mooring dimension"
            )

        # compute missing grid velocities from datasets if necessary
        vel_grid = ["XU", "YU", "XV", "YV"]
        da_list = [var for var in od._ds.reset_coords().data_vars]
        check = all([item in da_list for item in vel_grid])
        if check:  # pragma: no cover
            manipulate_coords = {"coordsUVfromG": False}
        else:  # pragma: no cover
            manipulate_coords = {"coordsUVfromG": True}
        od = od.manipulate_coords(**manipulate_coords)
        od._ds = od._ds.set_coords(
            coords + vel_grid + ["mooring_dist", "mooring_midp_dist"]
        )
    else:
        od._ds = od._ds.set_coords(coords + ["mooring_midp_dist"])

    return od


def survey_stations(
    od,
    Ysurv,
    Xsurv,
    delta=None,
    xesmf_regridder_kwargs={"method": "bilinear"},
    **kwargs,
):
    """
    Extract survey stations.
    Trajectories are great circle paths if coordinates are spherical.

    Parameters
    ----------
    od: OceanDataset
        od that will be subsampled.
    Ysurv: 1D array_like
        Y coordinates of stations.
    Xsurv: 1D array_like,
        X coordinates of stations.
    delta: scalar, None
        Distance between stations.
        Units are km for spherical coordinate,
        same units of coordinates for cartesian.
        If None, only (Ysurv, Xsurv) stations are returned.
    xesmf_regridder_kwargs: dict
        Keyword arguments for xesmf.regridder, such as `method`.
        Defaul method: `bilinear`.
    **kwargs:
        Keyword arguments for :py:func:`oceanspy.subsample.cutout`.

    Returns
    -------
    od: OceanDataset
        Subsampled oceandataset.

    References
    ----------
    https://xesmf.readthedocs.io/en/stable/user_api.html#regridder

    Notes
    -----
    By default, kwargs['add_Hbdr'] = True.
    Try to play with add_Hbdr values if zeros/nans are returned.
    This function interpolates using xesmf.regridder,
    and does not support lazy computation.

    xesmf.regridder currently dosen't allow
    to set the coordinates system (default is spherical).
    Surveys using cartesian coordinates can be made
    by changing the xesmf source code
    as explained here: https://github.com/JiaweiZhuang/xESMF/issues/39
    """

    # Check
    _check_native_grid(od, "survey_stations")

    # Convert variables to numpy arrays and make some check
    Ysurv = _check_range(od, Ysurv, "Ysurv")
    Xsurv = _check_range(od, Xsurv, "Xsurv")

    # Check xesmf arguments
    _check_instance({"xesmf_regridder_kwargs": xesmf_regridder_kwargs}, "dict")

    # Earth Radius
    R = od.parameters["rSphere"]
    if R is None:
        _warnings.warn(
            "\noceanspy.survey_stations interpolates"
            " using xesmf.regridder."
            "\nxesmf.regridder currently dosen't allow"
            " to set the coordinates system (default is spherical)."
            "\nSurveys using cartesian coordinates can be made"
            " by changing the xesmf source code as explained here:"
            "https://github.com/JiaweiZhuang/xESMF/issues/39",
            stacklevel=2,
        )

    # Compute trajectory
    for i, (lat0, lon0, lat1, lon1) in enumerate(
        zip(Ysurv[:-1], Xsurv[:-1], Ysurv[1:], Xsurv[1:])
    ):
        if R is not None:
            # SPHERICAL: follow great circle path
            this_Y, this_X, this_dists = _utils.great_circle_path(
                lat0, lon0, lat1, lon1, delta, R=R
            )
        else:  # pragma: no cover
            # CARTESIAN: just a simple interpolation
            this_Y, this_X, this_dists = _utils.cartesian_path(
                lat0, lon0, lat1, lon1, delta
            )
        if i == 0:
            Y_surv = this_Y
            X_surv = this_X
            dists_surv = this_dists
        else:
            this_Y = _np.delete(this_Y, 0, axis=None)
            this_X = _np.delete(this_X, 0, axis=None)
            this_dists = _np.delete(this_dists, 0, axis=None)
            if len(this_dists) == 0:
                continue
            this_dists = this_dists + dists_surv[-1]
            Y_surv = _np.concatenate((Y_surv, this_Y))
            X_surv = _np.concatenate((X_surv, this_X))
            dists_surv = _np.concatenate((dists_surv, this_dists))

    # Cutout
    if "YRange" not in kwargs:
        kwargs["YRange"] = Y_surv
    if "XRange" not in kwargs:
        kwargs["XRange"] = X_surv
    if "add_Hbdr" not in kwargs:
        kwargs["add_Hbdr"] = True
    od = od.subsample.cutout(**kwargs)

    # Message
    print("Carrying out survey.")

    # Unpack ds and grid
    ds = od._ds
    grid = od._grid

    # TODO: This is probably slowing everything down,
    #       and perhaps adding some extra error.
    #       I think we should have separate xesmf interpolations
    #       for different grids, then merge.

    # Move all variables on same spatial grid
    for var in [var for var in ds.variables if var not in ds.dims]:
        for dim in ["Xp1", "Yp1"]:
            if dim in ds[var].dims and var != dim:
                attrs = ds[var].attrs
                ds[var] = grid.interp(
                    ds[var],
                    axis=dim[0],
                    to="center",
                    boundary="fill",
                    fill_value=_np.nan,
                )
                ds[var].attrs = attrs

    # Create xesmf datsets
    ds_in = ds
    ds_in = ds_in.reset_coords()
    ds_in["lat"] = ds_in["YC"]
    ds_in["lon"] = ds_in["XC"]
    ds = _xr.Dataset(
        {"lat": (["lat"], Y_surv), "lon": (["lon"], X_surv)}, attrs=ds.attrs
    )

    # Interpolate
    try:
        regridder = _xe.Regridder(ds_in, ds, **xesmf_regridder_kwargs)
    except ValueError:
        raise ValueError(
            """
        An error occured when creating the xesmf.Regridder object,
        try add_Hbdr = M, where M>1.5 times horizontal spacing
        """
        )
    regridder._grid_in = None  # See https://github.com/JiaweiZhuang/xESMF/issues/71
    regridder._grid_out = None  # See https://github.com/JiaweiZhuang/xESMF/issues/71
    interp_vars = [
        var for var in ds_in.variables if var not in ["lon", "lat", "X", "Y"]
    ]
    print(
        "Variables to interpolate: {}."
        "".format(
            [var for var in interp_vars if set(["X", "Y"]).issubset(ds_in[var].dims)]
        )
    )
    for var in interp_vars:
        if set(["X", "Y"]).issubset(ds_in[var].dims):
            print("Interpolating [{}].".format(var))
            attrs = ds_in[var].attrs
            ds[var] = regridder(ds_in[var])
            ds[var].attrs = attrs
        elif var not in ["Xp1", "Yp1"]:
            ds[var] = ds_in[var].reset_coords(drop=True)
    if _parse_version(_xe.__version__) < _parse_version("0.4.0"):
        regridder.clean_weight_file()

    # Extract transect
    ds = ds.isel(
        lat=_xr.DataArray(_np.arange(len(Y_surv)), dims="station"),
        lon=_xr.DataArray(_np.arange(len(X_surv)), dims="station"),
    )

    # Add station dimension
    ds["station"] = _xr.DataArray(
        _np.arange(len(X_surv)),
        dims=("station"),
        attrs={"long_name": "index of survey station", "units": "none"},
    )

    # Add distance
    ds["station_dist"] = _xr.DataArray(
        dists_surv, dims=("station"), attrs={"long_name": "Distance from first station"}
    )
    if R is not None:
        # SPHERICAL
        ds["station_dist"].attrs["units"] = "km"
    else:  # pragma: no cover
        # CARTESIAN
        if "units" in ds["lat"].attrs:
            ds["station_dist"].attrs["units"] = ds["lat"].attrs["units"]
    ds = ds.set_coords("station_dist")

    # Return od
    od._ds = ds
    grid_coords = od.grid_coords
    grid_coords.pop("X", None)
    grid_coords.pop("Y", None)
    od = od.set_grid_coords(grid_coords, overwrite=True)
    od = od.set_grid_coords(
        {"station": {"station": -0.5}}, add_midp=True, overwrite=False
    )

    # Create dist_midp
    _grid = od._grid
    dist_midp = _xr.DataArray(
        _grid.interp(od._ds["station_dist"], "station"),
        attrs=od._ds["station_dist"].attrs,
    )
    od = od.merge_into_oceandataset(dist_midp.rename("station_midp_dist"))
    od._ds = od._ds.set_coords(
        [coord for coord in od._ds.coords] + ["station_midp_dist"]
    )

    return od


def stations(
    od,
    varList=None,
    tcoords=None,
    Zcoords=None,
    Ycoords=None,
    Xcoords=None,
    xoak_index="scipy_kdtree",
    method="nearest",
    dim_name="station",
):
    """
    Extract nearest-neighbor data from given spatial coordinate.
    Data may be isolated and unordered (`dim_name=stations`), or contiguous
    and unit distanced (`dim_name=mooring`).

    Following the C-grid convention,
    for every scalar point extracted (along the new dimension `dim_name`)
    returns 4 velocity points: 2 U-points, 2 V-points and their respective
    coordinates, and 4 corner coordinate points.

    Parameters
    ----------
    od: OceanDataset
        od that will be subsampled.
    varList: 1D array_lie, NoneType
        variable names to sample.
    tcoords: 1D array_like, NoneType
        time-coordinates (datetime).
    Zcoords: 1D array_like, NoneType
        Z coordinates at center point
    Ycoords: 1D array_like, NoneType
        Latitude coordinates of locations at center point.
    Xcoords: 1D array_like, NoneType
        lon coordinates of locations at center point.
    xoak_index: str
        xoak index to be used. `scipy_kdtree` by default.
    method: str, `nearest` (default).
        see .sel via xarray.dataSet.sel method
    dim_name: str
        `station` (default) or `mooring`.

    Returns
    -------
    Depending on the choice of dim_name, two types of returns:
    see https://github.com/hainegroup/oceanspy/issues/398

    1) if `dim_name: 'stations'`

    od: OceanDataset
        Subsampled oceandataset.

    2) if `dim_name: 'mooring'`

    ds: xarray.dataset
    diffX: numpy.array
    diffX: numpy.array


    See Also
    --------
    oceanspy.subsample.mooring_array

    """
    _check_native_grid(od, dim_name)

    # Convert variables to numpy arrays and make some check
    tcoords = _check_range(od, tcoords, "timeRange")
    Zcoords = _check_range(od, Zcoords, "Zcoords")
    Ycoords = _check_range(od, Ycoords, "Ycoords")
    Xcoords = _check_range(od, Xcoords, "Xcoords")

    # Message
    message = "Extracting " + dim_name
    if dim_name == "mooring":
        message = message + " array"
    else:
        message = message + "s"
    print(message)

    # Unpack ds
    od = _copy.deepcopy(od)
    ds = od._ds
    face_connections = od.face_connections["face"]

    if varList is not None:
        nvarlist = [var for var in ds.data_vars if var not in varList]
        ds = ds.drop_vars(nvarlist)

    # look up nearest neighbors in Z and time dims
    Zlist = ["Zl", "Z", "Zp1", "Zu"]
    tlist = ["time", "time_midp"]
    dimlist, Coords = [], []
    if Zcoords is not None:
        dimlist.append(Zlist)
        Coords.append(Zcoords)
    if tcoords is not None:
        dimlist.append(tlist)
        Coords.append(tcoords)

    for i in range(len(dimlist)):
        List = [k for k in dimlist[i] if k in ds.dims]
        args = {}
        for item in List:
            if len(ds[item]) > 0:
                args[item] = Coords[i]
        ds = ds.sel(**args, method="nearest")

    # create list of coordinates.
    co_list = [var for var in ds.coords if var not in ["face"]]

    if Xcoords is None and Ycoords is None:
        DS = ds

    if Xcoords is not None and Ycoords is not None:
        ds_grid = ds[["XC", "YC"]]

        if dim_name == "mooring":  # needed for transport
            for key, value in ds_grid.sizes.items():
                ds_grid["i" + f"{key}"] = DataArray(range(value), dims=key)

        if xoak_index not in _xoak.IndexRegistry():
            raise ValueError(
                "`xoak_index` [{}] is not supported."
                "\nAvailable options: {}"
                "".format(xoak_index, _xoak.IndexRegistry())
            )
        ds_grid.xoak.set_index(["XC", "YC"], xoak_index)

        cdata = {"XC": (dim_name, Xcoords), "YC": (dim_name, Ycoords)}
        ds_data = _xr.Dataset(cdata)

        # find nearest points to given data.
        nds = ds_grid.xoak.sel(XC=ds_data["XC"], YC=ds_data["YC"])

        if "face" not in ds.dims:  # pragma: no cover
            iX, iY = (nds[f"{i}"].data for i in ("X", "Y"))
            DS = eval_dataset(ds, iX, iY, _dim_name=dim_name)
            DS = DS.squeeze()
        else:
            ds = mates(ds)
            varlist = [var for var in ds.reset_coords().data_vars if var not in "face"]
            attrs = {}
            for var in varlist:
                attrs[var] = ds[var].attrs
            iX, iY, iface = (nds[f"{i}"].data for i in ("X", "Y", "face"))
            _dat = nds.face.values
            ll = _np.where(abs(_np.diff(_dat)))[0]
            order_iface = [_dat[i] for i in ll] + [_dat[-1]]
            Niter = len(order_iface)
            if Niter == 1:
                args = {
                    "_ds": ds,
                    "_ix": iX,
                    "_iy": iY,
                    "_faces": order_iface,  # single element list
                    "_iface": 0,  # index of face
                    "_face_connections": face_connections,
                }
                if dim_name == "mooring":
                    nix, niy = connector(iX, iY)
                    DS, nix, niy = mooring_singleface(**args)
                    if order_iface[0] in _np.arange(7, 13):
                        DS = flip_v(mates(DS))
                    diffX, diffY, *a = cross_face_diffs(
                        DS, nix, niy, order_iface, 0, face_connections
                    )
                    return DS.persist(), diffX, diffY
                if dim_name == "station":  # pragma: no cover
                    DS = station_singleface(**args).persist()
                    if order_iface[0] in _np.arange(7, 13):
                        DS = flip_v(mates(DS))
            if Niter > 1:
                nX0, nY0 = splitter(iX, iY, iface)
                args = {
                    "_ds": ds,
                    "_faces": order_iface,
                    "_face_connections": face_connections,
                }
                DSf = []
                shift = 0
                diffsX, diffsY = _np.array([]), _np.array([])
                for ii in range(Niter):
                    if dim_name == "station":
                        _returns = False
                        args1 = {"_ix": nX0[ii], "_iy": nY0[ii], "_iface": ii}
                        dse = station_singleface(**{**args, **args1})
                        if order_iface[ii] in _np.arange(7, 13):
                            dse = flip_v(mates(dse))
                    if dim_name == "mooring":
                        _returns = True
                        nix, niy = fill_path(
                            nX0, nY0, order_iface, ii, face_connections
                        )
                        args1 = {"_ix": nix, "_iy": niy, "_iface": ii}
                        dse, nix, niy = mooring_singleface(**{**args, **args1})
                        if order_iface[ii] in _np.arange(7, 13):
                            dse = flip_v(mates(dse))
                        diX, diY, *a = cross_face_diffs(
                            ds, nix, niy, order_iface, ii, face_connections
                        )
                        diffsX = _np.append(diffsX, diX)
                        diffsY = _np.append(diffsY, diY)
                    for var in dse.reset_coords().data_vars:
                        dse[var].attrs = {}
                    if ii > 0:
                        shift += len(DSf[ii - 1][dim_name])
                        dse = reset_dim(dse, shift, dim=dim_name)
                    DSf.append(dse)
                DS = _xr.combine_by_coords(DSf)
                Ndim = len(DS[dim_name])
                DS = DS.chunk({dim_name: Ndim}).persist()
                del DSf
                for var in DS.reset_coords().data_vars:
                    DS[var].attrs = attrs
                if _returns:
                    return DS, diffsX, diffsY
    DS = DS.set_coords(co_list)

    if Xcoords is None and Ycoords is None:
        od._ds = DS

    if Xcoords is not None and Ycoords is not None:
        if "face" in DS.variables:
            DS = DS.drop_vars(["face"])

        od._ds = DS

        if od.face_connections is not None:  # pragma: no cover
            new_face_connections = {"face_connections": {None: {None, None}}}
            od = od.set_face_connections(**new_face_connections)

        grid_coords = od.grid_coords
        od = od.set_grid_coords(grid_coords, overwrite=True)

    return od


def particle_properties(od, times, Ypart, Xpart, Zpart, **kwargs):
    """
    Extract Eulerian properties of particles
    using nearest-neighbor interpolation.

    Parameters
    ----------
    od: OceanDataset
        od that will be subsampled.
    times: 1D array_like or scalar
        time of particles.
    Ypart: 2D array_like or 1D array_like if times is scalar
        Y coordinates of particles. Dimensions order: (time, particle).
    Xpart: 2D array_like or 1D array_like if times is scalar
        X coordinates of particles. Dimensions order: (time, particle).
    Zpart: 2D array_like or 1D array_like if times is scalar
        Z of particles. Dimensions order: (time, particle).
    **kwargs:
        Keyword arguments for :py:func:`oceanspy.subsample.cutout`.

    Returns
    -------
    od: OceanDataset
        Subsampled oceandataset.

    See Also
    --------
    oceanspy.OceanDataset.create_tree
    """

    # Checks
    _check_native_grid(od, "particle_properties")
    InputDict = _check_part_position(
        od, {"times": times, "Ypart": Ypart, "Xpart": Xpart, "Zpart": Zpart}
    )
    times = InputDict["times"]
    Ypart = InputDict["Ypart"]
    Xpart = InputDict["Xpart"]
    Zpart = InputDict["Zpart"]

    check1 = not Ypart.shape == Xpart.shape == Zpart.shape
    check2 = not times.size == Ypart.shape[0]
    if check1 or check2:
        raise TypeError(
            "`times`, `Xpart`, `Ypart`, and `Zpart`" "have inconsistent shape"
        )

    # Cutout
    if "timeRange" not in kwargs:
        kwargs["timeRange"] = times
    if "YRange" not in kwargs:
        kwargs["YRange"] = [_np.min(Ypart), _np.max(Ypart)]
    if "XRange" not in kwargs:
        kwargs["XRange"] = [_np.min(Xpart), _np.max(Xpart)]
    if "ZRange" not in kwargs:
        kwargs["ZRange"] = [_np.min(Zpart), _np.max(Zpart)]
    if "add_Hbdr" not in kwargs:
        kwargs["add_Hbdr"] = True
    if "add_Vbdr" not in kwargs:
        kwargs["add_Vbdr"] = True
    od = od.subsample.cutout(**kwargs)

    # Message
    print("Extracting Eulerian properties of particles.")

    # Unpack ds and info
    ds = od._ds
    R = od.parameters["rSphere"]

    # Remove time_midp and warn
    vars2drop = [var for var in ds.variables if "time_midp" in ds[var].dims]
    if vars2drop:
        _warnings.warn(
            "\nParticle properties extraction"
            " drops variables on `time_midp` dimension."
            "\nDropped variables: {}.".format(vars2drop),
            stacklevel=2,
        )
        ds = ds.drop_vars(vars2drop)

    # New dimensions
    time = _xr.DataArray(times, dims=("time"), attrs=ds["time"].attrs)
    particle = _xr.DataArray(
        _np.arange(Ypart.shape[1]),
        dims=("particle"),
        attrs={"long_name": "index of particle", "units": "none"},
    )
    i_ds = _xr.Dataset({"time": time, "particle": particle})

    # Find vertical and time indexes
    for dim in ds.dims:
        if dim == "time":
            tmp = _xr.DataArray(times, dims=("time"))
        elif dim[0] == "Z":
            tmp = _xr.DataArray(Zpart, dims=("time", "particle"))
        else:
            locals().pop("tmp", None)
            continue
        itmp = _xr.DataArray(
            _np.arange(len(ds[dim])), coords={dim: ds[dim].values}, dims={dim}
        )
        itmp = itmp.sel({dim: tmp}, method="nearest")
        i_ds["i" + dim] = itmp

    # Convert 2 cartesian
    if R is not None:
        x, y, z = _utils.spherical2cartesian(Y=Ypart, X=Xpart, R=R)
    else:
        x = Xpart
        y = Ypart
        z = _np.zeros(y.shape)

    # Find horizontal indexes
    all_vars = {}
    for grid_pos in ["C", "U", "V", "G"]:
        # Don't create tree if no variables
        var_grid_pos = [
            var
            for var in ds.data_vars
            if set(["X" + grid_pos, "Y" + grid_pos]).issubset(ds[var].coords)
        ]
        if not var_grid_pos:
            continue
        this_ds = _xr.Dataset({var: od._ds[var] for var in var_grid_pos})

        # Useful variables
        Y = this_ds["Y" + grid_pos]
        X = this_ds["X" + grid_pos]
        shape = X.shape
        Yname = [dim for dim in Y.dims if dim[0] == "Y"][0]
        Xname = [dim for dim in X.dims if dim[0] == "X"][0]
        Yindex = X.dims.index(Yname)
        Xindex = X.dims.index(Xname)

        # Create tree
        tree = od.create_tree(grid_pos=grid_pos)

        # Indexes of nearest grid points
        _, indexes = tree.query(
            _np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        )
        indexes = _np.unravel_index(indexes, shape)
        iY = _xr.DataArray(
            _np.reshape(indexes[Yindex], y.shape), dims=("time", "particle")
        )
        iX = _xr.DataArray(
            _np.reshape(indexes[Xindex], x.shape), dims=("time", "particle")
        )

        # Transform indexes in DataArray and add to dictionary
        i_ds["i" + Yname] = iY
        i_ds["i" + Xname] = iX

        # Subsample (looping is faster)
        add_vars = {
            var: this_ds[var].isel({dim: i_ds["i" + dim] for dim in this_ds[var].dims})
            for var in this_ds.data_vars
        }
        add_vars = {k: v.drop_vars([Xname, Yname]) for k, v in add_vars.items()}
        all_vars = {**all_vars, **add_vars}

    # Recreate od
    new_ds = _xr.Dataset(all_vars)
    for var in od._ds.variables:
        if var in new_ds.variables:
            new_ds[var].attrs = od._ds[var].attrs
    od._ds = new_ds

    # Add time midp
    od = od.set_grid_coords({"time": {"time": -0.5}}, add_midp=True, overwrite=True)

    # Reset coordinates
    od._ds = od._ds.reset_coords()

    return od


class _subsampleMethods(object):
    """
    Enables use of functions as OceanDataset attributes.
    """

    def __init__(self, od):
        self._od = od

    @_functools.wraps(cutout)
    def cutout(self, **kwargs):
        return cutout(self._od, **kwargs)

    @_functools.wraps(mooring_array)
    def mooring_array(self, **kwargs):
        return mooring_array(self._od, **kwargs)

    @_functools.wraps(survey_stations)
    def survey_stations(self, **kwargs):
        return survey_stations(self._od, **kwargs)

    @_functools.wraps(stations)
    def stations(self, **kwargs):
        return stations(self._od, **kwargs)

    @_functools.wraps(particle_properties)
    def particle_properties(self, **kwargs):
        return particle_properties(self._od, **kwargs)
