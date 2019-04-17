# Instructions for developers:
# This modules collect useful functions used by OceanSpy.
# All functions here must be private (names start with underscore `_`)

# Import modules (can be public here)
import numpy
import warnings
import xgcm


# =========
# FUNCTIONS
# =========
def _create_grid(dataset, coords, periodic):
    """
    Create xgcm grid by adding comodo attributes to the
    dimensions of the dataset.

    Parameters
    ----------
    dataset: xarray.Dataset
    coords: dict
        E.g., {'Y': {Y: None, Yp1: 0.5}}
    periodic: list
        List of periodic axes.

    Returns
    -------
    grid: xgcm.Grid
    """
    # Clean up comodo (currently force user to specify axis using set_coords).
    for dim in dataset.dims:
        dataset[dim].attrs.pop('axis', None)
        dataset[dim].attrs.pop('c_grid_axis_shift', None)

    # Add comodo attributes.
    # TODO: it is possible to pass grid dict in xgcm.
    #       Should we implement it?
    warn_dims = []
    if coords:
        for axis in coords:
            for dim in coords[axis]:
                if dim not in dataset.dims:
                    warn_dims = warn_dims+[dim]
                else:
                    shift = coords[axis][dim]
                    dataset[dim].attrs['axis'] = axis
                    if shift:
                        dataset[dim].attrs['c_grid_axis_shift'] = str(shift)
    if len(warn_dims) != 0:
        warnings.warn("{} are not dimensions"
                      " and are not added"
                      " to the grid object.".format(warn_dims), stacklevel=2)
    # Create grid
    grid = xgcm.Grid(dataset, periodic=periodic)
    if len(grid.axes) == 0:
        grid = None

    return grid


def _check_instance(objs, classinfos):
    """
    Check if the object is an instance or subclass of classinfo class.

    Parameters
    ----------
    objs: dict
        {'obj_name': obj}
    classinfos: dict
        E.g.: {'obj_name': ['float', 'int']}
    """
    for key, value in objs.items():
        if isinstance(classinfos, str):
            classinfo = classinfos
        else:
            classinfo = classinfos[key]

        if isinstance(classinfo, str):
            classinfo = [classinfo]

        check = []
        for this_classinfo in classinfo:
            if '.' in this_classinfo:
                package = this_classinfo.split('.')[0]
                exec('import {}'.format(package))

            check = check + [eval('isinstance(value, {})'
                                  ''.format(this_classinfo))]

        if not any(check):
            raise TypeError("`{}` must be {}".format(key, classinfo))


def _check_oceanspy_axes(axes2check):
    """
    Check that axes are OceanSpy axes

    Parameters
    ----------
    axes2check: list
        List of axes
    """
    from oceanspy import OCEANSPY_AXES

    for axis in axes2check:
        if axis not in OCEANSPY_AXES:
            raise ValueError(_wrong_axes_error_message(axes2check))


def _check_list_of_string(obj, objName):
    """
    Check that object is a list of strings

    Parameters
    ----------
    obj: str or list
        Object to check
    objName: str
        Name of the object

    Returns
    -------
    obj: list
        List of strings
    """
    if obj is not None:
        obj = numpy.asarray(obj, dtype='str')
        if obj.ndim == 0:
            obj = obj.reshape(1)
        elif obj.ndim > 1:
            raise TypeError('Invalid `{}`'.format(objName))
    return obj


def _check_range(od, obj, objName):
    """
    Check values of a range, and return an object which is
    compatible with OceanSpy's functions.

    Parameters
    ----------
    od: OceanDataset
    obj: Range object
    objName: Name of the object

    Returns
    -------
    obj: Range object
    """
    if obj is not None:
        prefs = ['Y', 'X', 'Z', 'time']
        coords = ['YG', 'XG', 'Zp1', 'time']
        for _, (pref, coord) in enumerate(zip(prefs, coords)):
            if pref in objName:
                valchek = od._ds[coord]
                break
        obj = numpy.asarray(obj, dtype=valchek.dtype)
        if obj.ndim == 0:
            obj = obj.reshape(1)
        elif obj.ndim > 1:
            raise TypeError('Invalid `{}`'.format(objName))
        maxcheck = valchek.max().values
        mincheck = valchek.min().values
        if any(obj < mincheck) or any(obj > maxcheck):
            warnings.warn("\n{}Range of the oceandataset is: {}"
                          "\nRequested {} has values outside this range."
                          "".format(pref, [mincheck, maxcheck], objName),
                          stacklevel=2)
    return obj


def _handle_aliased(od, aliased, varNameList):
    """
    Return OceanSpy reference name and corresponding alises.

    Parameters
    ----------
    od: OceanDataset
    aliased: bool
    varNameList: list
        List of variables name

    Returns
    -------
    varNameListIN: list
        List of OceanSpy reference names
    varNameListOUT: list
        List of aliased names
    """
    if aliased:
        varNameListIN = _rename_aliased(od, varNameList)
    else:
        varNameListIN = varNameList
    varNameListOUT = varNameList
    return varNameListIN, varNameListOUT


def _rename_aliased(od, varNameList):
    """
    Check if there are aliases,
    and return the name of variables in the private dataset.
    This is used by smart-naming functions,
    where user asks for aliased variables.

    Parameters
    ----------
    od: OceanDataset
        oceandataset to check for missing variables
    varNameList: 1D array_like, str
        List of variables (strings).

    Returns
    -------
    varNameListIN: list of variables
        List of variable name to use on od._ds
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Check if input is a string
    if isinstance(varNameList, str):
        isstr = True
    else:
        isstr = False

    # Move to numpy array
    varNameList = _check_list_of_string(varNameList, 'varNameList')

    # Get _ds names
    if od._aliases_flipped is not None:
        varNameListIN = [od._aliases_flipped[varName]
                         if varName in od._aliases_flipped
                         else varName for varName in list(varNameList)]
    else:
        varNameListIN = varNameList

    # Same type of input
    if isstr:
        varNameListIN = varNameListIN[0]

    return varNameListIN


def _check_mean_and_int_axes(od, meanAxes, intAxes, exclude):
    """
    Check and return mean and integral axes
    when they can be used together (e.g., for plots).

    Parameters
    ----------
    od: OceanDataset
    meanAxes: list, bool, or str
    intAxes: list, bool, or str
    exclude: list
        List of axes to exclude

    Returns
    -------
    meanAxes: list
    intAxes: list
    """
    # Check type
    _check_instance({'meanAxes': meanAxes,
                     'intAxes': intAxes,
                     'exclude': exclude},
                    {'meanAxes': ['bool', 'list', 'str'],
                     'intAxes': ['bool', 'list', 'str'],
                     'exclude': 'list'})
    if not isinstance(meanAxes, bool):
        meanAxes = _check_list_of_string(meanAxes, 'meanAxes')
    if not isinstance(intAxes, bool):
        intAxes = _check_list_of_string(intAxes, 'intAxes')

    # Check both True
    check1 = (meanAxes is True and intAxes is not False)
    check2 = (intAxes is True and meanAxes is not False)
    if check1 or check2:
        raise ValueError('If one between `meanAxes` and `intAxes` is True,'
                         ' the other must be False')

    # Get axes to pass
    if meanAxes is True:
        meanAxes = [coord
                    for coord in od.grid_coords
                    if coord not in exclude]
    elif not isinstance(meanAxes, bool):
        if any([axis in exclude for axis in meanAxes]):
            raise ValueError('These axes can not be in `meanAxes`:'
                             ' {}'.format(exclude))

    if intAxes is True:
        intAxes = [coord
                   for coord in od.grid_coords
                   if coord not in exclude]
    elif not isinstance(intAxes, bool):
        if any([axis in exclude for axis in intAxes]):
            raise ValueError('These axes can not be in `intAxes`:'
                             ' {}'.format(exclude))

    return meanAxes, intAxes


def _rename_coord_attrs(ds):
    """
    to_netcdf and to_zarr don't like coordinates attribute

    Parameters
    ----------
    ds: xarray.Dataset

    Returns
    -------
    ds: xarray.Dataset
    """

    for var in ds.variables:
        attrs = ds[var].attrs
        coordinates = attrs.pop('coordinates', None)
        ds[var].attrs = attrs
        if coordinates is not None:
            ds[var].attrs['_coordinates'] = coordinates
    return ds


def _restore_coord_attrs(ds):
    """
    Put back coordinates attribute that
    to_netcdf and to_zarr didn't like.

    Parameters
    ----------
    ds: xarray.Dataset

    Returns
    -------
    ds: xarray.Dataset
    """

    for var in ds.variables:
        attrs = ds[var].attrs
        coordinates = attrs.pop('_coordinates', None)
        ds[var].attrs = attrs
        if coordinates is not None:
            ds[var].attrs['coordinates'] = coordinates
    return ds


# ========
# MESSAGES
# ========
def _check_part_position(od, InputDict):
    for InputName, InputField in InputDict.items():
        if 'time' in InputName:
            InputField = numpy.asarray(InputField, dtype=od._ds['time'].dtype)
            if InputField.ndim == 0:
                InputField = InputField.reshape(1)
            ndim = 1
        else:
            InputField = numpy.asarray(InputField)
            if InputField.ndim < 2 and InputField.size == 1:
                InputField = InputField.reshape((1, InputField.size))
            ndim = 2
        if InputField.ndim > ndim:
            raise TypeError('Invalid `{}`'.format(InputName))
        else:
            InputDict[InputName] = InputField
    return InputDict


def _check_ijk_components(od, iName=None, jName=None, kName=None):
    ds = od._ds
    for _, (Name, dim) in enumerate(zip([iName, jName, kName],
                                        ['Xp1', 'Yp1', 'Zl'])):
        if Name is not None and dim not in ds[Name].dims:
            raise ValueError('[{}] must have dimension [{}]'.format(Name, dim))


def _check_native_grid(od, func_name):
    wrong_dims = ['mooring', 'station', 'particle']
    for wrong_dim in wrong_dims:
        if wrong_dim in od._ds.dims:
            raise ValueError('`{}` cannot subsample {} oceandatasets'
                             ''.format(func_name, wrong_dims))


def _check_options(name, selected, options):
    if selected not in options:
        raise ValueError('`{}` [{}] not available.'
                         ' Options are: {}'.format(name, selected, options))


def _ax_warning(kwargs):
    ax = kwargs.pop('ax', None)
    if ax is not None:
        warnings.warn("\n`ax` can not be provided for animations. "
                      "This function will use the current axis", stacklevel=2)
    return kwargs


def _wrong_axes_error_message(axes2check):
    from oceanspy import OCEANSPY_AXES
    return ("{} contains non-valid axes."
            " OceanSpy axes are: {}").format(axes2check, OCEANSPY_AXES)


def _setter_error_message(attribute_name):
    return "Set new `{}` using .set_{}".format(attribute_name, attribute_name)
