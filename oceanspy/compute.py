"""
Create new variables using OceanDataset objects.
"""

# Instructions for developers:
# 1. All funcions must return a xr.Dataset.
# 2. All functions must operate on private objects of an od (_ds, _grid),
#    and use OceanSpy reference names.
# 3. Check that DataArrays used by functions are available
#    using _add_missing_variables.
# 4. Add new functions in _FUNC2VARS:
#    key is name of the function, value is list of new DataArrays.
# 5. Add new functions to _computeMethods
# 6. Add new functions to docs/api.rst

#############################################################
# TODO: when velocities are mutiplied by hfac,
#       we should use mass weighted velocities if available
# TODO: compute transport for survey
# TODO: compute velocity magnitude
#############################################################

# Required dependencies (private)
import xarray as _xr
import oceanspy as _ospy
import numpy as _np
import warnings as _warnings
import copy as _copy
import functools as _functools
from collections import OrderedDict as _OrderedDict

# From OceanSpy (private)
from . import utils as _utils
from ._ospy_utils import (_check_instance, _check_list_of_string,
                          _handle_aliased, _check_ijk_components)

# Hard coded  list of variables outputed by functions
_FUNC2VARS = _OrderedDict(potential_density_anomaly=['Sigma0'],
                          Brunt_Vaisala_frequency=['N2'],
                          vertical_relative_vorticity=['momVort3'],
                          velocity_magnitude=['vel'],
                          horizontal_velocity_magnitude=['hor_vel'],
                          relative_vorticity=['momVort1',
                                              'momVort2',
                                              'momVort3'],
                          kinetic_energy=['KE'],
                          eddy_kinetic_energy=['EKE'],
                          horizontal_divergence_velocity=['hor_div_vel'],
                          shear_strain=['s_strain'],
                          normal_strain=['n_strain'],
                          Okubo_Weiss_parameter=['Okubo_Weiss'],
                          Ertel_potential_vorticity=['Ertel_PV'],
                          mooring_volume_transport=['transport',
                                                    'Vtransport',
                                                    'Utransport',
                                                    'Y_transport',
                                                    'X_transport',
                                                    'Y_Utransport',
                                                    'X_Utransport',
                                                    'Y_Vtransport',
                                                    'X_Vtransport',
                                                    'dir_Utransport',
                                                    'dir_Vtransport'],
                          heat_budget=['tendH',
                                       'adv_hConvH',
                                       'adv_vConvH',
                                       'dif_vConvH',
                                       'kpp_vConvH',
                                       'forcH'],
                          salt_budget=['tendS',
                                       'adv_hConvS',
                                       'adv_vConvS',
                                       'dif_vConvS',
                                       'kpp_vConvS',
                                       'forcS'],
                          geographical_aligned_velocities=['U_zonal',
                                                           'V_merid'],
                          survey_aligned_velocities=['rot_ang_Vel',
                                                     'tan_Vel',
                                                     'ort_Vel'],
                          missing_horizontal_spacing=['dxF',
                                                      'dxV',
                                                      'dyF',
                                                      'dyU'])


def _add_missing_variables(od, varList, FUNC2VARS=_FUNC2VARS):
    """
    If any variable in varList is missing in the oceandataset,
    try to compute it.

    Parameters
    ----------
    od: OceanDataset
        oceandataset to check for missing variables.
    varList: 1D array_like, str
        List of variables to check (strings).
    FUNC2VARS: dict
        Dictionary that connect function names to computed variables.
        Keys are functions, values are list of variables.

    Returns
    -------
    od: OceanDataset
        oceandataset with new variables.
    """

    # Check parameters
    _check_instance({'od': od}, ' oceanspy.OceanDataset')
    varList = _check_list_of_string(varList, 'varList')

    # Return here if all variables already exist
    varList = [var
               for var in varList
               if var not in od._ds.variables]
    if len(varList) == 0:
        return od

    # Raise error if variable not availabe
    VAR2FUNC = {VAR: FUNC
                for FUNC in FUNC2VARS
                for VAR in FUNC2VARS[FUNC]}
    var_error = [var for var in varList if var not in VAR2FUNC]
    if len(var_error) != 0:
        raise ValueError('These variables are not available'
                         ' and can not be computed: {}.'
                         '\nIf you think that OceanSpy'
                         ' should be able to compute them,'
                         ' please open an issue on GitHub:'
                         '\n https://github.com/malmans2/oceanspy/issues'
                         ''.format(var_error))

    # Compute new variables
    funcList = list(set([VAR2FUNC[var] for var in varList if var in VAR2FUNC]))
    allds = []
    for func in funcList:
        allds = allds+[eval('{}(od)'.format(func))]
    ds = _xr.merge(allds)
    ds = ds.drop([var for var in ds.variables if var not in varList])

    # Merge to od
    od = od.merge_into_oceandataset(ds)

    return od


# ==========
# SMART-NAME
# ==========
def gradient(od, varNameList=None, axesList=None, aliased=True):
    """
    Compute gradient along specified axes, returning all terms (not summed).

    .. math::
        \\nabla \\chi =
        \\frac{\\partial \\chi}{\\partial x}\\mathbf{\\hat{x}}
        + \\frac{\\partial \\chi}{\\partial y}\\mathbf{\\hat{y}}
        + \\frac{\\partial \\chi}{\\partial z}\\mathbf{\\hat{z}}

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.
    varNameList: 1D array_like, str, None
        List of variables to differenciate.
        If None, use all variables.
    axesList: None, list
        List of axes. If None, compute gradient along all axes.
    aliased: bool
        Set it False when working with private ds and grid.

    Returns
    -------
    ds: xarray.Dataset
        | d[varName]_d[axis]

    References
    ----------
    Numerical Method:
    https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#notation

    See Also
    --------
    divergence
    curl
    laplacian
    """
    # TODO: We are assuming default units,
    #       while we should actually read from metadata.
    #       m for space, time in datetime64 format,
    #       and VSections distances in km.

    # Check parameters
    _check_instance({'od': od,
                     'aliased': aliased},
                    {'od': 'oceanspy.OceanDataset',
                     'aliased': 'bool'})

    varNameList = _check_list_of_string(varNameList, 'varNameList')
    if varNameList is None:
        varNameList = list(od.dataset.data_vars)

    if axesList is not None:
        axesList = _check_list_of_string(axesList, 'varNameList')

    grid_axes = [coord for coord in od.grid_coords]
    if axesList is None:
        axesList = grid_axes
    else:
        err_axes = [axis for axis in axesList if axis not in grid_axes]
        if len(err_axes) != 0:
            raise ValueError('{} are not available. '
                             'Available axes are {}'
                             ''.format(err_axes, grid_axes))

    # Handle aliases
    varNameListIN, varNameListOUT = _handle_aliased(od, aliased, varNameList)

    # Add missing variables
    od = _add_missing_variables(od, list(varNameListIN))

    # Message
    print('Computing gradient.')

    # Loop through variables
    grad = {}
    for _, (varName, varNameOUT) in enumerate(zip(varNameListIN,
                                                  varNameListOUT)):

        # Skip if variable doesn't have axis!
        for axis in axesList:
            S1 = set(od._ds[varName].dims)
            S2 = set([dim for dim in od.grid_coords[axis].keys()])
            if len(S1.intersection(S2)) == 0:
                continue

            # Numerator
            dnum = od._grid.diff(od._ds[varName], axis,
                                 boundary='fill', fill_value=_np.nan)

            # Horizontal gradient
            if axis in ['X', 'Y']:
                # Select denominator
                pointList = ['C', 'F', 'G']
                if axis == 'X':
                    # Add missing variables
                    varList = ['dxC', 'dxF', 'dxG', 'dxV']
                    od = _add_missing_variables(od, varList)
                    pointList = pointList+['V']
                else:
                    # Add missing variables
                    varList = ['dyC', 'dyF', 'dyG', 'dyU']
                    od = _add_missing_variables(od, varList)
                    pointList = pointList+['U']
                ddenNames = ['d'+axis.lower()+point for point in pointList]
                for ddenName in ddenNames:
                    if set(od._ds[ddenName].dims).issubset(dnum.dims):
                        dden = od._ds[ddenName]
                        continue

            # Vertical gradient
            if axis == 'Z':

                # Add missing variables
                varList = ['HFacC', 'HFacW', 'HFacS']
                od = _add_missing_variables(od, varList)

                # Extract HFac
                if set(['X', 'Y']).issubset(dnum.dims):
                    HFac = od._ds['HFacC']
                elif set(['Xp1', 'Y']).issubset(dnum.dims):
                    HFac = od._ds['HFacW']
                elif set(['X', 'Yp1']).issubset(dnum.dims):
                    HFac = od._ds['HFacS']
                elif set(['Xp1', 'Yp1']).issubset(dnum.dims):
                    grid = od._grid
                    HFac = grid.interp(grid.interp(od._ds['HFacC'], 'X',
                                                   boundary='extend'),
                                       'Y', boundary='extend')
                else:
                    HFac = 1

                # Don't use dr, but compute
                grid = od._grid
                coords = grid.axes[axis].coords
                for dim in [coords[coord] for coord in coords]:
                    if dim in od._ds[varName].dims:
                        dden = grid.diff(od._ds[dim], axis,
                                         boundary='fill', fill_value=_np.nan)
                        if not isinstance(HFac, int):
                            for coord in coords:
                                if coords[coord] in dden.dims:
                                    if coord != 'center':
                                        HFac = grid.interp(HFac, axis,
                                                           to=coord,
                                                           boundary='fill',
                                                           fill_value=_np.nan)
                                continue
                        # Apply HFac
                        dden = dden * HFac
                        continue

            # Time and vertical sections
            if axis in ['mooring', 'station', 'time']:

                if axis in ['mooring', 'station']:
                    convert_units = 1.E-3
                    add_dist = '_dist'
                else:
                    convert_units = _np.timedelta64(1, 's')
                    add_dist = ''

                if axis in dnum.dims:
                    dden = od._grid.diff(od._ds[axis+'_midp'+add_dist], axis,
                                         boundary='fill', fill_value=_np.nan)
                else:
                    # midp
                    dden = od._grid.diff(od._ds[axis+add_dist], axis,
                                         boundary='fill', fill_value=_np.nan)
                dden = dden / convert_units

            # Add and clear
            outName = 'd'+varNameOUT+'_d'+axis
            grad[outName] = dnum / dden
            add_units = {'X': ' m',
                         'Y': ' m',
                         'Z': ' m',
                         'time': ' s',
                         'station': ' m',
                         'mooring': ' m'}
            if 'units' in od._ds[varName].attrs:
                units = od._ds[varName].attrs['units']+add_units[axis]+'^-1'
                grad[outName].attrs['units'] = units

            del dnum, dden

    return _xr.Dataset(grad, attrs=od.dataset.attrs)


def divergence(od, iName=None, jName=None, kName=None, aliased=True):
    """
    Compute divergence of a vector field.

    .. math::
        \\nabla \\cdot {\\bf F} =
        \\frac{\\partial F_x}{\\partial x}
        + \\frac{\\partial F_y}{\\partial y}
        + \\frac{\\partial F_z}{\\partial z}

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.
    iName: str or None
        Name of variable corresponding to i-component.
    jName: str or None
        Name of variable corresponding to j-component.
    kName: str or None
        Name of variable corresponding to k-component.
    aliased: bool
        Set it False when working with private ds and grid.

    Returns
    -------
    ds: xarray.Dataset
        | d[varName]_d[axis]

    References
    ----------
    Numerical Method:
    https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#notation

    See Also
    --------
    gradient
    curl
    laplacian
    """

    # Check parameters
    _check_instance({'od': od,
                     'iName': iName,
                     'jName': jName,
                     'kName': kName,
                     'aliased': aliased},
                    {'od': 'oceanspy.OceanDataset',
                     'iName': '(str, type(None))',
                     'jName': '(str, type(None))',
                     'kName': '(str, type(None))',
                     'aliased': 'bool'})

    if iName == jName == kName is None:
        raise ValueError('At least 1 component must be provided.')

    # Message
    print('Computing divergence.')

    div = {}
    pref = 'd'
    if iName is not None:

        # Handle aliases
        NameIN, NameOUT = _handle_aliased(od, aliased, iName)

        # Add missing variables
        varList = ['HFacC', 'rA', 'dyG', 'HFacW', NameIN]
        od = _add_missing_variables(od, varList)

        # Check components
        _check_ijk_components(od, iName=NameIN)

        # Add div
        suf = '_dX'
        grid = od._grid
        diff = grid.diff(od._ds[NameIN]*od._ds['HFacW']*od._ds['dyG'], suf[-1],
                         boundary='fill', fill_value=_np.nan)
        div[pref+NameOUT+suf] = diff / (od._ds['HFacC'] * od._ds['rA'])

        # Units
        if 'units' in od._ds[NameIN].attrs:
            units = od._ds[NameIN].attrs['units']+' m^-1'
            div[pref+NameOUT+suf].attrs['units'] = units

    if jName is not None:

        # Handle aliases
        NameIN, NameOUT = _handle_aliased(od, aliased, jName)

        # Add missing variables
        varList = ['HFacC', 'rA', 'dxG', 'HFacS', NameIN]
        od = _add_missing_variables(od, varList)

        # Check components
        _check_ijk_components(od, jName=NameIN)

        # Add div
        suf = '_dY'
        grid = od._grid
        diff = grid.diff(od._ds[NameIN]*od._ds['HFacS']*od._ds['dxG'], suf[-1],
                         boundary='fill', fill_value=_np.nan)
        div[pref+NameOUT+suf] = diff / (od._ds['HFacC'] * od._ds['rA'])
        # Units
        if 'units' in od._ds[NameIN].attrs:
            units = od._ds[NameIN].attrs['units']+' m^-1'
            div[pref+NameOUT+suf].attrs['units'] = units

    if kName is not None:

        # Handle aliases
        NameIN, NameOUT = _handle_aliased(od, aliased, kName)

        # Add missing variables
        od = _add_missing_variables(od, NameIN)

        # Check components
        _check_ijk_components(od, kName=NameIN)

        # Add div (same of gradient)
        suf = '_dZ'
        div[pref+NameOUT+suf] = gradient(od, varNameList=NameIN,
                                         axesList=suf[-1],
                                         aliased=False)[pref+NameIN+suf]

        # Units
        if 'units' in od._ds[NameIN].attrs:
            units = od._ds[NameIN].attrs['units'] + ' m^-1'
            div[pref+NameOUT+suf].attrs['units'] = units

    return _xr.Dataset(div, attrs=od.dataset.attrs)


def curl(od, iName=None, jName=None, kName=None, aliased=True):
    """
    Compute curl of a vector field.

    .. math::
        \\nabla \\times {\\bf F} =
        \\left( \\frac{\\partial F_z}{\\partial y}
        - \\frac{\\partial F_y}{\\partial z} \\right)\\mathbf{\\hat{x}}
        + \\left( \\frac{\\partial F_x}{\\partial z}
        - \\frac{\\partial F_z}{\\partial x} \\right)\\mathbf{\\hat{y}}
        + \\left( \\frac{\\partial F_y}{\\partial x}
        - \\frac{\\partial F_x}{\\partial y} \\right)\\mathbf{\\hat{z}}

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.
    iName: str or None
        Name of variable corresponding to i-component.
    jName: str or None
        Name of variable corresponding to j-component.
    kName: str or None
        Name of variable corresponding to k-component.
    aliased: bool
        Set it False when working with private ds and grid.

    Returns
    -------
    ds: xarray.Dataset
        | d[jName]_dX-d[iName]_dY
        | d[kName]_dY-d[jName]_dZ
        | d[iName]_dZ-d[kName]_dX

    References
    ----------
    Numerical Method:
    https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#notation

    See Also
    --------
    gradient
    divergence
    laplacian
    """

    # Check parameters
    _check_instance({'od': od,
                     'iName': iName,
                     'jName': jName,
                     'kName': kName,
                     'aliased': aliased},
                    {'od': 'oceanspy.OceanDataset',
                     'iName': '(str, type(None))',
                     'jName': '(str, type(None))',
                     'kName': '(str, type(None))',
                     'aliased': 'bool'})
    if sum(x is None for x in [iName, jName, kName]) >= 2:
        raise ValueError('At least 2 out of 3 components must be provided.')

    # Message
    print('Computing curl.')

    crl = {}
    if iName is not None and jName is not None:

        # Handle aliases
        iNameIN, iNameOUT = _handle_aliased(od, aliased, iName)
        jNameIN, jNameOUT = _handle_aliased(od, aliased, jName)

        # Add missing variables
        varList = ['rAz', 'dyC', 'dxC', iNameIN, jNameIN]
        od = _add_missing_variables(od, varList)

        # Check components
        _check_ijk_components(od, iName=iNameIN, jName=jNameIN)

        # Add curl
        Name = 'd'+jNameOUT+'_dX-d'+iNameOUT+'_dY'
        grid = od._grid
        crl[Name] = (grid.diff(od._ds[jNameIN]*od._ds['dyC'], 'X',
                               boundary='fill', fill_value=_np.nan)
                     - grid.diff(od._ds[iNameIN]*od._ds['dxC'], 'Y',
                                 boundary='fill', fill_value=_np.nan))
        crl[Name] = crl[Name] / od._ds['rAz']

        # Units
        checks = ['units' in od._ds[iNameIN].attrs,
                  'units' in od._ds[jNameIN].attrs]
        if all(checks) and (od._ds[iNameIN].attrs['units']
                            == od._ds[jNameIN].attrs['units']):
            crl[Name].attrs['units'] = od._ds[iNameIN].attrs['units'] + ' m^-1'

    if jName is not None and kName is not None:

        # Handle aliases
        jNameIN, jNameOUT = _handle_aliased(od, aliased, jName)
        kNameIN, kNameOUT = _handle_aliased(od, aliased, kName)

        # Add missing variables
        varList = [jNameIN, kNameIN]
        od = _add_missing_variables(od, varList)

        # Check components
        _check_ijk_components(od, jName=jNameIN, kName=kNameIN)

        # Add curl using gradients
        Name = 'd'+kNameOUT+'_dY-d'+jNameOUT+'_dZ'
        crl[Name] = (gradient(od, kNameIN, 'Y',
                              aliased=False)['d'+kNameIN+'_dY']
                     - gradient(od, jNameIN, 'Z',
                                aliased=False)['d'+jNameIN+'_dZ'])

        # Units
        checks = ['units' in od._ds[jNameIN].attrs,
                  'units' in od._ds[kNameIN].attrs]
        if all(checks) and (od._ds[jNameIN].attrs['units']
                            == od._ds[kNameIN].attrs['units']):
            crl[Name].attrs['units'] = od._ds[jNameIN].attrs['units']+' m^-1'

    if kName is not None and iName is not None:

        # Handle aliases
        iNameIN, iNameOUT = _handle_aliased(od, aliased, iName)
        kNameIN, kNameOUT = _handle_aliased(od, aliased, kName)

        # Add missing variables
        varList = [iNameIN, kNameIN]
        od = _add_missing_variables(od, varList)

        # Check components
        _check_ijk_components(od, iName=iNameIN, kName=kNameIN)

        # Add curl using gradients
        Name = 'd'+iNameOUT+'_dZ-d'+kNameOUT+'_dX'
        crl[Name] = (gradient(od, iNameIN, 'Z',
                              aliased=False)['d'+iNameIN+'_dZ']
                     - gradient(od, kNameIN, 'X',
                                aliased=False)['d'+kNameIN+'_dX'])

        # Units
        checks = ['units' in od._ds[iNameIN].attrs,
                  'units' in od._ds[kNameIN].attrs]
        if all(checks) and (od._ds[iNameIN].attrs['units']
                            == od._ds[kNameIN].attrs['units']):
            crl[Name].attrs['units'] = od._ds[iNameIN].attrs['units']+' m^-1'

    return _xr.Dataset(crl, attrs=od.dataset.attrs)


def laplacian(od, varNameList=None, axesList=None, aliased=True):
    """
    Compute laplacian along specified axis

    .. math::
        \\nabla^2 \\chi =
        \\nabla \\cdot \\nabla \\chi

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.
    varNameList: 1D array_like, str
        Name of variables to differenciate.
        If None, use all variables.
    axesList: None, list
        List of axes. If None, compute gradient along all space axes.
    aliased: bool
        Set it False when working with private ds and grid.

    Returns
    -------
    ds: xarray.Dataset
        | dd[varName]_d[axis]_ds[axis]

    References
    ----------
    Numerical Method:
    https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#notation

    See Also
    --------
    gradient
    divergence
    curl
    """

    # Check parameters
    _check_instance({'od': od,
                     'aliased': aliased},
                    {'od': 'oceanspy.OceanDataset',
                     'aliased': 'bool'})

    varNameList = _check_list_of_string(varNameList, 'varNameList')

    if axesList is not None:
        axesList = _check_list_of_string(axesList, 'varNameList')

    good_axes = ['X', 'Y', 'Z']
    grid_axes = [coord for coord in od.grid_coords if coord in good_axes]
    if axesList is None:
        axesList = grid_axes
    else:
        err_axes = [axis for axis in axesList if axis not in grid_axes]
        if len(err_axes) != 0:

            raise ValueError('These axes are not supported: {}.'
                             '\nThe laplacian operator is'
                             ' currently implemented '
                             'for the following axes only: {}'
                             ''.format(err_axes, good_axes))

    # Handle aliases
    varNameListIN, varNameListOUT = _handle_aliased(od, aliased, varNameList)

    # Check scalars
    for axis in list(grid_axes):
        for _, (varIN, varOUT) in enumerate(zip(list(varNameListIN),
                                                list(varNameListOUT))):
            if axis not in od._ds[varIN].dims:
                raise ValueError('[{}] has wrong dimensions.'
                                 '\nThe laplacian operator'
                                 ' along the axis [{}]'
                                 ' currently only supports'
                                 ' variables with dimension [{}].'
                                 ''.format(varOUT, axis, axis))

    # Message
    print('Computing laplacian.')

    # Loop through variables
    lap = []
    for _, (varName, varNameOUT) in enumerate(zip(varNameListIN,
                                                  varNameListOUT)):

        # Compute gradients
        grad = gradient(od,
                        varNameList=varName, axesList=axesList, aliased=False)

        # Add to od
        od = _copy.copy(od)
        attrs = od._ds.attrs
        od._ds = _xr.merge([od._ds, grad])
        od._ds.attrs = attrs

        # Compute laplacian
        compNames = {}
        rename_dict = {}
        for compName in grad.variables:
            if compName in grad.coords:
                continue
            elif compName[-1] == 'X':
                compNames['iName'] = compName
                nameA = 'dd{}_dX_dX'.format(varName)
                nameB = 'dd{}_dX_dX'.format(varNameOUT)
                rename_dict[nameA] = nameB
            elif compName[-1] == 'Y':
                compNames['jName'] = compName
                nameA = 'dd{}_dY_dY'.format(varName)
                nameB = 'dd{}_dY_dY'.format(varNameOUT)
                rename_dict[nameA] = nameB
            else:
                # Z
                compNames['kName'] = compName
                nameA = 'dd{}_dZ_dZ'.format(varName)
                nameB = 'dd{}_dZ_dZ'.format(varNameOUT)
                rename_dict[nameA] = nameB

        div = divergence(od, **compNames, aliased=False).rename(rename_dict)
        lap = lap + [div]

    # Merge
    ds = _xr.merge(lap)
    ds.attrs = od.dataset.attrs

    return ds


def weighted_mean(od,
                  varNameList=None, axesList=None,
                  storeWeights=True, aliased=True):
    """
    Compute weighted means using volumes, surfaces, or distances.

    .. math::
        \\overline{\\chi} =
        \\frac{\\sum_{i=1}^n w_i\\chi_i}{\\sum_{i=1}^n w_i}

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.
    varNameList: 1D array_like, str, or None
        Name of variables to average.
        If None, use all variables.
    axesList: None, list
        List of axes. If None, compute average along all axes
        (excluding mooring and station).
    storeWeights: bool
        True: store weight values.
        False: drop weight values.
    aliased: bool
        Set it False when working with private ds and grid.

    Returns
    -------
    ds: xarray.Dataset
        | w_mean_[varName]
        | weight_[varName]

    See Also
    --------
    integral
    """

    return _integral_and_mean(od,
                              varNameList=varNameList,
                              axesList=axesList,
                              storeWeights=storeWeights,
                              aliased=aliased,
                              operation='weighted_mean')


def integral(od, varNameList=None, axesList=None, aliased=True):
    """
    Compute integrals along specified axes (simple discretization).

    .. math::
        I =
        \\int \\cdots \\int \\chi \\; d x_1 \\cdots d x_n

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.
    varNameList: 1D array_like, str, or None
        Name of variables to integrate.
        If None, use all variables.
    axesList: None, list
        List of axes. If None, compute integral along all axes.
    aliased: bool
        Set it False when working with private ds and grid.

    Returns
    -------
    ds: xarray.Dataset
        | I([varName])d[axis]

    See Also
    --------
    weighted_mean
    """

    return _integral_and_mean(od,
                              varNameList=varNameList,
                              axesList=axesList,
                              aliased=aliased,
                              operation='integral')


def _integral_and_mean(od, operation='integral',
                       varNameList=None, axesList=None,
                       aliased=True, storeWeights=True):

    # Check parameters
    _check_instance({'od': od,
                     'aliased': aliased,
                     'storeWeights': storeWeights},
                    {'od': 'oceanspy.OceanDataset',
                     'aliased': 'bool',
                     'storeWeights': 'bool'})

    varNameList = _check_list_of_string(varNameList, 'varNameList')
    if varNameList is None:
        varNameList = list(od.dataset.data_vars)

    if axesList is not None:
        axesList = _check_list_of_string(axesList, 'varNameList')

    grid_axes = [coord for coord in od.grid_coords]
    if axesList is None:
        axesList = grid_axes
    else:
        err_axes = [axis for axis in axesList if axis not in grid_axes]
        if len(err_axes) != 0:
            raise ValueError('{} are not available.'
                             ' Available axes are {}'
                             ''.format(err_axes, grid_axes))

    # Handle aliases
    varNameListIN, varNameListOUT = _handle_aliased(od, aliased, varNameList)

    # Add missing variables
    od = _add_missing_variables(od, list(varNameListIN))

    # Message
    print('Computing {}.'.format(operation))

    # Loop through variables
    to_return = {}
    for _, (varName, varNameOUT) in enumerate(zip(varNameListIN,
                                                  varNameListOUT)):
        delta = 1
        suf = []
        units = []
        dims2sum = []
        attrs = od._ds[varName].attrs
        Int = od._ds[varName]

        # ====
        # TIME
        # ====
        if set(['time']).issubset(axesList):
            for t in ['time', 'time_midp']:
                if t in Int.dims:
                    grid = od._grid
                    diff = grid.diff(od._ds[t], 'time', boundary='extend')
                    interp = grid.interp(diff / _np.timedelta64(1, 's'),
                                         'time', boundary='extend')
                    delta = delta * interp
                    dims2sum = dims2sum + list(od._ds[t].dims)
                    suf = suf + ['dtime']
                    units = units + ['s']

        # =======
        # MOORING
        # =======
        if set(['mooring']).issubset(axesList):
            for m in ['mooring', 'mooring_midp']:
                if m in Int.dims:
                    grid = od._grid
                    diff = grid.diff(od._ds[m+'_dist'], 'mooring',
                                     boundary='extend')
                    interp = grid.interp(diff / 1.E-3, 'mooring',
                                         boundary='extend')
                    delta = delta * interp
                    dims2sum = dims2sum + list(od._ds[m+'_dist'].dims)
                    suf = suf + ['dmoor']
                    units = units + ['m']

        # =======
        # STATION
        # =======
        if set(['station']).issubset(axesList):
            for s in ['station', 'station_midp']:
                if s in Int.dims:
                    grid = od._grid
                    diff = grid.diff(od._ds[s+'_dist'], 'station',
                                     boundary='extend')
                    interp = grid.interp(diff / 1.E-3, 'station',
                                         boundary='extend')
                    delta = delta * interp
                    dims2sum = dims2sum + list(od._ds[s+'_dist'].dims)
                    suf = suf + ['dstat']
                    units = units + ['m']

        # ==========
        # HORIZONTAL
        # ==========
        # Area
        if set(['X', 'Y']).issubset(axesList):

            # Add missing variables
            areaList = ['rA', 'rAw', 'rAs', 'rAz']
            od = _add_missing_variables(od, areaList)

            for area in areaList:
                if set(od._ds[area].dims).issubset(Int.dims):
                    delta = delta * od._ds[area]
                    dims2sum = dims2sum + [dim
                                           for dim in od._ds[area].dims
                                           if dim[0] == 'Y' or dim[0] == 'X']
                    suf = suf + ['dXdY']
                    units = units + ['m^2']
                    continue

        # Y
        elif set(['Y']).issubset(axesList):

            # Add missing variables
            yList = ['dyC', 'dyF', 'dyG', 'dyU']
            od = _add_missing_variables(od, yList)

            for y in yList:
                if set(od._ds[y].dims).issubset(Int.dims):
                    delta = delta * od._ds[y]
                    dims2sum = dims2sum + [dim
                                           for dim in od._ds[y].dims
                                           if dim[0] == 'Y']
                    suf = suf + ['dY']
                    units = units + ['m']
                    continue

        # X
        elif set(['X']).issubset(axesList):

            # Add missing variables
            xList = ['dxC', 'dxF', 'dxG', 'dxV']
            od = _add_missing_variables(od, xList)

            for x in xList:
                if set(od._ds[x].dims).issubset(Int.dims):
                    delta = delta * od._ds[x]
                    dims2sum = dims2sum + [dim
                                           for dim in od._ds[x].dims
                                           if dim[0] == 'X']
                    suf = suf + ['dX']
                    units = units + ['m']
                    continue

        # ========
        # VERTICAL
        # ========
        if set(['Z']).issubset(axesList):

            # Add missing variables
            HFacList = ['HFacC', 'HFacW', 'HFacS']
            od = _add_missing_variables(od, HFacList)

            # Extract HFac
            if set(['X', 'Y']).issubset(Int.dims):
                HFac = od._ds['HFacC']
            elif set(['Xp1', 'Y']).issubset(Int.dims):
                HFac = od._ds['HFacW']
            elif set(['X', 'Yp1']).issubset(Int.dims):
                HFac = od._ds['HFacS']
            elif set(['Xp1', 'Yp1']).issubset(Int.dims):
                HFac = od._grid.interp(od._grid.interp(od._ds['HFacC'], 'X',
                                                       boundary='extend'),
                                       'Y', boundary='extend')
            else:
                HFac = None

            # Add missing variables
            zList = ['drC', 'drF']
            od = _add_missing_variables(od, zList)

            foundZ = False
            for z in zList:
                if set(od._ds[z].dims).issubset(Int.dims):
                    if z == 'drC' and HFac is not None:
                        HFac = od.grid.interp(HFac, 'Z', to='outer',
                                              boundary='extend')
                    if HFac is None:
                        HFac = 1
                    delta = delta * od._ds[z] * HFac
                    dims2sum = dims2sum + list(od._ds[z].dims)
                    suf = suf + ['dZ']
                    units = units + ['m']
                    foundZ = True
                    continue

            if foundZ is False:
                for coord in od._grid.axes['Z'].coords:
                    z = od._grid.axes['Z'].coords[coord]
                    if set([z]).issubset(Int.dims):
                        dr = od.grid.interp(od.dataset['drF'], 'Z', to=coord,
                                            boundary='extend')
                        if HFac is not None:
                            HFac = od._grid.interp(HFac, 'Z', to=coord,
                                                   boundary='extend')
                        else:
                            HFac = 1
                        delta = delta * dr * HFac
                        dims2sum = dims2sum + list(dr.dims)
                        suf = suf + ['dZ']
                        units = units + ['m']
                        foundZ = True
                        continue

        dims2sum = list(set(dims2sum))
        dims2ave = dims2sum
        if operation == 'integral':
            # Compute integral
            Int = (Int * delta).sum(dims2sum)
            to_return['I('+varNameOUT+')'+''.join(suf)] = Int
            if 'units' in attrs:
                Int.attrs['units'] = attrs['units'] + '*' + '*'.join(units)
            else:
                Int.attrs['units'] = '*'.join(units)

        else:
            # Weighted mean
            # Compute weighted mean
            wMean = Int
            weight = delta
            if not isinstance(weight, int):
                # Keep dimensions in the right order
                weight = weight.where(wMean.notnull())
                weight = weight.transpose(*[dim
                                            for dim in wMean.dims
                                            if dim in weight.dims])
                wMean = ((wMean * weight).sum(dims2ave))
                wMean = wMean / (weight.sum(dims2ave))

            # Store wMean
            wMean.attrs = attrs
            to_return['w_mean_'+varNameOUT] = wMean
            if storeWeights is True and not isinstance(weight, int):
                weight.attrs['long_name'] = 'Weights for average'
                weight.attrs['units'] = '*'.join(units)
                to_return['weight_'+varNameOUT] = weight

    return _xr.Dataset(to_return, attrs=od.dataset.attrs)


# ================
# FIXED-NAME
# ================
def potential_density_anomaly(od):
    """
    Compute potential density anomaly.

    .. math::
        \\sigma_\\theta =
        \\rho \\left(S, \\theta, \\text{pressure} = 0 \\text{ db} \\right)
        -1000\\text{ kgm}^{-3}

    Parameters used:
        | eq_state

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.

    Returns
    -------
    ds: xarray.Dataset
        | Sigma0: potential density anomaly

    See Also
    --------
    Brunt_Vaisala_frequency
    utils.densjmd95
    utils.densmdjwf
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Parameters
    paramsList = ['eq_state']
    params2use = {par: od.parameters[par]
                  for par in od.parameters
                  if par in paramsList}

    # Add missing variables
    varList = ['Temp', 'S']
    od = _add_missing_variables(od, varList)

    # Message
    print('Computing potential density anomaly'
          ' using the following parameters: {}.'.format(params2use))

    # Create DataArray
    Sigma0 = eval("_utils.dens{}(od._ds['S'], od._ds['Temp'], 0)-1000"
                  "".format(params2use['eq_state']))
    Sigma0.attrs['units'] = 'kg/m^3'
    Sigma0.attrs['long_name'] = 'potential density anomaly'
    Sigma0.attrs['OceanSpy_parameters'] = str(params2use)

    # Create ds
    ds = _xr.Dataset({'Sigma0': Sigma0}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def Brunt_Vaisala_frequency(od):
    """
    Compute Brunt-Väisälä Frequency.

    .. math::
        N^2 = -\\frac{g}{\\rho_0}\\frac{\\partial\\sigma_0}{\\partial z}

    Parameters used:
        | g
        | rho0

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.

    Returns
    -------
    ds: xarray.Dataset
        | N2: Brunt-Väisälä Frequency

    See Also
    --------
    potential_density_anomaly
    gradient
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Add missing variables
    varList = ['Sigma0']
    od = _add_missing_variables(od, varList)

    # Parameters
    paramsList = ['g', 'rho0']
    params2use = {par: od.parameters[par]
                  for par in od.parameters
                  if par in paramsList}

    # Extract parameters
    g = od.parameters['g']
    rho0 = od.parameters['rho0']

    # Message
    print('Computing Brunt-Väisälä Frequency'
          ' using the following parameters: {}.'.format(params2use))

    # Create DataArray
    grad = gradient(od, varNameList='Sigma0', axesList='Z', aliased=False)
    N2 = - g / rho0 * grad['dSigma0_dZ']
    N2.attrs['units'] = 's^-2'
    N2.attrs['long_name'] = 'Brunt-Väisälä Frequency'
    N2.attrs['OceanSpy_parameters'] = str(params2use)

    # Create ds
    ds = _xr.Dataset({'N2': N2}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def velocity_magnitude(od):
    """
    Compute velocity magnitude.

    .. math::
        ||\\mathbf{u}||=
        \\left(u^2+v^2+w^2\\right)^{1/2}

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.

    Returns
    -------
    ds: xarray.Dataset
        | hor_vel: velocity magnitude

    See Also
    --------
    horizontal_velocity_magnitude
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Add missing variables
    varList = ['U', 'V']
    od = _add_missing_variables(od, varList)

    # Extract variables
    U = od._ds['U']
    V = od._ds['V']
    W = od._ds['W']

    # Extract grid
    grid = od._grid

    # Message
    print('Computing velocity magnitude')

    # Interpolate horizontal velocities
    U = grid.interp(U, 'X', boundary='fill', fill_value=_np.nan)
    V = grid.interp(V, 'Y', boundary='fill', fill_value=_np.nan)
    W = grid.interp(W, 'Z', to='center',
                    boundary='fill', fill_value=_np.nan)

    # Compute horizontal velocity magnitude
    vel = _np.sqrt(_np.power(U, 2) + _np.power(V, 2)
                   + _np.power(W, 2))

    # Create DataArray
    vel.attrs['units'] = 'm/s'
    vel.attrs['long_name'] = 'velocity magnitude'

    # Create ds
    ds = _xr.Dataset({'vel': vel}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def horizontal_velocity_magnitude(od):
    """
    Compute magnitude of horizontal velocity.

    .. math::
        ||\\mathbf{u}_H||=
        \\left(u^2+v^2\\right)^{1/2}

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.

    Returns
    -------
    ds: xarray.Dataset
        | hor_vel: magnitude of horizontal velocity

    See Also
    --------
    velocity_magnitude
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Add missing variables
    varList = ['U', 'V']
    od = _add_missing_variables(od, varList)

    # Extract variables
    U = od._ds['U']
    V = od._ds['V']

    # Extract grid
    grid = od._grid

    # Message
    print('Computing magnitude of horizontal velocity')

    # Interpolate horizontal velocities
    U = grid.interp(U, 'X', boundary='fill', fill_value=_np.nan)
    V = grid.interp(V, 'Y', boundary='fill', fill_value=_np.nan)

    # Compute horizontal velocity magnitude
    hor_vel = _np.sqrt(_np.power(U, 2) + _np.power(V, 2))

    # Create DataArray
    hor_vel.attrs['units'] = 'm/s'
    hor_vel.attrs['long_name'] = 'magnitude of horizontal velocity'

    # Create ds
    ds = _xr.Dataset({'hor_vel': hor_vel}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def vertical_relative_vorticity(od):
    """
    Compute vertical component of relative  vorticity.

    .. math::
        \\zeta =
        \\frac{\\partial v}{\\partial x}
        -\\frac{\\partial u}{\\partial y}

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.

    Returns
    -------
    ds: xarray.Dataset
        | momVort3: vertical component of relative vorticity

    See Also
    --------
    relative_vorticity
    curl
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Message
    print('Computing vertical component of relative vorticity.')

    # Create DataArray
    crl = curl(od, iName='U', jName='V', kName=None, aliased=False)
    momVort3 = crl['dV_dX-dU_dY']
    momVort3.attrs['units'] = 's^-1'
    momVort3.attrs['long_name'] = 'vertical component of relative vorticity'

    # Create ds
    ds = _xr.Dataset({'momVort3': momVort3}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def relative_vorticity(od):
    """
    Compute relative vorticity.

    .. math::
        {\\bf \\omega} =
        \\left( \\zeta_H, \\zeta \\right) =
        \\nabla \\times {\\bf u}

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.

    Returns
    -------
    ds: xarray.Dataset
        | momVort1: i-component of relative vorticity
        | momVort2: j-component of relative vorticity
        | momVort3: k-component of relative vorticity

    See Also
    --------
    vertical_relative_vorticity
    curl
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Message
    print('Computing relative vorticity.')

    # Create DataArray
    ds = curl(od, iName='U', jName='V', kName='W', aliased=False)
    for _, (orig, out, comp) in enumerate(zip(['dV_dX-dU_dY',
                                               'dW_dY-dV_dZ',
                                               'dU_dZ-dW_dX'],
                                              ['momVort3',
                                               'momVort1',
                                               'momVort2'],
                                              ['k',
                                               'i',
                                               'j'])):
        ds = ds.rename({orig: out})
        long_name = '{}-component of relative vorticity'.format(comp)
        ds[out].attrs['long_name'] = long_name
        ds[out].attrs['units'] = 's^-1'

    return _ospy.OceanDataset(ds).dataset


def kinetic_energy(od):
    """
    Compute kinetic energy.

    .. math::
         KE =
         \\frac{1}{2}\\left(
         u^2 + v^2
         + \\epsilon_{nh} w^2\\right)

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.

    Parameters used:
        | eps_nh

    Returns
    -------
    ds: xarray.Dataset
        | KE: kinetic energy

    References
    ----------
    Numerical Method:
    https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#kinetic-energy

    See Also
    --------
    eddy_kinetic_energy
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Add missing variables
    varList = ['U', 'V']
    od = _add_missing_variables(od, varList)

    # Parameters
    paramsList = ['eps_nh']
    params2use = {par: od.parameters[par]
                  for par in od.parameters
                  if par in paramsList}

    # Extract variables
    U = od._ds['U']
    V = od._ds['V']

    # Extract grid
    grid = od._grid

    # Extract parameters
    eps_nh = od.parameters['eps_nh']

    # Message
    print('Computing kinetic energy'
          ' using the following parameters: {}.'.format(params2use))

    # Interpolate horizontal velocities
    U = grid.interp(U, 'X', boundary='fill', fill_value=_np.nan)
    V = grid.interp(V, 'Y', boundary='fill', fill_value=_np.nan)

    # Sum squared values
    sum2 = _np.power(U, 2) + _np.power(V, 2)

    # Non-hydrostatic case
    if eps_nh:
        # Add missing variables
        varList = ['W']
        od = _add_missing_variables(od, varList)

        # Extract variables
        W = od._ds['W']

        # Interpolate vertical velocity
        W = grid.interp(W, 'Z', to='center',
                        boundary='fill', fill_value=_np.nan)

        # Sum squared values
        sum2 = sum2 + eps_nh * _np.power(W, 2)

    # Create DataArray
    KE = sum2 / 2
    KE.attrs['units'] = 'm^2 s^-2'
    KE.attrs['long_name'] = 'kinetic energy'
    KE.attrs['OceanSpy_parameters'] = str(params2use)

    # Create ds
    ds = _xr.Dataset({'KE': KE}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def eddy_kinetic_energy(od):
    """
    Compute eddy kinetic energy.

    .. math::
        EKE = \\frac{1}{2}\\left[
        (u-\\overline{u})^2
        + (v-\\overline{v})^2
        + \\epsilon_{nh} (w-\\overline{w})^2 \\right]

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.

    Parameters used:
        | eps_nh

    Returns
    -------
    ds: xarray.Dataset
        | EKE: eddy kinetic energy

    References
    ----------
    Numerical Method:
    https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#kinetic-energy

    See Also
    --------
    kinetic_energy
    weighted_mean
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Add missing variables
    varList = ['U', 'V']
    od = _add_missing_variables(od, varList)

    # Parameters
    paramsList = ['eps_nh']
    params2use = {par: od.parameters[par]
                  for par in od.parameters
                  if par in paramsList}

    # Extract variables
    U = od._ds['U']
    V = od._ds['V']

    # Compute anomalies
    Umean = weighted_mean(od, 'U', 'time', False)
    Vmean = weighted_mean(od, 'V', 'time', False)
    U = U - Umean['w_mean_U']
    V = V - Vmean['w_mean_V']

    # Extract grid
    grid = od._grid

    # Extract parameters
    eps_nh = od.parameters['eps_nh']

    # Message
    print('Computing kinetic energy'
          ' using the following parameters: {}.'.format(params2use))

    # Interpolate horizontal velocities
    U = grid.interp(U, 'X', boundary='fill', fill_value=_np.nan)
    V = grid.interp(V, 'Y', boundary='fill', fill_value=_np.nan)

    # Sum squared values
    sum2 = _np.power(U, 2) + _np.power(V, 2)

    # Non-hydrostatic case
    if eps_nh:

        # Add missing variables
        varList = ['W']
        od = _add_missing_variables(od, varList)

        # Extract variables
        W = od._ds['W']

        # Compute anomalies
        Wmean = weighted_mean(od, 'W', 'time', False)
        W = W - Wmean['w_mean_W']

        # Interpolate vertical velocity
        W = grid.interp(W, 'Z', to='center',
                        boundary='fill', fill_value=_np.nan)

        # Sum squared values
        sum2 = sum2 + eps_nh * _np.power(W, 2)

    # Create DataArray
    EKE = sum2 / 2
    EKE.attrs['units'] = 'm^2 s^-2'
    EKE.attrs['long_name'] = 'eddy kinetic energy'
    EKE.attrs['OceanSpy_parameters'] = str(params2use)

    # Create ds
    ds = _xr.Dataset({'EKE': EKE}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def horizontal_divergence_velocity(od):
    """
    Compute horizontal divergence of the velocity field.

    .. math::
        \\nabla_{H} \\cdot {\\bf u} =
        \\frac{\\partial u}{\\partial x}
        +\\frac{\\partial v}{\\partial y}

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.

    Returns
    -------
    ds: xarray.Dataset
        | hor_div_vel: horizontal divergence of the velocity field

    References
    ----------
    Numerical Method:
    https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#horizontal-divergence

    See Also
    --------
    divergence
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Message
    print('Computing horizontal divergence of the velocity field.')

    # Create DataArray
    div = divergence(od, iName='U', jName='V', kName=None, aliased=False)
    hor_div_vel = div['dU_dX'] + div['dV_dY']
    hor_div_vel.attrs['units'] = 'm s^-2'
    long_name = 'horizontal divergence of the velocity field'
    hor_div_vel.attrs['long_name'] = long_name

    # Create ds
    ds = _xr.Dataset({'hor_div_vel': hor_div_vel}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def shear_strain(od):
    """
    Compute shear component of strain.

    .. math::
        S_s = \\frac{\\partial v}{\\partial x}
        +\\frac{\\partial u}{\\partial y}

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.

    Returns
    -------
    ds: xarray.Dataset
        | s_strain: potential density anomaly

    See Also
    --------
    normal_strain
    Okubo_Weiss_parameter
    curl
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Add missing variables
    varList = ['rAz', 'dyC', 'dxC', 'U', 'V']
    od = _add_missing_variables(od, varList)

    # Extract variables
    rAz = od._ds['rAz']
    dyC = od._ds['dyC']
    dxC = od._ds['dxC']
    U = od._ds['U']
    V = od._ds['V']

    # Extract grid
    grid = od._grid

    # Message
    print('Computing shear component of strain.')

    # Create DataArray
    # Same of vertical relative vorticity with + instead of -
    s_strain = (grid.diff(V*dyC, 'X',
                          boundary='fill', fill_value=_np.nan) +
                grid.diff(U*dxC, 'Y',
                          boundary='fill', fill_value=_np.nan)) / rAz
    s_strain.attrs['units'] = 's^-1'
    s_strain.attrs['long_name'] = 'shear component of strain'

    # Create ds
    ds = _xr.Dataset({'s_strain': s_strain}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def normal_strain(od):
    """
    Compute normal component of strain.

    .. math::
        S_n = \\frac{\\partial u}{\\partial x}
        -\\frac{\\partial v}{\\partial y}

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.

    Returns
    -------
    ds: xarray.Dataset
        | n_strain: normal component of strain

    See Also
    --------
    shear_strain
    Okubo_Weiss_parameter
    divergence
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Message
    print('Computing normal component of strain.')

    # Create DataArray
    # Same of horizontal divergence of velocity field with - instead of +
    div = divergence(od, iName='U', jName='V', kName=None, aliased=False)
    n_strain = div['dU_dX'] - div['dV_dY']
    n_strain.attrs['units'] = 's^-1'
    n_strain.attrs['long_name'] = 'normal component of strain'

    # Create ds
    ds = _xr.Dataset({'n_strain': n_strain}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def Okubo_Weiss_parameter(od):
    """
    Compute Okubo-Weiss parameter.
    Vertical component of relative vorticity
    and shear component of strain are interpolated to C grid points.

    .. math::
        OW =
        S_n^2 + S_s^2 - \\zeta^2

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.

    Returns
    -------
    ds: xarray.Dataset
        | Okubo_Weiss: Okubo-Weiss parameter

    See Also
    --------
    shear_strain
    normal_strain
    vertical_relative_vorticity
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Add missing variables
    varList = ['momVort3', 's_strain', 'n_strain']
    od = _add_missing_variables(od, varList)

    # Extract variables
    momVort3 = od._ds['momVort3']
    s_strain = od._ds['s_strain']
    n_strain = od._ds['n_strain']

    # Extract grid
    grid = od._grid

    # Message
    print('Computing Okubo-Weiss parameter.')

    # Interpolate to C grid points
    momVort3 = grid.interp(grid.interp(momVort3, 'X',
                                       boundary='fill', fill_value=_np.nan),
                           'Y', boundary='fill', fill_value=_np.nan)
    s_strain = grid.interp(grid.interp(s_strain, 'X',
                                       boundary='fill', fill_value=_np.nan),
                           'Y', boundary='fill', fill_value=_np.nan)

    # Create DataArray
    Okubo_Weiss = (_np.square(s_strain)
                   + _np.square(n_strain)
                   - _np.square(momVort3))
    Okubo_Weiss.attrs['units'] = 's^-2'
    Okubo_Weiss.attrs['long_name'] = 'Okubo-Weiss parameter'

    # Create ds
    ds = _xr.Dataset({'Okubo_Weiss': Okubo_Weiss}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def Ertel_potential_vorticity(od, full=True):
    """
    Compute Ertel Potential Vorticity.
    Interpolate all terms to C and Z points.

    .. math::
        Q =
        - \\frac{\\omega \\cdot \\nabla \\rho}{\\rho} =
        (f + \\zeta)\\frac{N^2}{g}
        + \\frac{\\left(\\zeta_H+e\\hat{\\mathbf{y}}\\right)
        \\cdot\\nabla_H\\rho}{\\rho_0}

    Parameters used:
        | g
        | rho0
        | omega

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute.
    full: bool
        If False, only use
        vertical component of the vorticity vectors (fCoriG, momVort3).
        If True,
        use both vertical and horizontal components.

    Returns
    -------
    ds: xarray.Dataset
        | Ertel_PV: Ertel Potential Vorticity

    See Also
    --------
    gradient
    relative_vorticity
    Brunt_Vaisala_frequency
    utils.Coriolis_parameter
    """

    # Check parameters
    _check_instance({'od': od,
                     'full': full},
                    {'od': 'oceanspy.OceanDataset',
                     'full': 'bool'})

    # Add missing variables
    varList = ['fCoriG', 'momVort3', 'N2']
    if full:
        varList = varList + ['momVort1', 'momVort2', 'YC']
    od = _add_missing_variables(od, varList)

    # Parameters
    paramsList = ['g', 'rho0']
    if full:
        paramsList = paramsList + ['omega']
    params2use = {par: od.parameters[par]
                  for par in od.parameters
                  if par in paramsList}

    # Extract variables
    momVort3 = od._ds['momVort3']
    fCoriG = od._ds['fCoriG']
    N2 = od._ds['N2']
    if full:
        momVort1 = od._ds['momVort1']
        momVort2 = od._ds['momVort2']
        YC = od._ds['YC']

    # Extract parameters
    g = params2use['g']
    rho0 = params2use['rho0']
    if full:
        omega = params2use['omega']

    # Extract grid
    grid = od._grid

    # Message
    print('Computing Ertel potential vorticity'
          ' using the following parameters: {}.'.format(params2use))

    # Compute Sigma0 gradients
    Sigma0_grads = gradient(od, varNameList='Sigma0', axesList=['X', 'Y'],
                            aliased=False)

    # Interpolate fields
    fpluszeta = grid.interp(grid.interp(momVort3 + fCoriG, 'X',
                                        boundary='fill', fill_value=_np.nan),
                            'Y', boundary='fill', fill_value=_np.nan)
    N2 = grid.interp(N2, 'Z', to='center', boundary='fill', fill_value=_np.nan)
    if full:
        momVort1 = grid.interp(grid.interp(momVort1, 'Y'), 'Z', to='center',
                               boundary='fill', fill_value=_np.nan)
        momVort2 = grid.interp(grid.interp(momVort2, 'X'), 'Z', to='center',
                               boundary='fill', fill_value=_np.nan)
        dSigma0_dX = grid.interp(Sigma0_grads['dSigma0_dX'], 'X',
                                 boundary='fill', fill_value=_np.nan)
        dSigma0_dY = grid.interp(Sigma0_grads['dSigma0_dY'], 'Y',
                                 boundary='fill', fill_value=_np.nan)
        _, e = _utils.Coriolis_parameter(YC, omega)

    # Create DataArray
    Ertel_PV = fpluszeta * N2 / g
    if full:
        Ertel_PV = (Ertel_PV
                    + (momVort1 * dSigma0_dX
                       + (momVort2 + e) * dSigma0_dY)/rho0)
    Ertel_PV.attrs['units'] = 'm^-1 s^-1'
    Ertel_PV.attrs['long_name'] = 'Ertel potential vorticity'
    Ertel_PV.attrs['OceanSpy_parameters'] = str(params2use)

    # Create ds
    ds = _xr.Dataset({'Ertel_PV': Ertel_PV}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def mooring_volume_transport(od):
    """
    Compute horizontal volume flux
    through a mooring array section (in/outflow).
    If the array is closed,
    transport at the first mooring is not computed.
    Otherwise,
    transport at both the first and last mooring is not computed.
    Transport can be computed following two paths,
    so the dimension `path` is added.

    .. math::
        T(mooring, Z, time, path) =
        T_x + T_y =
        u \\Delta y \\Delta z + v \\Delta x \\Delta z

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute

    Returns
    -------
    ds: xarray.Dataset
        | transport: Horizontal volume transport
        | Vtransport: Meridional volume transport
        | Utransport: Zonal volume transport
        | Y_transport: Y coordinate of horizontal volume transport
        | X_transport: X coordinate of horizontal volume transport
        | Y_Utransport: Y coordinate of zonal volume transport
        | X_Utransport: X coordinate of zonal volume transport
        | Y_Vtransport: Y coordinate of meridional volume transport
        | X_Vtransport: X coordinate of meridional volume transport
        | dir_Utransport: Direction of zonal volume transport
        | dir_Vtransport: Direction of meridional volume transport

    See Also
    --------
    subsample.mooring_array
    geographical_aligned_velocities
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    if 'mooring' not in od._ds.dims:
        raise ValueError('oceandatasets must be subsampled'
                         ' using `subsample.mooring_array`')

    # Add missing variables
    varList = ['XC', 'YC',
               'dyG', 'dxG', 'drF',
               'U', 'V',
               'HFacS', 'HFacW',
               'XU', 'YU', 'XV', 'YV']
    od = _add_missing_variables(od, varList)

    # Message
    print('Computing horizontal volume transport.')

    # Extract variables
    mooring = od._ds['mooring']
    XC = od._ds['XC'].squeeze(('Y', 'X'))
    YC = od._ds['YC'].squeeze(('Y', 'X'))
    XU = od._ds['XU'].squeeze(('Y'))
    YU = od._ds['YU'].squeeze(('Y'))
    XV = od._ds['XV'].squeeze(('X'))
    YV = od._ds['YV'].squeeze(('X'))

    # Compute transport
    U_tran = (od._ds['U']
              * od._ds['dyG']
              * od._ds['HFacW']
              * od._ds['drF']).squeeze('Y')
    V_tran = (od._ds['V']
              * od._ds['dxG']
              * od._ds['HFacS']
              * od._ds['drF']).squeeze('X')

    # Extract left and right values
    U1 = U_tran.isel(Xp1=1)
    U0 = U_tran.isel(Xp1=0)
    V1 = V_tran.isel(Yp1=1)
    V0 = V_tran.isel(Yp1=0)

    # Initialize direction
    U0_dir = _np.zeros((len(XC), 2))
    U1_dir = _np.zeros((len(XC), 2))
    V0_dir = _np.zeros((len(YC), 2))
    V1_dir = _np.zeros((len(YC), 2))

    # Steps
    diffX = _np.diff(XC)
    diffY = _np.diff(YC)

    # Closed array?
    if XC[0] == XC[-1] and YC[0] == YC[-1]:
        # Add first at the end
        closed = True
        diffX = _np.append(diffX, diffX[0])
        diffY = _np.append(diffY, diffY[0])
    else:
        closed = False

    # Loop
    Usign = 1
    Vsign = 1
    keepXf = False
    keepYf = False
    for i in range(len(diffX)-1):
        if diffY[i] == 0 and diffY[i+1] == 0:  # Zonal
            V1_dir[i+1, :] = Vsign
            V0_dir[i+1, :] = Vsign
        elif diffX[i] == 0 and diffX[i+1] == 0:  # Meridional
            U1_dir[i+1, :] = Usign
            U0_dir[i+1, :] = Usign

        # Corners
        elif (diffY[i] < 0 and diffX[i+1] > 0):  # |_
            Vsign = Usign
            keepYf = keepXf
            U0_dir[i+1, :] = Usign
            V0_dir[i+1, :] = Vsign
        elif (diffY[i+1] > 0 and diffX[i] < 0):
            Usign = Vsign
            keepXf = keepYf
            U0_dir[i+1, :] = Usign
            V0_dir[i+1, :] = Vsign

        elif (diffY[i] > 0 and diffX[i+1] > 0):  # |‾
            Vsign = -Usign
            keepYf = not keepXf
            U0_dir[i+1, :] = Usign
            V1_dir[i+1, :] = Vsign
        elif (diffY[i+1] < 0 and diffX[i] < 0):
            Usign = -Vsign
            keepXf = not keepYf
            U0_dir[i+1, :] = Usign
            V1_dir[i+1, :] = Vsign

        elif (diffX[i] > 0 and diffY[i+1] < 0):  # ‾|
            Usign = Vsign
            keepXf = keepYf
            V1_dir[i+1, :] = Vsign
            U1_dir[i+1, :] = Usign
        elif (diffX[i+1] < 0 and diffY[i] > 0):
            Vsign = Usign
            keepYf = keepXf
            V1_dir[i+1, :] = Vsign
            U1_dir[i+1, :] = Usign

        elif (diffX[i] > 0 and diffY[i+1] > 0):  # _|
            Usign = -Vsign
            keepXf = not keepYf
            V0_dir[i+1, :] = Vsign
            U1_dir[i+1, :] = Usign
        elif (diffX[i+1] < 0 and diffY[i] < 0):
            Vsign = -Usign
            keepYf = not keepXf
            V0_dir[i+1, :] = Vsign
            U1_dir[i+1, :] = Usign

        if keepXf:
            U1_dir[i+1, 0] = 0
            U0_dir[i+1, 1] = 0
        else:
            U0_dir[i+1, 0] = 0
            U1_dir[i+1, 1] = 0
        if keepYf:
            V1_dir[i+1, 0] = 0
            V0_dir[i+1, 1] = 0
        else:
            V0_dir[i+1, 0] = 0
            V1_dir[i+1, 1] = 0

    # Create direction DataArrays.
    # Add a switch to return this? Useful to debug and/or plot velocities.
    U1_dir = _xr.DataArray(U1_dir,
                           coords={'mooring': mooring,
                                   'path': [0, 1]},
                           dims=('mooring', 'path'))
    U0_dir = _xr.DataArray(U0_dir,
                           coords={'mooring': mooring,
                                   'path': [0, 1]},
                           dims=('mooring', 'path'))
    V1_dir = _xr.DataArray(V1_dir,
                           coords={'mooring': mooring,
                                   'path': [0, 1]},
                           dims=('mooring', 'path'))
    V0_dir = _xr.DataArray(V0_dir,
                           coords={'mooring': mooring,
                                   'path': [0, 1]},
                           dims=('mooring', 'path'))

    # Mask first mooring
    U1_dir = U1_dir.where(U1_dir['mooring']
                          != U1_dir['mooring'].isel(mooring=0))
    U0_dir = U0_dir.where(U0_dir['mooring']
                          != U0_dir['mooring'].isel(mooring=0))
    V1_dir = V1_dir.where(V1_dir['mooring']
                          != V1_dir['mooring'].isel(mooring=0))
    V0_dir = V0_dir.where(V0_dir['mooring']
                          != V0_dir['mooring'].isel(mooring=0))

    if not closed:
        # Mask first mooring
        U1_dir = U1_dir.where(U1_dir['mooring']
                              != U1_dir['mooring'].isel(mooring=-1))
        U0_dir = U0_dir.where(U0_dir['mooring']
                              != U0_dir['mooring'].isel(mooring=-1))
        V1_dir = V1_dir.where(V1_dir['mooring']
                              != V1_dir['mooring'].isel(mooring=-1))
        V0_dir = V0_dir.where(V0_dir['mooring']
                              != V0_dir['mooring'].isel(mooring=-1))

    # Compute transport
    transport = (U1*U1_dir + U0*U0_dir + V1*V1_dir + V0*V0_dir) * 1.E-6
    transport.attrs['units'] = 'Sv'
    transport.attrs['long_name'] = 'Horizontal volume transport'
    Vtransport = (V1*V1_dir+V0*V0_dir)*1.E-6
    Vtransport.attrs['units'] = 'Sv'
    Vtransport.attrs['long_name'] = 'Meridional volume transport'
    Utransport = (U1*U1_dir+U0*U0_dir)*1.E-6
    Utransport.attrs['units'] = 'Sv'
    Utransport.attrs['long_name'] = 'Zonal volume transport'

    # Additional info
    Y_transport = YC
    long_name = 'Y coordinate of horizontal volume transport'
    Y_transport.attrs['long_name'] = long_name
    X_transport = XC
    long_name = 'X coordinate of horizontal volume transport'
    X_transport.attrs['long_name'] = long_name

    Y_Utransport = _xr.where(U1_dir != 0, YU.isel(Xp1=1), _np.nan)
    Y_Utransport = _xr.where(U0_dir != 0, YU.isel(Xp1=0), Y_Utransport)
    long_name = 'Y coordinate of zonal volume transport'
    Y_Utransport.attrs['long_name'] = long_name
    X_Utransport = _xr.where(U1_dir != 0, XU.isel(Xp1=1), _np.nan)
    X_Utransport = _xr.where(U0_dir != 0, XU.isel(Xp1=0), X_Utransport)
    long_name = 'X coordinate of zonal volume transport'
    X_Utransport.attrs['long_name'] = long_name

    Y_Vtransport = _xr.where(V1_dir != 0, YV.isel(Yp1=1), _np.nan)
    Y_Vtransport = _xr.where(V0_dir != 0, YV.isel(Yp1=0), Y_Vtransport)
    long_name = 'Y coordinate of meridional volume transport'
    Y_Vtransport.attrs['long_name'] = long_name
    X_Vtransport = _xr.where(V1_dir != 0, XV.isel(Yp1=1), _np.nan)
    X_Vtransport = _xr.where(V0_dir != 0, XV.isel(Yp1=0), X_Vtransport)
    long_name = 'X coordinate of meridional volume transport'
    X_Vtransport.attrs['long_name'] = long_name

    dir_Vtransport = _xr.where(V1_dir != 0, V1_dir, _np.nan)
    dir_Vtransport = _xr.where(V0_dir != 0, V0_dir, dir_Vtransport)
    long_name = 'Direction of meridional volume transport'
    dir_Vtransport.attrs['long_name'] = long_name
    dir_Vtransport.attrs['units'] = '1: original, -1: flipped'
    dir_Utransport = _xr.where(U1_dir != 0, U1_dir, _np.nan)
    dir_Utransport = _xr.where(U0_dir != 0, U0_dir, dir_Utransport)
    long_name = 'Direction of zonal volume transport'
    dir_Utransport.attrs['long_name'] = long_name
    dir_Utransport.attrs['units'] = '1: original, -1: flipped'

    # Create ds
    ds = _xr.Dataset({'transport': transport,
                      'Vtransport': Vtransport,
                      'Utransport': Utransport,
                      'Y_transport': Y_transport,
                      'X_transport': X_transport,
                      'Y_Utransport': Y_Utransport,
                      'X_Utransport': X_Utransport,
                      'Y_Vtransport': Y_Vtransport,
                      'X_Vtransport': X_Vtransport,
                      'dir_Utransport': dir_Utransport,
                      'dir_Vtransport': dir_Vtransport},
                     attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def geographical_aligned_velocities(od):
    """
    Compute zonal and meridional velocities
    from U and V on orthogonal curvilinear grid.

    .. math::
        (u_{zonal}, v_{merid}) =
         (u\\cos{\\phi} - v\\sin{\\phi}, u\\sin{\\phi} + v\\cos{\\phi})

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute

    Returns
    -------
    ds: xarray.Dataset
        | U_zonal: zonal velocity
        | V_merid: meridional velocity
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Add missing variables
    varList = ['U', 'V', 'AngleCS', 'AngleSN']
    od = _add_missing_variables(od, varList)

    # Message
    print('Computing geographical aligned velocities.')

    # Extract Variables
    U = od._ds['U']
    V = od._ds['V']
    AngleCS = od._ds['AngleCS']
    AngleSN = od._ds['AngleSN']

    # Extract grid
    grid = od._grid

    # Move to C grid
    U = grid.interp(U, 'X', boundary='fill', fill_value=_np.nan)
    V = grid.interp(V, 'Y', boundary='fill', fill_value=_np.nan)

    # Compute velocities
    U_zonal = U * AngleCS - V * AngleSN
    if 'units' in od._ds['U'].attrs:
        U_zonal.attrs['units'] = od._ds['U'].attrs['units']
    U_zonal.attrs['long_name'] = 'zonal velocity'
    U_zonal.attrs['direction'] = 'positive: eastwards'

    V_merid = U * AngleSN + V * AngleCS
    if 'units' in od._ds['V'].attrs:
        V_merid.attrs['units'] = od._ds['V'].attrs['units']
    V_merid.attrs['long_name'] = 'meridional velocity'
    V_merid.attrs['direction'] = 'positive: northwards'

    # Create ds
    ds = _xr.Dataset({'U_zonal': U_zonal,
                      'V_merid': V_merid}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def survey_aligned_velocities(od):
    """
    Compute horizontal velocities orthogonal and tangential to a survey.

    .. math::
        (v_{tan}, v_{ort}) = (u\\cos{\\phi} + v\\sin{\\phi},
         v\\cos{\\phi} - u\\sin{\\phi})

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute

    Returns
    -------
    ds: xarray.Dataset
        | rot_ang_Vel: Angle to rotate geographical
         to survey aligned velocities
        | tan_Vel: Velocity component tangential to survey
        | ort_Vel: Velocity component orthogonal to survey

    See Also
    --------
    subsample.survey_stations
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    if 'station' not in od._ds.dims:
        raise ValueError('oceandatasets must be subsampled using'
                         ' `subsample.survey_stations`')

    # Get zonal and meridional velocities
    var_list = ['lat', 'lon']
    try:
        # Add missing variables
        varList = ['U_zonal', 'V_merid'] + var_list
        od = _add_missing_variables(od, varList)

        # Extract variables
        U = od._ds['U_zonal']
        V = od._ds['V_merid']
    except Exception as e:
        # Assume U=U_zonal and V=V_zonal
        _warnings.warn(("\n{}"
                        "\nAssuming U=U_zonal and V=V_merid."
                        "\nIf you are using curvilinear coordinates,"
                        " run `compute.geographical_aligned_velocities`"
                        " before `subsample.survey_stations`").format(e),
                       stacklevel=2)

        # Add missing variables
        varList = ['U', 'V'] + var_list
        od = _add_missing_variables(od, varList)

        # Extract variables
        U = od._ds['U']
        V = od._ds['V']

    # Extract varibles
    lat = _np.deg2rad(od._ds['lat'])
    lon = _np.deg2rad(od._ds['lon'])

    # Extract grid
    grid = od._grid

    # Message
    print('Computing survey aligned velocities.')

    # Compute azimuth
    # Translated from matlab:
    # https://www.mathworks.com/help/map/ref/azimuth.html
    az = _np.arctan2(_np.cos(lat[1:]).values
                     * _np.sin(grid.diff(lon, 'station')),
                     _np.cos(lat[:-1]).values * _np.sin(lat[1:]).values
                     - _np.sin(lat[:-1]).values
                     * _np.cos(lat[1:]).values
                     * _np.cos(grid.diff(lon, 'station')))
    az = grid.interp(az, 'station', boundary='extend')
    az = _xr.where(_np.rad2deg(az) < 0, _np.pi*2 + az, az)

    # Compute rotation angle
    rot_ang_rad = _np.pi/2 - az
    rot_ang_rad = _xr.where(rot_ang_rad < 0,
                            _np.pi*2 + rot_ang_rad, rot_ang_rad)
    rot_ang_deg = _np.rad2deg(rot_ang_rad)
    rot_ang_Vel = rot_ang_deg
    long_name = 'Angle to rotate geographical to survey aligned velocities'
    rot_ang_Vel.attrs['long_name'] = long_name
    rot_ang_Vel.attrs['units'] = 'deg (+: counterclockwise)'

    # Rotate velocities
    tan_Vel = U*_np.cos(rot_ang_rad) + V*_np.sin(rot_ang_rad)
    tan_Vel.attrs['long_name'] = 'Velocity component tangential to survey'
    if 'units' in U.attrs:
        units = U.attrs['units']
    else:
        units = ' '
    tan_Vel.attrs['units'] = ('{} '
                              '(+: flow towards station indexed'
                              ' with higher number)'
                              ''.format(units))
    ort_Vel = V*_np.cos(rot_ang_rad) - U*_np.sin(rot_ang_rad)
    ort_Vel.attrs['long_name'] = 'Velocity component orthogonal to survey'
    if 'units' in V.attrs:
        units = V.attrs['units']
    else:
        units = ' '
    ort_Vel.attrs['units'] = ('{} '
                              '(+: flow keeps station indexed'
                              ' with higher number to the right)'
                              ''.format(units))

    # Create ds
    ds = _xr.Dataset({'rot_ang_Vel': rot_ang_Vel,
                      'ort_Vel': ort_Vel,
                      'tan_Vel': tan_Vel}, attrs=od.dataset.attrs)

    return _ospy.OceanDataset(ds).dataset


def heat_budget(od):
    """
    Compute terms to close heat budget as explained by [Pie17]_.

    .. math::
        \\text{tendH = adv_hConvH + adv_vConvH
         + dif_vConvH + kpp_vConvH + forcH}

    Parameters used:
        | c_p
        | rho0

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute

    Returns
    -------
    ds: xarray.Dataset
        | tendH: Heat total tendency
        | adv_hConvH: Heat horizontal advective convergence
        | adv_vConvH: Heat vertical advective convergence
        | dif_vConvH: Heat vertical diffusive convergence
        | kpp_vConvH: Heat vertical kpp convergence
        | forcH: Heat surface forcing

    Notes
    -----
    This function is currently suited for the setup by [AHPM17]_:
    e.g., z* vertical coordinates, zero explicit diffusive fluxes, KPP.

    References
    ----------
    .. [Pie17] `<https://dspace.mit.edu/bitstream/handle/1721.1/111094/\
    memo_piecuch_2017_evaluating_budgets_in_eccov4r3.pdf?sequence=1>`_
    .. [AHPM17] Almansi, M., T.W. Haine, R.S. Pickart, M.G. Magaldi,\
    R. Gelderloos, and D. Mastropole, 2017:\
    High-Frequency Variability in the Circulation and Hydrography\
    of the Denmark Strait Overflow from a High-Resolution Numerical Model.\
    J. Phys. Oceanogr., 47, 2999–3013,\
    https://doi.org/10.1175/JPO-D-17-0129.1

    See Also
    --------
    salt_budget
    gradient
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Add missing variables
    varList = ['Temp', 'Eta', 'Depth',
               'ADVx_TH', 'ADVy_TH', 'ADVr_TH',
               'DFrI_TH', 'KPPg_TH', 'TFLUX', 'oceQsw_AVG',
               'time', 'HFacC', 'HFacW', 'HFacS',
               'drF', 'rA', 'Zp1']
    od = _add_missing_variables(od, varList)

    # Parameters
    paramsList = ['rho0', 'c_p']
    params2use = {par: od.parameters[par]
                  for par in od.parameters
                  if par in paramsList}

    # Message
    print('Computing heat budget terms using'
          ' the following parameters: {}.'.format(params2use))

    # Extract variables
    Temp = od._ds['Temp']
    Eta = od._ds['Eta']
    Depth = od._ds['Depth']
    ADVx_TH = od._ds['ADVx_TH']
    ADVy_TH = od._ds['ADVy_TH']
    ADVr_TH = od._ds['ADVr_TH']
    DFrI_TH = od._ds['DFrI_TH']
    KPPg_TH = od._ds['KPPg_TH']
    TFLUX = od._ds['TFLUX']
    oceQsw_AVG = od._ds['oceQsw_AVG']
    HFacC = od._ds['HFacC']
    HFacW = od._ds['HFacW']
    HFacS = od._ds['HFacS']
    drF = od._ds['drF']
    rA = od._ds['rA']
    Zp1 = od._ds['Zp1']

    # Extract parameters
    rho0 = od.parameters['rho0']
    c_p = od.parameters['c_p']

    # Extract grid
    grid = od._grid

    # Compute useful variables
    dzMat = drF * HFacC
    CellVol = rA * dzMat

    # Initialize dataset
    ds = _xr.Dataset({})

    # Total tendency
    z_star_scale = (1+Eta/Depth)
    tomerge = (Temp*z_star_scale).where(HFacC != 0).rename('Tscaled')
    od = od.merge_into_oceandataset(tomerge)
    units = 'degC/s'
    ds['tendH'] = gradient(od, 'Tscaled', 'time')['dTscaled_dtime']
    ds['tendH'].attrs['units'] = units
    ds['tendH'].attrs['long_name'] = 'Heat total tendency'
    ds['tendH'].attrs['OceanSpy_parameters'] = str(params2use)

    # Horizontal convergence
    ds['adv_hConvH'] = -(grid.diff(ADVx_TH.where(HFacW != 0), 'X',
                                   boundary='fill',
                                   fill_value=_np.nan)
                         + grid.diff(ADVy_TH.where(HFacS != 0), 'Y',
                                     boundary='fill',
                                     fill_value=_np.nan))
    ds['adv_hConvH'] = ds['adv_hConvH'] / CellVol
    ds['adv_hConvH'].attrs['units'] = units
    long_name = 'Heat horizontal advective convergence'
    ds['adv_hConvH'].attrs['long_name'] = long_name
    ds['adv_hConvH'].attrs['OceanSpy_parameters'] = str(params2use)

    # Vertical convergence
    for i, (var_in, name_out, long_name) in enumerate(zip([ADVr_TH,
                                                           DFrI_TH,
                                                           KPPg_TH],
                                                          ['adv_vConvH',
                                                           'dif_vConvH',
                                                           'kpp_vConvH'],
                                                          ['advective',
                                                           'diffusive',
                                                           'kpp'])):
        ds[name_out] = grid.diff(var_in, 'Z',
                                 boundary='fill', fill_value=_np.nan)
        ds[name_out] = ds[name_out].where(HFacC != 0) / CellVol
        ds[name_out].attrs['units'] = units
        ds[name_out].attrs['long_name'] = ('Heat vertical {} convergence'
                                           ''.format(long_name))
        ds[name_out].attrs['OceanSpy_parameters'] = str(params2use)

    # Surface flux
    # TODO: add these to parameters list?
    R = 0.62
    zeta1 = 0.6
    zeta2 = 20
    q = (R * _np.exp(Zp1/zeta1) + (1-R) * _np.exp(Zp1 / zeta2))
    q = q.where(Zp1 >= -200, 0)
    forcH = -grid.diff(q, 'Z').where(HFacC != 0)
    if Zp1.isel(Zp1=0) == 0:
        forcH_surf = (TFLUX
                      - (1-forcH.isel(Z=0))
                      * oceQsw_AVG).expand_dims('Z', Temp.dims.index('Z'))
        forcH_bott = forcH.isel(Z=slice(1, None)) * oceQsw_AVG
        forcH = _xr.concat([forcH_surf, forcH_bott], dim='Z')
    else:
        forcH = forcH * oceQsw_AVG
    ds['forcH'] = (forcH/(rho0*c_p*dzMat))
    ds['forcH'].attrs['units'] = units
    ds['forcH'].attrs['long_name'] = 'Heat surface forcing'
    ds['forcH'].attrs['OceanSpy_parameters'] = str(params2use)

    return _ospy.OceanDataset(ds).dataset


def salt_budget(od):
    """
    Compute terms to close salt budget as explained by [Pie17]_.

    .. math::
        \\text{tendS = adv_hConvS + adv_vConvS +
         dif_vConvS + kpp_vConvS + forcS}

    Parameters used:
        | rho0

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute

    Returns
    -------
    ds: xarray.Dataset
        | tendS: Salt total tendency
        | adv_hConvS: Salt horizontal advective convergence
        | adv_vConvS: Salt vertical advective convergence
        | dif_vConvS: Salt vertical diffusive convergence
        | kpp_vConvS: Salt vertical kpp convergence
        | forcS: Salt surface forcing

    Notes
    -----
    This function is currently suited for the setup by [AHPM17]_:
     e.g., z* vertical coordinates, zero explicit diffusive fluxes, KPP.

    References
    ----------
    .. [Pie17] `<https://dspace.mit.edu/bitstream/handle/1721.1/111094/\
    memo_piecuch_2017_evaluating_budgets_in_eccov4r3.pdf?sequence=1>`_
    .. [AHPM17] Almansi, M., T.W. Haine, R.S. Pickart, M.G. Magaldi,\
    R. Gelderloos, and D. Mastropole, 2017:\
    High-Frequency Variability in the Circulation and Hydrography\
    of the Denmark Strait Overflow from a High-Resolution Numerical Model.\
    J. Phys. Oceanogr., 47, 2999–3013,\
    https://doi.org/10.1175/JPO-D-17-0129.1

    See Also
    --------
    heat_budget
    gradient
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Add missing variables
    varList = ['S', 'Eta', 'Depth',
               'ADVx_SLT', 'ADVy_SLT', 'ADVr_SLT',
               'DFrI_SLT', 'KPPg_SLT', 'SFLUX', 'oceSPtnd',
               'time', 'HFacC', 'HFacW', 'HFacS', 'drF', 'rA', 'Zp1']
    od = _add_missing_variables(od, varList)

    # Parameters
    paramsList = ['rho0']
    params2use = {par: od.parameters[par]
                  for par in od.parameters
                  if par in paramsList}

    # Message
    print('Computing salt budget terms'
          ' using the following parameters: {}.'.format(params2use))

    # Extract variables
    S = od._ds['S']
    Eta = od._ds['Eta']
    Depth = od._ds['Depth']
    ADVx_SLT = od._ds['ADVx_SLT']
    ADVy_SLT = od._ds['ADVy_SLT']
    ADVr_SLT = od._ds['ADVr_SLT']
    DFrI_SLT = od._ds['DFrI_SLT']
    KPPg_SLT = od._ds['KPPg_SLT']
    SFLUX = od._ds['SFLUX']
    oceSPtnd = od._ds['oceSPtnd']
    HFacC = od._ds['HFacC']
    HFacW = od._ds['HFacW']
    HFacS = od._ds['HFacS']
    drF = od._ds['drF']
    rA = od._ds['rA']
    Zp1 = od._ds['Zp1']

    # Extract parameters
    rho0 = od.parameters['rho0']

    # Extract grid
    grid = od._grid

    # Compute useful variables
    dzMat = drF * HFacC
    CellVol = rA * dzMat

    # Initialize dataset
    ds = _xr.Dataset({})

    # Total tendency
    z_star_scale = (1+Eta/Depth)
    var2merge = (S * z_star_scale).where(HFacC != 0).rename('Sscaled')
    od = od.merge_into_oceandataset(var2merge)
    units = 'psu/s'
    ds['tendS'] = gradient(od, 'Sscaled', 'time')['dSscaled_dtime']
    ds['tendS'].attrs['units'] = units
    ds['tendS'].attrs['long_name'] = 'Salt total tendency'
    ds['tendS'].attrs['OceanSpy_parameters'] = str(params2use)

    # Horizontal convergence
    ds['adv_hConvS'] = -(grid.diff(ADVx_SLT.where(HFacW != 0), 'X',
                                   boundary='fill',
                                   fill_value=_np.nan)
                         + grid.diff(ADVy_SLT.where(HFacS != 0), 'Y',
                                     boundary='fill',
                                     fill_value=_np.nan))/CellVol
    ds['adv_hConvS'].attrs['units'] = units
    long_name = 'Salt horizontal advective convergence'
    ds['adv_hConvS'].attrs['long_name'] = long_name
    ds['adv_hConvS'].attrs['OceanSpy_parameters'] = str(params2use)

    # Vertical convergence
    for i, (var_in, name_out, long_name) in enumerate(zip([ADVr_SLT,
                                                           DFrI_SLT,
                                                           KPPg_SLT],
                                                          ['adv_vConvS',
                                                           'dif_vConvS',
                                                           'kpp_vConvS'],
                                                          ['advective',
                                                           'diffusive',
                                                           'kpp'])):
        ds[name_out] = grid.diff(var_in, 'Z',
                                 boundary='fill', fill_value=_np.nan)
        ds[name_out] = ds[name_out].where(HFacC != 0) / CellVol
        ds[name_out].attrs['units'] = units
        ds[name_out].attrs['long_name'] = ('Salt vertical {} convergence'
                                           ''.format(long_name))
        ds[name_out].attrs['OceanSpy_parameters'] = str(params2use)

    # Surface flux
    forcS = oceSPtnd
    if Zp1.isel(Zp1=0) == 0:
        forcS_surf = (SFLUX
                      + forcS.isel(Z=0)).expand_dims('Z', S.dims.index('Z'))
        forcS_bott = forcS.isel(Z=slice(1, None))
        forcS = _xr.concat([forcS_surf, forcS_bott], dim='Z')

    # Use same chunking
    ds['forcS'] = (forcS/(rho0*dzMat))
    ds['forcS'].attrs['units'] = units
    ds['forcS'].attrs['long_name'] = 'Salt surface forcing'
    ds['forcS'].attrs['OceanSpy_parameters'] = str(params2use)

    return _ospy.OceanDataset(ds).dataset


def missing_horizontal_spacing(od):
    """
    Compute missing horizontal spacing.

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute

    Returns
    -------
    ds: xarray.Dataset
        | dxF: x cell face separation
        | dxV: x v-velocity separation
        | dyF: y cell face separation
        | dyU: y u-velocity separation
    """

    # Check parameters
    _check_instance({'od': od}, 'oceanspy.OceanDataset')

    # Add missing variables
    varList = ['dxC', 'dxG', 'dyC', 'dyG']
    od = _add_missing_variables(od, varList)

    # Message
    print('Computing missing horizontal spacing.')

    # Unpack
    ds = od._ds
    grid = od._grid

    # Compute
    deltas = {}
    for var in [var
                for var in ['dxF', 'dxV', 'dyF', 'dyU']
                if var not in ds.variables]:

        # Pick dx
        if 'x' in var:
            pref = 'dx'
            axis = 'X'
        else:
            pref = 'dy'
            axis = 'Y'
        if 'F' in var:
            suf = 'C'
        else:
            suf = 'G'

        # Interpolate
        deltas[var] = grid.interp(ds[pref+suf], axis,  boundary='extend')

        # Add attributes
        if 'U' in var:
            add_vel = 'u-velocity'
        elif 'V' in var:
            add_vel = 'v-velocity'
        else:
            add_vel = 'cell face'
        deltas[var].attrs['description'] = ('{} {} separation'
                                            ''.format(axis.lower(), add_vel))
        if 'units' in ds[pref+suf].attrs:
            deltas[var].attrs['units'] = ds[pref+suf].attrs['units']

    # Create dataset
    ds = _xr.Dataset(deltas)

    return _ospy.OceanDataset(ds).dataset


class _computeMethods(object):
    """
    Enables use of functions as OceanDataset attributes.
    """

    def __init__(self, od):
        self._od = od

    @_functools.wraps(gradient)
    def gradient(self, overwrite=False, **kwargs):
        ds = gradient(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(divergence)
    def divergence(self, overwrite=False, **kwargs):
        ds = divergence(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(curl)
    def curl(self, overwrite=False, **kwargs):
        ds = curl(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(laplacian)
    def laplacian(self, overwrite=False, **kwargs):
        ds = laplacian(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(weighted_mean)
    def weighted_mean(self, overwrite=False, **kwargs):
        ds = weighted_mean(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(integral)
    def integral(self, overwrite=False, **kwargs):
        ds = integral(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(potential_density_anomaly)
    def potential_density_anomaly(self, overwrite=False, **kwargs):
        ds = potential_density_anomaly(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(Brunt_Vaisala_frequency)
    def Brunt_Vaisala_frequency(self, overwrite=False, **kwargs):
        ds = Brunt_Vaisala_frequency(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(velocity_magnitude)
    def velocity_magnitude(self, overwrite=False, **kwargs):
        ds = velocity_magnitude(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(horizontal_velocity_magnitude)
    def horizontal_velocity_magnitude(self, overwrite=False, **kwargs):
        ds = horizontal_velocity_magnitude(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(vertical_relative_vorticity)
    def vertical_relative_vorticity(self, overwrite=False, **kwargs):
        ds = vertical_relative_vorticity(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(relative_vorticity)
    def relative_vorticity(self, overwrite=False, **kwargs):
        ds = relative_vorticity(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(kinetic_energy)
    def kinetic_energy(self, overwrite=False, **kwargs):
        ds = kinetic_energy(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(eddy_kinetic_energy)
    def eddy_kinetic_energy(self, overwrite=False, **kwargs):
        ds = eddy_kinetic_energy(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(horizontal_divergence_velocity)
    def horizontal_divergence_velocity(self, overwrite=False, **kwargs):
        ds = horizontal_divergence_velocity(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(shear_strain)
    def shear_strain(self, overwrite=False, **kwargs):
        ds = shear_strain(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(normal_strain)
    def normal_strain(self, overwrite=False, **kwargs):
        ds = normal_strain(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(Okubo_Weiss_parameter)
    def Okubo_Weiss_parameter(self, overwrite=False, **kwargs):
        ds = Okubo_Weiss_parameter(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(Ertel_potential_vorticity)
    def Ertel_potential_vorticity(self, overwrite=False, **kwargs):
        ds = Ertel_potential_vorticity(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(mooring_volume_transport)
    def mooring_volume_transport(self, overwrite=False, **kwargs):
        ds = mooring_volume_transport(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(geographical_aligned_velocities)
    def geographical_aligned_velocities(self, overwrite=False, **kwargs):
        ds = geographical_aligned_velocities(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(survey_aligned_velocities)
    def survey_aligned_velocities(self, overwrite=False, **kwargs):
        ds = survey_aligned_velocities(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(heat_budget)
    def heat_budget(self, overwrite=False, **kwargs):
        ds = heat_budget(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(salt_budget)
    def salt_budget(self, overwrite=False, **kwargs):
        ds = salt_budget(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)

    @_functools.wraps(missing_horizontal_spacing)
    def missing_horizontal_spacing(self, overwrite=False, **kwargs):
        ds = missing_horizontal_spacing(self._od, **kwargs)
        return self._od.merge_into_oceandataset(ds, overwrite=overwrite)
