"""
Create new variables using OceanDataset objects.
"""

import xarray   as _xr
import oceanspy as _ospy
import numpy    as _np
import warnings as _warnings
import copy     as _copy
from . import utils as _utils

# Instructions for developers:
# 1) Every function operates on od, and returns ds 
# 2) Return ds as _ospy.OceanDataset(ds).dataset, so aliases are applied
# 3) Only use od._ds and od._grid
# 4) Make sure you don't lose global attributes (e.g., when creating a dataset)
# 5) Add new functions in docs/api.rst under subsample 
# 6) Create a shortcut in _oceandataset (merge+name of functions) and add the shortcut in docs/api.rst under OceanDataset - Shortcuts
# 7) Add function name, and variables returned in _FUNC2VARS


# TODO: add functions to compute dx, dy, dr, rA, ... when not available (maybe add in utils?)
# TODO: add functions to compute AngleCS and AngleSN when not available (maybe add in utils?)
#       http://mailman.mitgcm.org/pipermail/mitgcm-support/2014-January/008797.html
#       https://github.com/dcherian/tools/blob/master/mitgcm/matlab/cs_grid/cubeCalcAngle.m
# TODO: gradient, curl, divergence, laplacian currently can't handle aliases.
# TODO: any time velocities are mutiplied by hfac, we should use mass weighted velocities if available (mass_weighted function?)
# TODO: add area weighted mean for 2D variables, e.g. SSH (are_weighted_mean, SI, ...), or vertical sections
# TODO: add error when function will fail (e.g., divergence of W fails when only one level is available)
# TODO: compute transport for survey
# TODO: add integrals?

# Hard coded  list of variables outputed by functions
_FUNC2VARS = {'potential_density_anomaly'     : ['Sigma0'],
              'Brunt_Vaisala_frequency'       : ['N2'],
              'vertical_relative_vorticity'   : ['momVort3'],
              'relative_vorticity'            : ['momVort1', 'momVort2', 'momVort3'],
              'kinetic_energy'                : ['KE'],
              'eddy_kinetic_energy'           : ['EKE'],
              'horizontal_divergence_velocity': ['hor_div_vel'],
              'shear_strain'                  : ['s_strain'],
              'normal_strain'                 : ['n_strain'],
              'Okubo_Weiss_parameter'         : ['Okubo_Weiss'],
              'Ertel_potential_vorticity'     : ['Ertel_PV'],
              'mooring_horizontal_volume_transport' : ['transport', 
                                                       'Vtransport',     'Utransport', 
                                                       'Y_transport',    'X_transport', 
                                                       'Y_Utransport',   'X_Utransport',
                                                       'Y_Vtransport',   'X_Vtransport',
                                                       'dir_Utransport', 'dir_Vtransport'],
             'heat_budget'                    : ['tendH', 'adv_hConvH', 'adv_vConvH', 'dif_vConvH', 'kpp_vConvH', 'forcH'],
             'salt_budget'                    : ['tendS', 'adv_hConvS', 'adv_vConvS', 'dif_vConvS', 'kpp_vConvS', 'forcS'],
             'geographical_aligned_velocities': ['U_zonal', 'V_merid'],
             'survey_aligned_velocities'      : ['rot_ang_Vel', 'tan_Vel', 'ort_Vel']}



def _add_missing_variables(od, varList, FUNC2VARS = _FUNC2VARS, raiseError=True):
    """
    If any variable in varList is missing in the oceandataset, try to compute it.
    
    Parameters
    ----------
    od: OceanDataset
        oceandataset to check for missing variables
    varList: 1D array_like, str
        List of variables to check (strings).     
    FUNC2VARS: dict
        Dictionary that connect function names to computed variables.
        Keys are functions, values are list of variables
        
    Returns
    -------
    od: OceanDataset
        oceandataset with variables added
    """
    
    # Check parameters
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
    
    varList = _np.asarray(varList, dtype='str')
    if varList.ndim == 0: varList = varList.reshape(1)
    elif varList.ndim >1: raise TypeError('Invalid `varList`')
    
    # Return here if all variables already exist
    varList = [var for var in varList if var not in od._ds.variables]
    if len(varList)==0:
        return od
    
    # Raise error if variable not availabe
    VAR2FUNC  = {VAR: FUNC for FUNC in FUNC2VARS  for VAR in FUNC2VARS[FUNC]}
    var_error = [var for var in varList if var not in VAR2FUNC]
    if len(var_error)!=0:
        if od.aliases:
            var_error = [custom if ospy in var_error else ospy for ospy, custom in od.aliases.items()]
        message = 'These variables are not available and can not be computed: {}'.format(var_error)
        if raiseError:
            raise ValueError(message)
        else: 
            _warnings.warn(message, stacklevel=2)
            
            
    
    # Compute new variables
    funcList = list(set([VAR2FUNC[var] for var in varList if var in VAR2FUNC]))
    allds = []
    for func in funcList: 
        allds = allds+[eval('{}(od)'.format(func))]
    ds = _xr.merge(allds)
    ds = ds.drop([var for var in ds.variables if var not in varList])
    
    # Merge to od
    od = od.merge_Dataset(ds)
    
    return od


def gradient(od, varNameList, axesList=None, aliases = False):
    """
    Compute gradient along specified axes, returning all terms (not summed).
    
    .. math::
        \\nabla\\chi = \\sum\limits_{i=1}^n\\frac{\\partial \\chi}{\\partial x_i}\\hat{\\mathbf{x}}_i

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    varNameList: 1D array_like, str
        List of variables to differenciate
    
    Returns
    -------
    ds: xarray.Dataset 
        Contains all terms named as dvarName_daxis
    
    Examples
    --------
    >>> od = ospy.open_oceandataset.EGshelfIIseas2km_ASR()
    >>> ospy.compute.gradient(od, 'Temp')
    <xarray.Dataset>
    Dimensions:      (X: 960, Xp1: 961, Y: 880, Yp1: 881, Z: 216, Zl: 216, time: 1464, time_midp: 1463)
    Coordinates:
      * time         (time) datetime64[ns] 2007-09-01 ... 2008-08-31T18:00:00
      * Z            (Z) float64 -1.0 -3.5 -7.0 ... -3.112e+03 -3.126e+03 -3.142e+03
      * Yp1          (Yp1) float64 56.79 56.83 56.87 56.91 ... 76.43 76.46 76.5
      * X            (X) float64 -46.92 -46.83 -46.74 -46.65 ... 1.156 1.244 1.332
        XV           (Yp1, X) float64 dask.array<shape=(881, 960), chunksize=(881, 960)>
        YV           (Yp1, X) float64 dask.array<shape=(881, 960), chunksize=(881, 960)>
      * Y            (Y) float64 56.81 56.85 56.89 56.93 ... 76.37 76.41 76.44 76.48
      * Xp1          (Xp1) float64 -46.96 -46.87 -46.78 -46.7 ... 1.2 1.288 1.376
        XU           (Y, Xp1) float64 dask.array<shape=(880, 961), chunksize=(880, 961)>
        YU           (Y, Xp1) float64 dask.array<shape=(880, 961), chunksize=(880, 961)>
      * Zl           (Zl) float64 0.0 -2.0 -5.0 ... -3.104e+03 -3.119e+03 -3.134e+03
      * time_midp    (time_midp) datetime64[ns] 2007-09-01T03:00:00 ... 2008-08-31T15:00:00
    Data variables:
        dTemp_dY     (time, Z, Yp1, X) float64 dask.array<shape=(1464, 216, 881, 960), chunksize=(40, 216, 1, 960)>
        dTemp_dX     (time, Z, Y, Xp1) float64 dask.array<shape=(1464, 216, 880, 961), chunksize=(40, 216, 880, 1)>
        dTemp_dZ     (time, Zl, Y, X) float64 dask.array<shape=(1464, 216, 880, 960), chunksize=(40, 1, 880, 960)>
        dTemp_dtime  (time_midp, Z, Y, X) float64 dask.array<shape=(1463, 216, 880, 960), chunksize=(39, 216, 880, 960)>
            
    Notes
    -----
    Denominator is delta of distances for mooring and survey (units: km)
    
    References
    ----------
    MITgcm: https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#notation
    """
    # TODO: should I remove summatory in doc?
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
    
    varNameList = _np.asarray(varNameList, dtype='str')
    if varNameList.ndim == 0: varNameList = varNameList.reshape(1)
    elif varNameList.ndim >1: raise TypeError('Invalid `varNameList`')
        
    if axesList is not None:
        axesList = _np.asarray(axesList, dtype='str')
        if axesList.ndim == 0: axesList = axesList.reshape(1)
        elif axesList.ndim >1: raise TypeError('Invalid `axesList`')
            
    grid_axes = [coord for coord in od.grid_coords]
    if axesList is None: 
        axesList = grid_axes
    else:
        err_axes = [axis for axis in axesList if axis not in grid_axes]
        if len(err_axes)!=0: raise ValueError('{} are not available. Available axes are {}'.format(err_axes, grid_axes))
        
    # Add missing variables
    od = _add_missing_variables(od, list(varNameList))
    
    # Loop through variables
    grad = {}
    for varName in varNameList:
        
        for axis in axesList:
            # Numerator
            dnum = od._grid.diff(od._ds[varName], axis, boundary='fill', fill_value=_np.nan)

            # Horizontal gradient
            if axis in ['X', 'Y']:

                # Add missing variables
                varList = ['dxC', 'dxF', 'dxG', 'dxV', 'dyC', 'dyF', 'dyG', 'dyU']
                od = _add_missing_variables(od, varList)

                # Select d
                pointList = ['C', 'F', 'G']
                if   axis=='X': pointList = pointList+['V']
                elif axis=='Y': pointList = pointList+['U']
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
                if set(['X', 'Y']).issubset(dnum.dims):     HFac = od._ds['HFacC']
                elif set(['Xp1', 'Y']).issubset(dnum.dims): HFac = od._ds['HFacW']
                elif set(['X', 'Yp1']).issubset(dnum.dims): HFac = od._ds['HFacS']

                # Don't use dr, but compute 
                for dim in [od._grid.axes[axis].coords[coord].name for coord in od._grid.axes[axis].coords]:
                    if dim in od._ds[varName].dims:
                        dden = od._grid.diff(od._ds[dim], axis, boundary='fill', fill_value=_np.nan)
                        for coord in od._grid.axes[axis].coords:
                            if od._grid.axes[axis].coords[coord].name in dden.dims and coord!='center':
                                HFac = od._grid.interp(HFac, axis, to=coord, boundary='fill', fill_value=_np.nan)
                                continue
                        # Apply HFac
                        dden = dden * HFac
                        continue

            # Vertical gradient
            if axis == 'time':
                
                # Compute and conver in s
                dden = od._grid.diff(od._ds['time'], axis, boundary='fill', fill_value=_np.nan)
                dden = dden / _np.timedelta64(1, 's')
                
            # Vertical survey and mooring
            if axis in ['mooring', 'station']:
        
                # Compute and conver in s
                dden = od._grid.diff(od._ds[axis+'_dist'], axis, boundary='fill', fill_value=_np.nan)
                    
            # Add and clear
            grad['d'+varName+'_d'+axis] = dnum / dden
            del dnum, dden
            
    return _xr.Dataset(grad)
                
def divergence(od, iName=None, jName=None, kName=None, aliases = False):
    """
    Compute divergence of a vector field 

    .. math::
        \\nabla \\cdot \\overline{F} = 
        \\frac{\\partial F_x}{\\partial x}\\hat{\\mathbf{i}} +
        \\frac{\\partial F_y}{\\partial y}\\hat{\\mathbf{j}} + 
        \\frac{\\partial F_z}{\\partial z}\\hat{\\mathbf{k}} 

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    iName: str or None
        Name of variable corresponding to i-component
    jName: str or None
        Name of variable corresponding to j-component
    kName: str or None
        Name of variable corresponding to k-component
    
    Returns
    -------
    ds: xarray.Dataset 
        Contains all terms named as dName_daxis
    
    Examples
    --------
    >>> od = ospy.open_oceandataset.EGshelfIIseas2km_ASR()
    >>> ospy.compute.divergence(od, 'U', 'V', 'W')
    <xarray.Dataset>
    Dimensions:  (X: 960, Y: 880, Z: 216, time: 1464)
    Coordinates:
      * X        (X) float64 -46.92 -46.83 -46.74 -46.65 ... 1.069 1.156 1.244 1.332
      * Y        (Y) float64 56.81 56.85 56.89 56.93 ... 76.37 76.41 76.44 76.48
        XC       (Y, X) float64 dask.array<shape=(880, 960), chunksize=(880, 960)>
        YC       (Y, X) float64 dask.array<shape=(880, 960), chunksize=(880, 960)>
      * Z        (Z) float64 -1.0 -3.5 -7.0 ... -3.112e+03 -3.126e+03 -3.142e+03
      * time     (time) datetime64[ns] 2007-09-01 ... 2008-08-31T18:00:00
    Data variables:
        dU_dX    (Y, X, Z, time) float64 dask.array<shape=(880, 960, 216, 1464), chunksize=(880, 960, 216, 40)>
        dV_dY    (Y, X, Z, time) float64 dask.array<shape=(880, 960, 216, 1464), chunksize=(880, 960, 216, 40)>
        dW_dZ    (time, Z, Y, X) float64 dask.array<shape=(1464, 216, 880, 960), chunksize=(40, 215, 880, 960)>
        
    References
    ----------
    MITgcm: https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#notation
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
    if not isinstance(iName, (str, type(None))):
        raise TypeError('`iName` must be str or None')
    if not isinstance(jName, (str, type(None))):
        raise TypeError('`jName` must be str or None')
    if not isinstance(kName, (str, type(None))):
        raise TypeError('`kName` must be str or None')
    
    div = {}
    if iName is not None:
        # Add missing variables
        varList = ['HFacC', 'rA', 'dyG', 'HFacW', iName]
        od = _add_missing_variables(od, varList)
        
        # Add div
        div['d'+iName+'_dX'] = (od._grid.diff(od._ds[iName]*od._ds['HFacW']*od._ds['dyG'], 'X') / 
                                (od._ds['HFacC'] * od._ds['rA']))
    if jName is not None:
        # Add missing variables
        varList = ['HFacC', 'rA', 'dxG', 'HFacS', jName]
        od = _add_missing_variables(od, varList)
        
        # Add div
        div['d'+jName+'_dY'] = (od._grid.diff(od._ds[jName]*od._ds['HFacS']*od._ds['dxG'], 'Y') / 
                                (od._ds['HFacC'] * od._ds['rA']))
    if kName is not None:
        # Add missing variables
        od = _add_missing_variables(od, kName)
        
        # Add div (same of gradient)
        div['d'+kName+'_dZ'] = gradient(od, varNameList = kName, axesList='Z')['d'+kName+'_dZ']
    
    return _xr.Dataset(div)
                                                        
def curl(od, iName=None, jName=None, kName=None, aliases = False):
    """
    Compute curl of a vector field 

    .. math::
        \\nabla \\times \\overline{F} = 
        \\left(\\frac{\\partial F_z}{\\partial y}-\\frac{\\partial F_y}{\\partial z}\\right)\\hat{\\mathbf{i}} +
        \\left(\\frac{\\partial F_x}{\\partial z}-\\frac{\\partial F_z}{\\partial x}\\right)\\hat{\\mathbf{j}} + 
        \\left(\\frac{\\partial F_y}{\\partial x}-\\frac{\\partial F_x}{\\partial y}\\right)\\hat{\\mathbf{k}}

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    iName: str or None
        Name of variable corresponding to i-component
    jName: str or None
        Name of variable corresponding to j-component
    kName: str or None
        Name of variable corresponding to k-component
    
    Returns
    -------
    ds: xarray.Dataset 
        Contains all terms named as dName_daxis-dName_daxis
    
    Examples
    --------
    >>> od = ospy.open_oceandataset.EGshelfIIseas2km_ASR()
    >>> ospy.compute.curl(od, 'U', 'V', 'W')
    <xarray.Dataset>
    Dimensions:      (X: 960, Xp1: 961, Y: 880, Yp1: 881, Z: 216, Zl: 216, time: 1464)
    Coordinates:
      * Xp1          (Xp1) float64 -46.96 -46.87 -46.78 -46.7 ... 1.2 1.288 1.376
      * Yp1          (Yp1) float64 56.79 56.83 56.87 56.91 ... 76.43 76.46 76.5
        XG           (Yp1, Xp1) float64 dask.array<shape=(881, 961), chunksize=(881, 961)>
        YG           (Yp1, Xp1) float64 dask.array<shape=(881, 961), chunksize=(881, 961)>
      * time         (time) datetime64[ns] 2007-09-01 ... 2008-08-31T18:00:00
      * Z            (Z) float64 -1.0 -3.5 -7.0 ... -3.112e+03 -3.126e+03 -3.142e+03
      * Zl           (Zl) float64 0.0 -2.0 -5.0 ... -3.104e+03 -3.119e+03 -3.134e+03
      * X            (X) float64 -46.92 -46.83 -46.74 -46.65 ... 1.156 1.244 1.332
        XV           (Yp1, X) float64 dask.array<shape=(881, 960), chunksize=(881, 960)>
        YV           (Yp1, X) float64 dask.array<shape=(881, 960), chunksize=(881, 960)>
      * Y            (Y) float64 56.81 56.85 56.89 56.93 ... 76.37 76.41 76.44 76.48
        XU           (Y, Xp1) float64 dask.array<shape=(880, 961), chunksize=(880, 961)>
        YU           (Y, Xp1) float64 dask.array<shape=(880, 961), chunksize=(880, 961)>
    Data variables:
        dV_dX-dU_dY  (Yp1, Xp1, time, Z) float64 dask.array<shape=(881, 961, 1464, 216), chunksize=(1, 1, 40, 216)>
        dW_dY-dV_dZ  (time, Zl, Yp1, X) float64 dask.array<shape=(1464, 216, 881, 960), chunksize=(40, 1, 1, 960)>
        dU_dZ-dW_dX  (time, Zl, Y, Xp1) float64 dask.array<shape=(1464, 216, 880, 961), chunksize=(40, 1, 880, 1)>
        
    References
    ----------
    MITgcm: https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#notation
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
    if not isinstance(iName, (str, type(None))):
        raise TypeError('`iName` must be str or None')
    if not isinstance(jName, (str, type(None))):
        raise TypeError('`jName` must be str or None')
    if not isinstance(kName, (str, type(None))):
        raise TypeError('`kName` must be str or None')
        
    crl = {}
    if iName is not None and jName is not None:
        # Add missing variables
        varList = ['rAz', 'dyC', 'dxC', iName, jName]
        od = _add_missing_variables(od, varList)
        
        # Add curl
        Name = 'd'+jName+'_dX-d'+iName+'_dY'
        crl[Name] = (od._grid.diff(od._ds[jName]*od._ds['dyC'], 'X',
                                   boundary='fill', fill_value=_np.nan) -
                     od._grid.diff(od._ds[iName]*od._ds['dxC'], 'Y',
                                   boundary='fill', fill_value=_np.nan)) / od._ds['rAz']
                         
    if jName is not None and kName is not None:
        # Add missing variables
        varList = [jName, kName]
        od = _add_missing_variables(od, varList)
        
        # Add curl using gradients
        Name = 'd'+kName+'_dY-d'+jName+'_dZ'
        crl[Name] =(gradient(od, kName, 'Y')['d'+kName+'_dY'] -
                     gradient(od, jName, 'Z')['d'+jName+'_dZ'])
    if kName is not None and iName is not None:
        # Add curl using gradients
        Name = 'd'+iName+'_dZ-d'+kName+'_dX' 
        crl[Name] =(gradient(od, iName, 'Z')['d'+iName+'_dZ'] -
                     gradient(od, kName, 'X')['d'+kName+'_dX'])
        
    return _xr.Dataset(crl)           

def laplacian(od, varNameList, axesList=None, aliases = False):
    """
    Compute laplacian along specified axis

    .. math::
        \\nabla^2 \\chi = \\nabla \\cdot \\nabla \\chi

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    varNameList: 1D array_like, str
        Name of variables to differenciate
    
    Returns
    -------
    ds: xarray.Dataset 
        Contains all terms named as ddvarName_daxis_daxis
    
    Examples
    --------
    >>> od = ospy.open_oceandataset.EGshelfIIseas2km_ASR()
    >>> ospy.compute.laplacian(od, 'Temp')
    <xarray.Dataset>
    Dimensions:       (X: 960, Y: 880, Z: 216, time: 1464)
    Coordinates:
      * X             (X) float64 -46.92 -46.83 -46.74 -46.65 ... 1.156 1.244 1.332
      * Y             (Y) float64 56.81 56.85 56.89 56.93 ... 76.41 76.44 76.48
        XC            (Y, X) float64 dask.array<shape=(880, 960), chunksize=(880, 960)>
        YC            (Y, X) float64 dask.array<shape=(880, 960), chunksize=(880, 960)>
      * Z             (Z) float64 -1.0 -3.5 -7.0 ... -3.126e+03 -3.142e+03
      * time          (time) datetime64[ns] 2007-09-01 ... 2008-08-31T18:00:00
    Data variables:
        ddTemp_dX_dX  (Y, X, Z, time) float64 dask.array<shape=(880, 960, 216, 1464), chunksize=(880, 1, 216, 40)>
        ddTemp_dY_dY  (Y, X, Z, time) float64 dask.array<shape=(880, 960, 216, 1464), chunksize=(1, 960, 216, 40)>
        ddTemp_dZ_dZ  (time, Z, Y, X) float64 dask.array<shape=(1464, 216, 880, 960), chunksize=(40, 1, 880, 960)>
            
    References
    ----------
    MITgcm: https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#notation
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
    
    varNameList = _np.asarray(varNameList, dtype='str')
    if varNameList.ndim == 0: varNameList = varNameList.reshape(1)
    elif varNameList.ndim >1: raise TypeError('Invalid `varNameList`')
        
    # Loop through variables
    div = []
    for varName in varNameList:
        # Compute gradients
        grad = gradient(od, varNameList = varName, axesList = axesList)

        # Add to od
        od = _copy.copy(od)
        attrs  = od._ds.attrs
        od._ds = _xr.merge([od._ds, grad])
        od._ds.attrs = attrs

        # Compute laplacian
        compNames = {}
        for compName in grad.variables:
            if compName in grad.coords: continue
            elif compName[-1] == 'X': compNames['iName'] = compName
            elif compName[-1] == 'Y': compNames['jName'] = compName 
            elif compName[-1] == 'Z': compNames['kName'] = compName 
                
        div = div + [divergence(od, **compNames)]
    
    return _xr.merge(div)
     

def volume_cells(od, varNameList = None):
    """
    Compute volume of cells.

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    varNameList: 1D array_like, str, or None
        List of variables (strings).  
        If None, compute volumes for each grid in the dataset.
        Otherwise, compute volumes for grids of requested variables 
    
    Returns
    -------
    ds: xarray.Dataset 
        Contains cell volumes named as cellVol[grid point]_[Z dim]
    
    Examples
    --------
    >>> od = ospy.open_oceandataset.EGshelfIIseas2km_ASR()
    >>> ospy.compute.volume_cells(od)
    <xarray.Dataset>
    Dimensions:      (X: 960, Xp1: 961, Y: 880, Yp1: 881, Z: 216, Zl: 216)
    Coordinates:
      * Z            (Z) float64 -1.0 -3.5 -7.0 ... -3.112e+03 -3.126e+03 -3.142e+03
      * X            (X) float64 -46.92 -46.83 -46.74 -46.65 ... 1.156 1.244 1.332
      * Y            (Y) float64 56.81 56.85 56.89 56.93 ... 76.37 76.41 76.44 76.48
        XC           (Y, X) float64 dask.array<shape=(880, 960), chunksize=(880, 960)>
        YC           (Y, X) float64 dask.array<shape=(880, 960), chunksize=(880, 960)>
      * Xp1          (Xp1) float64 -46.96 -46.87 -46.78 -46.7 ... 1.2 1.288 1.376
        XU           (Y, Xp1) float64 dask.array<shape=(880, 961), chunksize=(880, 961)>
        YU           (Y, Xp1) float64 dask.array<shape=(880, 961), chunksize=(880, 961)>
      * Yp1          (Yp1) float64 56.79 56.83 56.87 56.91 ... 76.43 76.46 76.5
        XV           (Yp1, X) float64 dask.array<shape=(881, 960), chunksize=(881, 960)>
        YV           (Yp1, X) float64 dask.array<shape=(881, 960), chunksize=(881, 960)>
      * Zl           (Zl) float64 0.0 -2.0 -5.0 ... -3.104e+03 -3.119e+03 -3.134e+03
    Data variables:
        cellVolC_Z   (Z, Y, X) float64 dask.array<shape=(216, 880, 960), chunksize=(216, 880, 960)>
        cellVolW_Z   (Z, Y, Xp1) float64 dask.array<shape=(216, 880, 961), chunksize=(216, 880, 961)>
        cellVolS_Z   (Z, Yp1, X) float64 dask.array<shape=(216, 881, 960), chunksize=(216, 881, 960)>
        cellVolC_Zl  (Zl, Y, X) float64 dask.array<shape=(216, 880, 960), chunksize=(1, 880, 960)>
        
    Notes
    -----
    This function will take a few seconds when it need to check all variables. Use `varNameList` to make it quicker.  
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    if varNameList is not None:
        varNameList = _np.asarray(varNameList, dtype='str')
        if varNameList.ndim == 0: varNameList = varNameList.reshape(1)
        elif varNameList.ndim >1: raise TypeError('Invalid `varNameList`')
    else:
        # Check all variables is varNameList is not provided
        varNameList = od._ds.data_vars

    # Add missing variables
    # TODO: We don't really need to always check all of them!
    varList = ['HFacC', 'HFacW', 'HFacS', 'rA', 'rAs', 'rAw', 'rAz', 'drF', 'drC']
    od = _add_missing_variables(od, varList)
 
    # Loop through all variables
    dims_done = []
    vols = {}
    cont = False
    for varName in varNameList:
        for dims in dims_done:
            if set(dims).issubset(od._ds[varName].dims) or len(od._ds[varName].dims)<3: 
                cont = True
                continue
        if cont: 
            cont = False
            continue
        
        # Check dimensions
        for axis in ['X', 'Y', 'Z']:
            dims = [od._grid.axes[axis].coords[coord].name for coord in od._grid.axes[axis].coords]
            if not any(dim in od._ds[varName].dims for dim in dims): 
                cont = True
                continue
        if cont:
            cont = False
            continue
        
        # Extract HFac
        point = None
        if   set(['X', 'Y']).issubset(od._ds[varName].dims)  : point = 'C'
        elif set(['Xp1', 'Y']).issubset(od._ds[varName].dims): point = 'W'
        elif set(['X', 'Yp1']).issubset(od._ds[varName].dims): point = 'S'
            
        if point is None: continue
        else: HFac = od._ds['HFac'+point]
        
        # Extract rA
        rA = None
        for Aname in ['rA', 'rAs', 'rAw', 'rAz']:
            if set(od._ds[Aname].dims).issubset(od._ds[varName].dims): rA = od._ds[Aname]
        if rA is None: continue

        # Extract dr
        dr = None
        for rName in ['drF', 'drC']:
            if set(od._ds[rName].dims).issubset(od._ds[varName].dims):  dr = od._ds[rName]
        for coord in od._grid.axes['Z'].coords:
            if od._grid.axes['Z'].coords[coord].name in od._ds[varName].dims and coord!='center':
                HFac = od._grid.interp(HFac, 'Z', to=coord, boundary='fill', fill_value=_np.nan)
                if coord!='outer':
                    dr = od._grid.interp(od._ds['drF'], 'Z', to=coord, boundary='fill', fill_value=_np.nan)
        if dr is None: continue

        # Compute Volume
        cellVol = dr * HFac * rA
        name = 'cellVol'+point+'_'+dr.dims[0]
        
        # Add to dataset
        vols[name] = cellVol
        dims_done = dims_done + [cellVol.dims]
    
    return _xr.Dataset(vols)
    
def volume_weighted_mean(od, varNameList, aliases = False):
    """
    Compute volume weighted mean.

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    varNameList: 1D array_like, str
        List of variables (strings). 
    
    Returns
    -------
    ds: xarray.Dataset 
        Contains averaged variable named as varName_vw_mean
    
    Examples
    --------
    >>> od = ospy.open_oceandataset.EGshelfIIseas2km_ASR()
    >>> ospy.compute.volume_weighted_mean(od, 'Temp')
    <xarray.Dataset>
    Dimensions:       (time: 1464)
    Coordinates:
      * time          (time) datetime64[ns] 2007-09-01 ... 2008-08-31T18:00:00
    Data variables:
        Temp_vw_mean  (time) float64 dask.array<shape=(1464,), chunksize=(40,)>
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    varNameList = _np.asarray(varNameList, dtype='str')
    if varNameList.ndim == 0: varNameList = varNameList.reshape(1)
    elif varNameList.ndim >1: raise TypeError('Invalid `varNameList`')
    
    # Add missing variables
    varList = list(varNameList)
    od = _add_missing_variables(od, varList)
    
    # Message
    print('Computing volume weighted means')
    
    # Loop through variables
    mean = {}
    for varName in varNameList:

        # Compute Volume
        cellVol = volume_cells(od, varName)
        Volname = [var for var in cellVol.data_vars][0]
        cellVol = cellVol[Volname]

        # Add and clear
        mean[varName+'_vw_mean'] = ((od._ds[varName]*cellVol).sum(cellVol.dims)/
                                    cellVol.where(~od._ds[varName].isnull()).sum(cellVol.dims))
        mean[varName+'_vw_mean'].attrs = od._ds[varName].attrs
        
    return _xr.Dataset(mean)
    
    

def potential_density_anomaly(od):
    """
    Compute potential density anomaly.
    
    .. math::
        \\sigma_\\theta = \\rho_{S, \\theta, 0} -1000 \\text{ kg m}^{-3}
        
    Parameters used: 
        | eq_state

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    
    Returns
    -------
    ds: xarray.Dataset 
        | Sigma0: potential density anomaly
    
    See Also
    --------
    utils.densjmd95
    utils.densmdjwf
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    # Parameters
    paramsList = ['eq_state']
    params2use = {par:od.parameters[par] for par in od.parameters if par in paramsList}
    
    # Add missing variables
    varList = ['Temp', 'S']
    od = _add_missing_variables(od, varList)
    
    # Extract variables
    S    = od._ds['S']
    Temp = od._ds['Temp']
    
    # Message
    print('Computing potential density anomaly using the following parameters: {}'.format(params2use))
    
    # Create DataArray
    Sigma0 = eval('_utils.dens{}(S, Temp, 0)-1000'.format(params2use['eq_state']))
    Sigma0.attrs['units']      = 'kg/m^3'
    Sigma0.attrs['long_name']  = 'potential density anomaly'
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
        oceandataset used to compute
    
    Returns
    -------
    ds: xarray.Dataset
        | N2: Brunt-Väisälä Frequency
            
    See Also
    --------
    potential_density_anomaly
    gradient
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    # Add missing variables
    varList = ['Sigma0']
    od = _add_missing_variables(od, varList)
    
    # Parameters
    paramsList = ['eq_state']
    params2use = {par:od.parameters[par] for par in od.parameters if par in paramsList}
    
    # Extract parameters
    g    = od.parameters['g']
    rho0 = od.parameters['rho0']
    
    # Message
    print('Computing Brunt-Väisälä Frequency using the following parameters: {}'.format(params2use))
    
    # Create DataArray
    grad = gradient(od, varNameList = 'Sigma0', axesList= 'Z', aliases = False)
    N2   = - g / rho0 * grad['dSigma0_dZ']
    N2.attrs['units']      = 's^-2'
    N2.attrs['long_name']  = 'Brunt-Väisälä Frequency'
    N2.attrs['OceanSpy_parameters'] = str(params2use)
    
    # Create ds
    ds = _xr.Dataset({'N2': N2}, attrs=od.dataset.attrs)
    
    return _ospy.OceanDataset(ds).dataset
 
    
def vertical_relative_vorticity(od):
    """
    Compute vertical component of relative  vorticity.

    .. math::
        \\zeta = \\frac{\\partial v}{\\partial x}-\\frac{\\partial u}{\\partial y}
    
    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    
    Returns
    -------
    ds: xarray.Dataset 
        | momVort3: vertical component of relative vorticity
            
    See Also
    --------
    curl
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    # Message
    print('Computing vertical component of relative vorticity')
    
    # Create DataArray
    crl      = curl(od, iName='U', jName='V', kName=None, aliases = False)
    momVort3 = crl['dV_dX-dU_dY']
    momVort3.attrs['units']      = 's^-1'
    momVort3.attrs['long_name']  = 'Vertical component of relative vorticity'
    
    # Create ds
    ds = _xr.Dataset({'momVort3': momVort3}, attrs=od.dataset.attrs)
    
    return _ospy.OceanDataset(ds).dataset
    
    
    
def relative_vorticity(od):
    """
    Compute relative vorticity.

    .. math::
        \\overline{\\omega} = \\nabla \\times \\overline{u} = 
        \\left(\\frac{\\partial w}{\\partial y}-\\frac{\\partial v}{\\partial z}\\right)\\mathbf{i} +
        \\left(\\frac{\\partial u}{\\partial z}-\\frac{\\partial w}{\\partial x}\\right)\\mathbf{j} + 
        \\left(\\frac{\\partial v}{\\partial x}-\\frac{\\partial u}{\\partial y}\\right)\\mathbf{k}
    
    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    
    Returns
    -------
    ds: xarray.Dataset
        | momVort1: i-component of relative vorticity
        | momVort2: j-component of relative vorticity\
        | momVort3: k-component of relative vorticity
            
    See Also
    --------
    curl
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    # Message
    print('Computing relative vorticity')
    
    # Create DataArray
    ds = curl(od, iName='U', jName='V', kName='W', aliases = False)
    for _, (orig, out, comp) in enumerate(zip(['dV_dX-dU_dY', 'dW_dY-dV_dZ', 'dU_dZ-dW_dX'],
                                              ['momVort3', 'momVort1', 'momVort2'],
                                              ['k', 'i', 'j'])):
        ds = ds.rename({orig: out})
        ds[out].attrs['long_name'] = '{}-component of relative vorticity'.format(comp)
        ds[out].attrs['units']     = 's^-1'
    
    return _ospy.OceanDataset(ds).dataset    
    
    
    
def kinetic_energy(od):
    """
    Compute kinetic energy.

    .. math::
        KE = \\frac{1}{2}\\left(u^2 + v^2 + \\epsilon_{nh} w^2\\right)
    
    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    
    Parameters used: 
        | eps_nh
        
    Returns
    -------
    ds: xarray.Dataset 
        | KE: kinetic energy
            
    References
    ----------
    MITgcm: https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#kinetic-energy
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
     
    # Add missing variables
    varList = ['U', 'V']
    od = _add_missing_variables(od, varList)
    
    # Parameters
    paramsList = ['eps_nh']
    params2use = {par:od.parameters[par] for par in od.parameters if par in paramsList}
    
    # Extract variables
    U = od._ds['U']
    V = od._ds['V']
    
    # Extract grid
    grid = od._grid
    
    # Extract parameters
    eps_nh = od.parameters['eps_nh']
    
    # Message
    print('Computing kinetic energy using the following parameters: {}'.format(params2use))
    
    # Interpolate horizontal velocities
    U = grid.interp(U, 'X')
    V = grid.interp(V, 'Y')
    
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
        W = grid.interp(W, 'Z')
        
        # Sum squared values
        sum2 = sum2 + eps_nh * _np.power(W, 2)
        
    # Create DataArray
    KE = sum2 / 2
    KE.attrs['units']      = 'm^2 s^-2'
    KE.attrs['long_name']  = 'kinetic energy'
    KE.attrs['OceanSpy_parameters'] = str(params2use)
    
    # Create ds
    ds = _xr.Dataset({'KE': KE}, attrs=od.dataset.attrs)
    
    return _ospy.OceanDataset(ds).dataset



def eddy_kinetic_energy(od):
    """
    Compute eddy kinetic energy.

    .. math::
        KE = \\frac{1}{2}\\left((u-<u>_{time})^2 + (v-<v>_{time})^2 + \\epsilon_{nh} (w-<w>_{time})^2\\right)
    
    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    
    Parameters used: 
        | eps_nh
        
    Returns
    -------
    ds: xarray.Dataset
        | EKE: eddy kinetic energy
            
    References
    ----------
    MITgcm: https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#kinetic-energy
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
     
    # Add missing variables
    varList = ['U', 'V']
    od = _add_missing_variables(od, varList)
    
    # Parameters
    paramsList = ['eps_nh']
    params2use = {par:od.parameters[par] for par in od.parameters if par in paramsList}
    
    # Extract variables
    U = od._ds['U']
    V = od._ds['V'] 
    
    # Compute anomalies
    U = U - U.mean('time')
    V = V - V.mean('time')
    
    # Extract grid
    grid = od._grid
    
    # Extract parameters
    eps_nh = od.parameters['eps_nh']
    
    # Message
    print('Computing kinetic energy using the following parameters: {}'.format(params2use))
    
    # Interpolate horizontal velocities
    U = grid.interp(U, 'X')
    V = grid.interp(V, 'Y')
    
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
        W = W - W.mean('time')
        
        # Interpolate vertical velocity
        W = grid.interp(W, 'Z')
        
        # Sum squared values
        sum2 = sum2 + eps_nh * _np.power(W, 2)
        
    # Create DataArray
    EKE = sum2 / 2
    EKE.attrs['units']      = 'm^2 s^-2'
    EKE.attrs['long_name']  = 'eddy kinetic energy'
    EKE.attrs['OceanSpy_parameters'] = str(params2use)
    
    # Create ds
    ds = _xr.Dataset({'EKE': EKE}, attrs=od.dataset.attrs)
    
    return _ospy.OceanDataset(ds).dataset




def horizontal_divergence_velocity(od):
    """
    Compute horizontal divergence of the velocity field.

    .. math::
        \\nabla_{H} \\cdot \\overline{u} = \\frac{\\partial u}{\\partial x}+\\frac{\\partial v}{\\partial y}
    
    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    
    Returns
    -------
    ds: xarray.Dataset 
        | hor_div_vel: horizontal divergence of the velocity field
            
    See Also
    --------
    divergence
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    # Message
    print('Computing horizontal divergence of the velocity field')
    
    # Create DataArray
    div = divergence(od, iName='U', jName='V', kName=None, aliases = False)
    hor_div_vel = div['dU_dX'] + div['dV_dY']
    hor_div_vel.attrs['units']      = 'm s^-2'
    hor_div_vel.attrs['long_name']  = 'horizontal divergence of the velocity field'
    
    # Create ds
    ds = _xr.Dataset({'hor_div_vel': hor_div_vel}, attrs=od.dataset.attrs)
    
    return _ospy.OceanDataset(ds).dataset 



def shear_strain(od):
    """
    Compute shear component of strain.
    
    .. math::
        S_s = \\frac{\\partial v}{\\partial x}+\\frac{\\partial u}{\\partial y}

    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    
    Returns
    -------
    ds: xarray.Dataset 
        | s_strain: potential density anomaly
    
    See Also
    --------
    vertical_relative_vorticity
    curl
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
    
    # Add missing variables
    varList = ['rAz', 'dyC', 'dxC', 'U', 'V']
    od = _add_missing_variables(od, varList)
    
    # Extract variables
    rAz = od._ds['rAz']
    dyC = od._ds['dyC'] 
    dxC = od._ds['dxC'] 
    U   = od._ds['U']
    V   = od._ds['V'] 
    
    # Extract grid
    grid = od._grid
    
    # Message
    print('Computing shear component of strain')
        
    # Create DataArray
    # Same of vertical relative vorticity with + instead of -
    s_strain = (od._grid.diff(V*dyC, 'X', boundary='fill', fill_value=_np.nan) +
                od._grid.diff(U*dxC, 'Y',  boundary='fill', fill_value=_np.nan)) / rAz
    s_strain.attrs['units']     = 's^-1'
    s_strain.attrs['long_name'] = 'shear component of strain'

    # Create ds
    ds = _xr.Dataset({'s_strain': s_strain}, attrs=od.dataset.attrs)
    
    return _ospy.OceanDataset(ds).dataset

def normal_strain(od):
    """
    Compute normal component of strain.

    .. math::
        S_n = \\frac{\\partial u}{\\partial x}-\\frac{\\partial v}{\\partial y}
    
    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute 
    
    Returns
    -------
    ds: xarray.Dataset 
        | n_strain: normal component of strain
            
    See Also
    --------
    horizontal_divergence_velocity
    divergence
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    # Message
    print('Computing normal component of strain')
    
    # Create DataArray
    # Same of horizontal divergence of velocity field with - instead of +
    div = divergence(od, iName='U', jName='V', kName=None, aliases = False)
    n_strain = div['dU_dX'] - div['dV_dY'] 
    n_strain.attrs['units']      = 'm s^-2'
    n_strain.attrs['long_name']  = 'normal component of strain'
    
    # Create ds
    ds = _xr.Dataset({'n_strain': n_strain}, attrs=od.dataset.attrs)
    
    return _ospy.OceanDataset(ds).dataset 


def Okubo_Weiss_parameter(od):
    """
    Compute Okubo-Weiss parameter.  
    Vertical component of relative vorticity and shear component of strain are interpolated to C grid points.

    .. math::
        OW = S_n^2 + S_s^2 - \\zeta^2 = 
        \\left(\\frac{\\partial u}{\\partial x}-\\frac{\\partial v}{\\partial y}\\right)^2 + 
        \\left(\\frac{\\partial v}{\\partial x}+\\frac{\\partial u}{\\partial y}\\right)^2 -
        \\left(\\frac{\\partial v}{\\partial x}-\\frac{\\partial u}{\\partial y}\\right)^2
    
    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute
    
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
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
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
    print('Computing Okubo-Weiss parameter')
    
    # Interpolate to C grid points
    momVort3 = grid.interp(grid.interp(momVort3, 'X'), 'Y')
    s_strain = grid.interp(grid.interp(s_strain, 'X'), 'Y')
    
    # Create DataArray
    Okubo_Weiss = _np.square(s_strain) + _np.square(n_strain) - _np.square(momVort3)
    Okubo_Weiss.attrs['units']      = 's^-2'
    Okubo_Weiss.attrs['long_name']  = 'Okubo-Weiss parameter'
    
    # Create ds
    ds = _xr.Dataset({'Okubo_Weiss': Okubo_Weiss}, attrs=od.dataset.attrs)
    
    return _ospy.OceanDataset(ds).dataset


def Ertel_potential_vorticity(od):
    """
    Compute Ertel Potential Vorticity.  
    Interpolate all terms to C and Z points.

    .. math::
        Q = (f + \\zeta)\\frac{N^2}{g} +
            \\frac{\\left(\\mathbf{\\zeta_h}+e\\hat{\\mathbf{y}}\\right)\\cdot\\nabla_h\\rho}{\\rho_0}
    
    Parameters used: 
        | g
        | rho0
        | omega
        
    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute
    
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
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    # Add missing variables
    varList = ['fCoriG', 'momVort1', 'momVort2', 'momVort3', 'N2', 'YC']
    od = _add_missing_variables(od, varList)
    
    # Parameters
    paramsList = ['g', 'rho0', 'omega']
    params2use = {par:od.parameters[par] for par in od.parameters if par in paramsList}
    
    # Extract variables
    momVort1 = od._ds['momVort1']
    momVort2 = od._ds['momVort2']
    momVort3 = od._ds['momVort3']
    fCoriG   = od._ds['fCoriG']
    N2       = od._ds['N2']
    YC       = od._ds['YC']
    
    # Extract parameters
    g     = params2use['g']
    rho0  = params2use['rho0']
    omega = params2use['omega']
    
    # Extract grid
    grid = od._grid
    
    # Message
    print('Computing Ertel potential vorticity using the following parameters: {}'.format(params2use))
    
    # Compute Sigma0 gradients
    Sigma0_grads = gradient(od, varNameList='Sigma0', axesList=['X', 'Y'], aliases = False)
    
    # Interpolate fields
    fpluszeta  = grid.interp(grid.interp(momVort3 + fCoriG, 'X'), 'Y')
    N2         = grid.interp(N2, 'Z', to='center', boundary='fill', fill_value=_np.nan)
    momVort1   = grid.interp(grid.interp(momVort1, 'Y'), 'Z', to='center', boundary='fill', fill_value=_np.nan)
    momVort2   = grid.interp(grid.interp(momVort2, 'X'), 'Z', to='center', boundary='fill', fill_value=_np.nan)
    dSigma0_dX = grid.interp(Sigma0_grads['dSigma0_dX'], 'X')  
    dSigma0_dY = grid.interp(Sigma0_grads['dSigma0_dY'], 'Y')  
    _, e = _utils.Coriolis_parameter(YC, omega)
    
    # Create DataArray
    Ertel_PV = fpluszeta * N2 / g + (momVort1 * dSigma0_dX + (momVort2 + e) * dSigma0_dY)/rho0
    Ertel_PV.attrs['units']      = 'm^-1 s^-1'
    Ertel_PV.attrs['long_name']  = 'Ertel potential vorticity'
    Ertel_PV.attrs['OceanSpy_parameters'] = str(params2use)
    
    # Create ds
    ds = _xr.Dataset({'Ertel_PV': Ertel_PV}, attrs=od.dataset.attrs)
    
    return _ospy.OceanDataset(ds).dataset

def mooring_horizontal_volume_transport(od):
    """
    Compute horizontal volume flux through a mooring array section (in/outflow).  
    If the array is closed, transport at the first mooring is not computed.  
    Otherwise, transport at both the first and last mooring is not computed.  
    Transport can be computed following two paths, so the dimension `path` is added.

    .. math::
        T(mooring, Z, time, path) = T_x + T_y = u \\Delta y \\Delta z + v \\Delta x \\Delta z
    
    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute
    
    Returns
    -------
    ds: xarray.Dataset
        | transport     : Horizontal volume transport
        | Vtransport    : Meridional volume transport
        | Utransport    : Zonal volume transport
        | Y_transport   : Y coordinate of horizontal volume transport
        | X_transport   : X coordinate of horizontal volume transport
        | Y_Utransport  : Y coordinate of zonal volume transport
        | X_Utransport  : X coordinate of zonal volume transport
        | Y_Vtransport  : Y coordinate of meridional volume transport
        | X_Vtransport  : X coordinate of meridional volume transport
        | dir_Utransport: Direction of zonal volume transport
        | dir_Vtransport: Direction of meridional volume transport
            
    See Also
    --------
    subsample.mooring_array
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    if 'mooring' not in od._ds.dims:
        raise TypeError('oceadatasets must be subsampled using `subsample.mooring_array`')
        
    # Add missing variables    
    varList  = ['XC', 'YC', 'dyG', 'dxG', 'drF', 'U', 'V', 'HFacS', 'HFacW', 'XU', 'YU', 'XV', 'YV']
    od = _add_missing_variables(od, varList)
    
    # Message
    print('Computing horizontal volume transport')
    
    # Extract variables
    mooring = od._ds['mooring']
    XC      = od._ds['XC'].squeeze(('Y', 'X'))
    YC      = od._ds['YC'].squeeze(('Y', 'X'))
    XU      = od._ds['XU'].squeeze(('Y'))
    YU      = od._ds['YU'].squeeze(('Y'))
    XV      = od._ds['XV'].squeeze(('X'))
    YV      = od._ds['YV'].squeeze(('X'))
    
    # Compute transport
    U_tran = (od._ds['U'] * od._ds['dyG'] * od._ds['HFacW'] * od._ds['drF']).squeeze('Y')
    V_tran = (od._ds['V'] * od._ds['dxG'] * od._ds['HFacS'] * od._ds['drF']).squeeze('X')
    
    # Extract left and right values
    U1=U_tran.isel(Xp1=1); U0=U_tran.isel(Xp1=0)
    V1=V_tran.isel(Yp1=1); V0=V_tran.isel(Yp1=0)
    
    # Initialize direction
    U0_dir = _np.zeros((len(XC),2)); U1_dir = _np.zeros((len(XC),2))
    V0_dir = _np.zeros((len(YC),2)); V1_dir = _np.zeros((len(YC),2))
    
    # Steps
    diffX = _np.diff(XC); diffY = _np.diff(YC)
    
    # Closed array?
    if XC[0]==XC[-1] and YC[0]==YC[-1]:
        # Add first at the end
        closed = True
        diffX  = _np.append(diffX,diffX[0])
        diffY  = _np.append(diffY,diffY[0])
    else: 
        closed = False
        
    # Loop
    Usign = 1; Vsign = 1
    keepXf = False; keepYf = False
    for i in range(len(diffX)-1):
        if diffY[i]==0 and diffY[i+1]==0:   # Zonal
            V1_dir[i+1,:]=Vsign; V0_dir[i+1,:]=Vsign
        elif diffX[i]==0 and diffX[i+1]==0: # Meridional
            U1_dir[i+1,:]=Usign; U0_dir[i+1,:]=Usign
            
        # Corners
        elif (diffY[i]<0  and diffX[i+1]>0):  # |_
            Vsign=Usign; keepYf=keepXf
            U0_dir[i+1,:]=Usign; V0_dir[i+1,:]=Vsign  
        elif (diffY[i+1]>0  and diffX[i]<0):  
            Usign=Vsign; keepXf=keepYf
            U0_dir[i+1,:]=Usign; V0_dir[i+1,:]=Vsign 

        elif (diffY[i]>0  and diffX[i+1]>0): # |‾
            Vsign=-Usign; keepYf=not keepXf
            U0_dir[i+1,:]=Usign; V1_dir[i+1,:]=Vsign
        elif (diffY[i+1]<0  and diffX[i]<0):
            Usign=-Vsign; keepXf=not keepYf
            U0_dir[i+1,:]=Usign; V1_dir[i+1,:]=Vsign    

        elif (diffX[i]>0  and diffY[i+1]<0): # ‾|  
            Usign=Vsign; keepXf=keepYf
            V1_dir[i+1,:]=Vsign; U1_dir[i+1,:]=Usign
        elif (diffX[i+1]<0  and diffY[i]>0):  
            Vsign=Usign; keepYf=keepXf
            V1_dir[i+1,:]=Vsign; U1_dir[i+1,:]=Usign

        elif (diffX[i]>0  and diffY[i+1]>0): # _| 
            Usign=-Vsign; keepXf=not keepYf
            V0_dir[i+1,:]= Vsign; U1_dir[i+1,:]=Usign
        elif (diffX[i+1]<0  and diffY[i]<0):  
            Vsign=-Usign; keepYf=not keepXf
            V0_dir[i+1,:]= Vsign; U1_dir[i+1,:]=Usign 
    
        if keepXf: U1_dir[i+1,0]=0; U0_dir[i+1,1]=0
        else:      U0_dir[i+1,0]=0; U1_dir[i+1,1]=0
        if keepYf: V1_dir[i+1,0]=0; V0_dir[i+1,1]=0
        else:      V0_dir[i+1,0]=0; V1_dir[i+1,1]=0
            
    # Create direction DataArrays. 
    # Add a switch to return this? Useful to debug and/or plot velocities.
    U1_dir  = _xr.DataArray(U1_dir,  coords={'mooring': mooring, 'path': [0, 1]}, dims=('mooring', 'path'))
    U0_dir  = _xr.DataArray(U0_dir,  coords={'mooring': mooring, 'path': [0, 1]}, dims=('mooring', 'path'))
    V1_dir  = _xr.DataArray(V1_dir,  coords={'mooring': mooring, 'path': [0, 1]}, dims=('mooring', 'path'))
    V0_dir  = _xr.DataArray(V0_dir,  coords={'mooring': mooring, 'path': [0, 1]}, dims=('mooring', 'path'))
    
    # Mask first mooring
    U1_dir = U1_dir.where(U1_dir['mooring']!=U1_dir['mooring'].isel(mooring=0))
    U0_dir = U0_dir.where(U0_dir['mooring']!=U0_dir['mooring'].isel(mooring=0))
    V1_dir = V1_dir.where(V1_dir['mooring']!=V1_dir['mooring'].isel(mooring=0))
    V0_dir = V0_dir.where(V0_dir['mooring']!=V0_dir['mooring'].isel(mooring=0))
    
    if not closed:
        # Mask first mooring
        U1_dir = U1_dir.where(U1_dir['mooring']!=U1_dir['mooring'].isel(mooring=-1))
        U0_dir = U0_dir.where(U0_dir['mooring']!=U0_dir['mooring'].isel(mooring=-1))
        V1_dir = V1_dir.where(V1_dir['mooring']!=V1_dir['mooring'].isel(mooring=-1))
        V0_dir = V0_dir.where(V0_dir['mooring']!=V0_dir['mooring'].isel(mooring=-1))
    
    # Compute transport
    transport  = (U1*U1_dir+U0*U0_dir+V1*V1_dir+V0*V0_dir)*1.E-6
    transport.attrs['units']      = 'Sv'
    transport.attrs['long_name']  = 'Horizontal volume transport'
    Vtransport = (V1*V1_dir+V0*V0_dir)*1.E-6
    Vtransport.attrs['units']      = 'Sv'
    Vtransport.attrs['long_name']  = 'Meridional volume transport'
    Utransport = (U1*U1_dir+U0*U0_dir)*1.E-6
    Utransport.attrs['units']      = 'Sv'
    Utransport.attrs['long_name']  = 'Zonal volume transport'

    # Additional info
    Y_transport = YC
    Y_transport.attrs['long_name']  = 'Y coordinate of horizontal volume transport'
    X_transport = XC
    X_transport.attrs['long_name']  = 'X coordinate of horizontal volume transport'
    
    Y_Utransport = _xr.where(U1_dir!=0, YU.isel(Xp1=1), _np.nan)
    Y_Utransport = _xr.where(U0_dir!=0, YU.isel(Xp1=0), Y_Utransport)
    Y_Utransport.attrs['long_name']  = 'Y coordinate of zonal volume transport'
    X_Utransport = _xr.where(U1_dir!=0, XU.isel(Xp1=1), _np.nan)
    X_Utransport = _xr.where(U0_dir!=0, XU.isel(Xp1=0), X_Utransport)
    X_Utransport.attrs['long_name']  = 'X coordinate of zonal volume transport'
    
    Y_Vtransport = _xr.where(V1_dir!=0, YV.isel(Yp1=1), _np.nan)
    Y_Vtransport = _xr.where(V0_dir!=0, YV.isel(Yp1=0), Y_Vtransport)
    Y_Vtransport.attrs['long_name']  = 'Y coordinate of meridional volume transport'
    X_Vtransport = _xr.where(V1_dir!=0, XV.isel(Yp1=1), _np.nan)
    X_Vtransport = _xr.where(V0_dir!=0, XV.isel(Yp1=0), X_Vtransport)
    X_Vtransport.attrs['long_name']  = 'X coordinate of meridional volume transport'
    
    dir_Vtransport = _xr.where(V1_dir!=0, V1_dir, _np.nan)
    dir_Vtransport = _xr.where(V0_dir!=0, V0_dir, dir_Vtransport)
    dir_Vtransport.attrs['long_name']  = 'Direction of meridional volume transport'
    dir_Vtransport.attrs['units']      = '1: original, -1: flipped'
    dir_Utransport = _xr.where(U1_dir!=0, U1_dir, _np.nan)
    dir_Utransport = _xr.where(U0_dir!=0, U0_dir, dir_Utransport)
    dir_Utransport.attrs['long_name']  = 'Direction of zonal volume transport'
    dir_Utransport.attrs['units']      = '1: original, -1: flipped'
    
    # Create ds
    ds = _xr.Dataset({'transport'     : transport,
                      'Vtransport'    : Vtransport,
                      'Utransport'    : Utransport,
                      'Y_transport'   : Y_transport,
                      'X_transport'   : X_transport,
                      'Y_Utransport'  : Y_Utransport,
                      'X_Utransport'  : X_Utransport,
                      'Y_Vtransport'  : Y_Vtransport,
                      'X_Vtransport'  : X_Vtransport,
                      'dir_Utransport': dir_Utransport,
                      'dir_Vtransport': dir_Vtransport,}, attrs=od.dataset.attrs)
    
    return _ospy.OceanDataset(ds).dataset
    
    
def heat_budget(od):
    """
    Compute terms to close heat budget as explained by [Pie17]_.
    
    .. math::
        \\text{tendH = adv_hConvH + adv_vConvH + dif_vConvH + kpp_vConvH + forcH}
    
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
    This function is currently suited for the setup by [AHPM17]_: e.g., z* vertical coordinates, zero explicit diffusive fluxes, KPP.
        
    See Also
    --------
    gradient
    
    References
    ----------
    .. [Pie17] https://dspace.mit.edu/bitstream/handle/1721.1/111094/memo_piecuch_2017_evaluating_budgets_in_eccov4r3.pdf?sequence=1
    .. [AHPM17] Almansi, M., T.W. Haine, R.S. Pickart, M.G. Magaldi, R. Gelderloos, and D. Mastropole, 2017: High-Frequency Variability in the Circulation and Hydrography of the Denmark Strait Overflow from a High-Resolution Numerical Model. J. Phys. Oceanogr., 47, 2999–3013, https://doi.org/10.1175/JPO-D-17-0129.1
    """
  
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    # Add missing variables
    varList = ['Temp', 'Eta', 'Depth', 'ADVx_TH', 'ADVy_TH', 'ADVr_TH', 'DFrI_TH', 'KPPg_TH', 'TFLUX', 'oceQsw_AVG', 
               'time', 'HFacC', 'HFacW', 'HFacS', 'drF', 'rA', 'Zp1']
    od = _add_missing_variables(od, varList)
    
    # Parameters
    paramsList = ['rho0', 'c_p']
    params2use = {par:od.parameters[par] for par in od.parameters if par in paramsList}
                  
    # Message
    print('Computing heat budget terms using the following parameters: {}'.format(params2use))
    
    # Extract variables
    Temp       = od._ds['Temp']
    Eta        = od._ds['Eta']
    Depth      = od._ds['Depth']
    ADVx_TH    = od._ds['ADVx_TH']
    ADVy_TH    = od._ds['ADVy_TH']
    ADVr_TH    = od._ds['ADVr_TH']
    DFrI_TH    = od._ds['DFrI_TH']
    KPPg_TH    = od._ds['KPPg_TH']
    TFLUX      = od._ds['TFLUX']
    oceQsw_AVG = od._ds['oceQsw_AVG']
    HFacC      = od._ds['HFacC']
    HFacW      = od._ds['HFacW']
    HFacS      = od._ds['HFacS']
    drF        = od._ds['drF']
    rA         = od._ds['rA']
    Zp1        = od._ds['Zp1']
    
    # Extract parameters
    rho0 = od.parameters['rho0']
    c_p  = od.parameters['c_p']
    
    # Extract grid
    grid = od._grid
    
    # Compute useful variables
    dzMat     = drF * HFacC
    CellVol   = rA  * dzMat
    
    # Initialize dataset
    ds = _xr.Dataset({})
    
    # Total tendency
    z_star_scale = (1+Eta/Depth)
    od = od.add_DataArray((Temp*z_star_scale).where(HFacC!=0).rename('Tscaled'))
    units = 'degC/s'
    ds['tendH'] = gradient(od, 'Tscaled', 'time')['dTscaled_dtime']
    ds['tendH'].attrs['units']     = units
    ds['tendH'].attrs['long_name'] = 'Heat total tendency'
    ds['tendH'].attrs['OceanSpy_parameters'] = str(params2use)
    
    # Horizontal convergence
    ds['adv_hConvH'] = -(grid.diff(ADVx_TH.where(HFacW!=0),'X') +  grid.diff(ADVy_TH.where(HFacS!=0),'Y'))/CellVol
    ds['adv_hConvH'].attrs['units']     = units
    ds['adv_hConvH'].attrs['long_name'] = 'Heat horizontal advective convergence'
    ds['adv_hConvH'].attrs['OceanSpy_parameters'] = str(params2use)
    
    # Vertical convergence
    for i, (var_in, name_out, long_name) in enumerate(zip([ADVr_TH, DFrI_TH, KPPg_TH], 
                                                          ['adv_vConvH', 'dif_vConvH', 'kpp_vConvH'], 
                                                          ['advective', 'diffusive', 'kpp'])):
        ds[name_out] = grid.diff(var_in, 'Z', boundary='fill', fill_value=_np.nan).where(HFacC!=0)/CellVol
        ds[name_out].attrs['units'] = units
        ds[name_out].attrs['units'] = 'Heat vertical {} convergence'.format(long_name)
        ds[name_out].attrs['OceanSpy_parameters'] = str(params2use)
        
    
    # Surface flux
    # TODO: add these to parameters list?
    R       = 0.62
    zeta1   = 0.6
    zeta2   = 20
    q = (R * _np.exp(Zp1/zeta1) + (1-R)*_np.exp(Zp1/zeta2)).where(Zp1>=-200,0)
    forcH   = -grid.diff(q,'Z').where(HFacC!=0)
    if Zp1.isel(Zp1=0)==0:
        forcH_surf = (TFLUX-(1-forcH.isel(Z=0))*oceQsw_AVG).expand_dims('Z', Temp.dims.index('Z'))
        forcH_bott = forcH.isel(Z=slice(1,None))*oceQsw_AVG
        forcH = _xr.concat([forcH_surf, forcH_bott],dim='Z')
    else:
        forcH   = forcH * oceQsw_AVG
    ds['forcH'] = (forcH/(rho0*c_p*dzMat))
    ds['forcH'].attrs['units']     = units
    ds['forcH'].attrs['long_name'] = 'Heat surface forcing'
    ds['forcH'].attrs['OceanSpy_parameters'] = str(params2use)
    
    return _ospy.OceanDataset(ds).dataset

def salt_budget(od):
    """
    Compute terms to close salt budget as explained by [Pie17]_.
    
    .. math::
        \\text{tendS = adv_hConvS + adv_vConvS + dif_vConvS + kpp_vConvS + forcS}
    
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
    This function is currently suited for the setup by [AHPM17]_: e.g., z* vertical coordinates, zero explicit diffusive fluxes, KPP.
        
    See Also
    --------
    gradient
    
    References
    ----------
    .. [Pie17] https://dspace.mit.edu/bitstream/handle/1721.1/111094/memo_piecuch_2017_evaluating_budgets_in_eccov4r3.pdf?sequence=1
    .. [AHPM17] Almansi, M., T.W. Haine, R.S. Pickart, M.G. Magaldi, R. Gelderloos, and D. Mastropole, 2017: High-Frequency Variability in the Circulation and Hydrography of the Denmark Strait Overflow from a High-Resolution Numerical Model. J. Phys. Oceanogr., 47, 2999–3013, https://doi.org/10.1175/JPO-D-17-0129.1
    """
  
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    # Add missing variables
    varList = ['S', 'Eta', 'Depth', 'ADVx_SLT', 'ADVy_SLT', 'ADVr_SLT', 'DFrI_SLT', 'KPPg_SLT', 'SFLUX', 'oceSPtnd', 
               'time', 'HFacC', 'HFacW', 'HFacS', 'drF', 'rA', 'Zp1']
    od = _add_missing_variables(od, varList)
                  
    # Parameters
    paramsList = ['rho0']
    params2use = {par:od.parameters[par] for par in od.parameters if par in paramsList}
                  
    # Message
    print('Computing salt budget terms using the following parameters: {}'.format(params2use))
    
    # Extract variables
    S         = od._ds['S']
    Eta       = od._ds['Eta']
    Depth     = od._ds['Depth']
    ADVx_SLT  = od._ds['ADVx_SLT']
    ADVy_SLT  = od._ds['ADVy_SLT']
    ADVr_SLT  = od._ds['ADVr_SLT']
    DFrI_SLT  = od._ds['DFrI_SLT']
    KPPg_SLT  = od._ds['KPPg_SLT']
    SFLUX     = od._ds['SFLUX']
    oceSPtnd  = od._ds['oceSPtnd']
    HFacC     = od._ds['HFacC']
    HFacW     = od._ds['HFacW']
    HFacS     = od._ds['HFacS']
    drF       = od._ds['drF']
    rA        = od._ds['rA']
    Zp1        = od._ds['Zp1']
    
    # Extrac parameters
    rho0 = od.parameters['rho0']
    
    # Extract grid
    grid = od._grid
    
    # Compute useful variables
    dzMat     = drF * HFacC
    CellVol   = rA  * dzMat
    
    # Initialize dataset
    ds = _xr.Dataset({})
    
    # Total tendency
    z_star_scale = (1+Eta/Depth)
    od = od.add_DataArray((S*z_star_scale).where(HFacC!=0).rename('Sscaled'))
    units = 'psu/s'
    ds['tendS'] = gradient(od, 'Sscaled', 'time')['dSscaled_dtime']
    ds['tendS'].attrs['units']     = units
    ds['tendS'].attrs['long_name'] = 'Salt total tendency'
    ds['tendS'].attrs['OceanSpy_parameters'] = str(params2use)
    
    # Horizontal convergence
    ds['adv_hConvS'] = -(grid.diff(ADVx_SLT.where(HFacW!=0),'X') +  grid.diff(ADVy_SLT.where(HFacS!=0),'Y'))/CellVol
    ds['adv_hConvS'].attrs['units']     = units
    ds['adv_hConvS'].attrs['long_name'] = 'Salt horizontal advective convergence'
    ds['adv_hConvS'].attrs['OceanSpy_parameters'] = str(params2use)
    
    # Vertical convergence
    for i, (var_in, name_out, long_name) in enumerate(zip([ADVr_SLT, DFrI_SLT, KPPg_SLT], 
                                                          ['adv_vConvS', 'dif_vConvS', 'kpp_vConvS'], 
                                                          ['advective', 'diffusive', 'kpp'])):
        ds[name_out] = grid.diff(var_in, 'Z', boundary='fill', fill_value=_np.nan).where(HFacC!=0)/CellVol
        ds[name_out].attrs['units'] = units
        ds[name_out].attrs['units'] = 'Salt vertical {} convergence'.format(long_name)

    # Surface flux    
    forcS = oceSPtnd
    if Zp1.isel(Zp1=0)==0:
        forcS_surf = (SFLUX + forcS.isel(Z=0)).expand_dims('Z', S.dims.index('Z'))
        forcS_bott = forcS.isel(Z=slice(1,None))
        forcS = _xr.concat([forcS_surf, forcS_bott],dim='Z')
    ds['forcS'] = (forcS/(rho0*dzMat))
    ds['forcS'].attrs['units']     = units
    ds['forcS'].attrs['long_name'] = 'Salt surface forcing'
    ds['forcS'].attrs['OceanSpy_parameters'] = str(params2use)
    
    return _ospy.OceanDataset(ds).dataset

def geographical_aligned_velocities(od):
    """
    Compute zonal and meridional velocities from U and V on orthogonal curvilinear grid.
    
    .. math::
        (u_{zonal}, v_{merid}) = (u\\cos{\\phi} - v\\sin{\\phi}, u\\sin{\\phi} + v\\cos{\\phi})
        
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
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
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
    U = grid.interp(U, 'X')
    V = grid.interp(V, 'Y')
    
    # Compute velocities
    U_zonal = U * AngleCS - V * AngleSN
    if 'units' in od._ds['U'].attrs: U_zonal.attrs['units'] = od._ds['U'].attrs['units']
    U_zonal.attrs['long_name'] = 'zonal velocity'
    U_zonal.attrs['direction'] = 'positive: eastwards'
    
    V_merid = U * AngleSN + V * AngleCS
    if 'units' in od._ds['V'].attrs: V_merid.attrs['units'] = od._ds['V'].attrs['units']
    V_merid.attrs['long_name'] = 'meridional velocity'
    V_merid.attrs['direction'] = 'positive: northwards'
    
    # Create ds
    ds = _xr.Dataset({'U_zonal': U_zonal,
                      'V_merid': V_merid,}, attrs=od.dataset.attrs)
    
    return _ospy.OceanDataset(ds).dataset


def survey_aligned_velocities(od):
    """
    Compute horizontal velocities orthogonal and tengential to a survey.
    
    .. math::
        (v_{tan}, v_{ort}) = (u\\cos{\\phi} + v\\sin{\\phi}, v\\cos{\\phi} - u\\sin{\\phi})
        
    Parameters
    ----------
    od: OceanDataset
        oceandataset used to compute
        
    Returns
    -------
    ds: xarray.Dataset
        | rot_ang_Vel: Angle used to rotate geographical to survey aligned velocities
        | tan_Vel: Velocity component tangential to survey
        | ort_Vel: Velocity component orthogonal to survey
    """
    
    # Check input
    if not isinstance(od, _ospy.OceanDataset):
        raise TypeError('`od` must be OceanDataset')
        
    if 'station' not in od._ds.dims:
        raise TypeError('oceadatasets must be subsampled using `subsample.survey_stations`')
        
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
                        "\nAssuming U=U_zonal and V=V_merid"
                        "\nIf you are using curvilinear coordinates, run `compute.geographical_aligned_velocities` before `subsample.survey_stations`").format(e), stacklevel=2)
        
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
    # Translated from matlab: https://www.mathworks.com/help/map/ref/azimuth.html
    az = _np.arctan2(_np.cos(lat[1:]).values  * _np.sin(grid.diff(lon, 'station')),
                 _np.cos(lat[:-1]).values * _np.sin(lat[1:]).values - 
                 _np.sin(lat[:-1]).values * _np.cos(lat[1:]).values * _np.cos(grid.diff(lon, 'station'))) 
    az = grid.interp(az, 'station', boundary = 'extend')
    az = _xr.where(_np.rad2deg(az)<0, _np.pi*2 + az, az)
    
    # Compute rotation angle
    rot_ang_rad = _np.pi/2 - az
    rot_ang_rad = _xr.where(rot_ang_rad < 0, _np.pi*2 + rot_ang_rad, rot_ang_rad)
    rot_ang_deg =_np.rad2deg(rot_ang_rad)
    rot_ang_Vel = rot_ang_deg
    rot_ang_Vel.attrs['long_name'] = 'Angle used to rotate geographical to survey aligned velocities'
    rot_ang_Vel.attrs['units']     = 'deg (+: counterclockwise)'
    
    # Rotate velocities
    tan_Vel =  U*_np.cos(rot_ang_rad) + V*_np.sin(rot_ang_rad)
    tan_Vel.attrs['long_name'] = 'Velocity component tangential to survey'
    if 'units' in U.attrs: units = U.attrs['units']
    else:                  units = ''
    tan_Vel.attrs['units'] = units + ' (+: flow towards station indexed with higher number)'
    
    ort_Vel =  V*_np.cos(rot_ang_rad) - U*_np.sin(rot_ang_rad)
    ort_Vel.attrs['long_name'] = 'Velocity component orthogonal to survey'
    if 'units' in V.attrs: units = V.attrs['units']
    else:                  units = ''
    ort_Vel.attrs['units'] = units + ' (+: flow keeps station indexed with higher number to the right'
    
    # Create ds
    ds = _xr.Dataset({'rot_ang_Vel': rot_ang_Vel, 
                      'ort_Vel'    : ort_Vel,
                      'tan_Vel'    : tan_Vel,}, attrs=od.dataset.attrs)
    
    return _ospy.OceanDataset(ds).dataset





