"""
Compute: add new variables to the dataset
"""

# Comments for developers:

# 1) Use the same structure for each function:
#    - Input and output ds and info (add new variables to ds and info)
#    - Always add deep_copy option, check missing variables, and print a message
#    - Always add the following attribute da.attrs['history'] = 'Computed offline by OceanSpy'
#    - Return ds applying _utils.reorder_ds(ds)

# 2) Keep imported modules secret using _

import xarray as _xr
import numpy as _np
import xgcm as _xgcm 
from . import utils as _utils

def Sigma0(ds, info,
           deep_copy = False):
    """
    Compute potential density anomaly and add to dataset.
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    
    REFERENCES
    ----------
    .. [1] Jackett and McDougall, 1995 https://doi.org/10.1175/1520-0426(1995)012<0381:MAOHPT>2.0.CO;2
    """

    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['Temp', 'S']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing Sigma0')
    
    if info.parameters['eq_state'].lower()=='jmd95':
        """
        Adapted from jmd95.py:
        Density of Sea Water using Jackett and McDougall 1995 (JAOT 12) polynomial
        created by mlosch on 2002-08-09
        converted to python by jahn on 2010-04-29
        """
                
        # coefficients nonlinear equation of state in pressure coordinates for
        # 1. density of fresh water at p = 0
        eosJMDCFw = [ 999.842594,
                      6.793952e-02,
                     -9.095290e-03,
                      1.001685e-04,
                     -1.120083e-06,
                      6.536332e-09,
                    ]
        
        # 2. density of sea water at p = 0
        eosJMDCSw = [ 8.244930e-01,
                     -4.089900e-03,
                      7.643800e-05,
                     -8.246700e-07,
                      5.387500e-09,
                     -5.724660e-03,
                      1.022700e-04,
                     -1.654600e-06,
                      4.831400e-04,
                    ]
        
        # coefficients in pressure coordinates for
        # 3. secant bulk modulus K of fresh water at p = 0
        eosJMDCKFw = [ 1.965933e+04,
                       1.444304e+02,
                      -1.706103e+00,
                       9.648704e-03,
                      -4.190253e-05,
                     ]
        
        # 4. secant bulk modulus K of sea water at p = 0
        eosJMDCKSw = [ 5.284855e+01,
                      -3.101089e-01,
                       6.283263e-03,
                      -5.084188e-05,
                       3.886640e-01,
                       9.085835e-03,
                      -4.619924e-04,
                     ]
        
        # 5. secant bulk modulus K of sea water at p
        eosJMDCKP = [ 3.186519e+00,
                      2.212276e-02,
                     -2.984642e-04,
                      1.956415e-06,
                      6.704388e-03,
                     -1.847318e-04,
                      2.059331e-07,
                      1.480266e-04,
                      2.102898e-04,
                     -1.202016e-05,
                      1.394680e-07,
                     -2.040237e-06,
                      6.128773e-08,
                      6.207323e-10,
                    ]

        # Define variables
        t = ds[info.var_names['Temp']]
        s = ds[info.var_names['S']]  
        p = 0.

        # Useful stuff
        t2   = t*t
        t3   = t2*t
        t4   = t3*t
        s3o2 = s*_xr.ufuncs.sqrt(s)
        p2   = p*p
        
        # secant bulk modulus of fresh water at the surface
        bulkmod = (  eosJMDCKFw[0]
                   + eosJMDCKFw[1]*t
                   + eosJMDCKFw[2]*t2
                   + eosJMDCKFw[3]*t3
                   + eosJMDCKFw[4]*t4
                  )
        
        # secant bulk modulus of sea water at the surface
        bulkmod = (  bulkmod
                   + s*(     eosJMDCKSw[0]
                           + eosJMDCKSw[1]*t
                           + eosJMDCKSw[2]*t2
                           + eosJMDCKSw[3]*t3
                       )
                   + s3o2*(  eosJMDCKSw[4]
                           + eosJMDCKSw[5]*t
                           + eosJMDCKSw[6]*t2
                          )
                  )
        
        # secant bulk modulus of sea water at pressure p
        bulkmod = (  bulkmod
                   + p*(      eosJMDCKP[0]
                            + eosJMDCKP[1]*t
                            + eosJMDCKP[2]*t2
                            + eosJMDCKP[3]*t3
                       )
                   + p*s*(    eosJMDCKP[4]
                            + eosJMDCKP[5]*t
                            + eosJMDCKP[6]*t2
                         )
                   + p*s3o2*eosJMDCKP[7]
                   + p2*(     eosJMDCKP[8]
                            + eosJMDCKP[9]*t
                            + eosJMDCKP[10]*t2
                        )
                   + p2*s*(  eosJMDCKP[11]
                           + eosJMDCKP[12]*t
                           + eosJMDCKP[13]*t2
                          )
                  )

        # density of freshwater at the surface
        rho = (  eosJMDCFw[0]
               + eosJMDCFw[1]*t
               + eosJMDCFw[2]*t2
               + eosJMDCFw[3]*t3
               + eosJMDCFw[4]*t4
               + eosJMDCFw[5]*t4*t
              )
        
        # density of sea water at the surface
        rho = (  rho
               + s*( eosJMDCSw[0]
                   + eosJMDCSw[1]*t
                   + eosJMDCSw[2]*t2
                   + eosJMDCSw[3]*t3
                   + eosJMDCSw[4]*t4
                   )
               + s3o2*(
                     eosJMDCSw[5]
                   + eosJMDCSw[6]*t
                   + eosJMDCSw[7]*t2
                      )
                   + eosJMDCSw[8]*s*s
              )

        # Compute density
        rho = rho / (1. - p/bulkmod)
    
    # Compute anomaly
    Sigma0 = rho - 1000
    
    # Create DataArray
    Sigma0.attrs['units']     = 'kg/m^3'
    Sigma0.attrs['long_name'] = 'potential density anomaly'
    Sigma0.attrs['history']   = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['Sigma0'] = Sigma0
    
    # Update var_names
    info.var_names['Sigma0'] = 'Sigma0'
    
    return _utils.reorder_ds(ds), info

def N2(ds, info,
       deep_copy = False):
    """
    Compute Brunt-Väisälä Frequency and add to dataset.
    -(g/rho0)(dSigma0/dZ)
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
    
    # Add missing variables
    varList = ['Sigma0', 'HFacC', 'drC', 'Z']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing N2')
    
    # Variables
    Sigma0 = ds[info.var_names['Sigma0']]
    HFacC  = ds[info.var_names['HFacC']]
    drC    = _xr.ufuncs.sign(ds['Z'][-1]-ds['Z'][0]) * ds[info.var_names['drC']]
    
    # Parameters
    g    = info.parameters['g'] # m/s^2
    rho0 = info.parameters['rho0'] # m/s^2
    
    # Compute Brunt-Vaisala   
    N2 =  ( - g / rho0   
            * info.grid.diff(Sigma0, 'Z', to='outer',
                             boundary='fill', fill_value=float('nan'))
            * info.grid.interp(HFacC, 'Z', to='outer',
                               boundary='fill', fill_value=float('nan'))
            / (drC)
          )
    
    # Create DataArray
    N2.attrs['units']     = 's^-2'
    N2.attrs['long_name'] = 'Brunt-Väisälä Frequency'
    N2.attrs['history']   = 'Computed offline by OceanSpy'

    # Add to dataset
    ds['N2'] = N2
    
    # Update var_names
    info.var_names['N2'] = 'N2'
    
    return _utils.reorder_ds(ds), info

def momVort1(ds, info,
             deep_copy = False):

    """
    Compute 1st component of Vorticity and add to dataset.
    dW/dY - dV/dZ
    Use the same discretization of the 3rd component:
    http://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#relative-vorticity
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['dyC', 'drC', 'W', 'V']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing momVort1')
    
    # Variables
    dyC = _xr.ufuncs.sign(ds['Y'][-1]-ds['Y'][0]) * ds[info.var_names['dyC']]
    drC = _xr.ufuncs.sign(ds['Z'][-1]-ds['Z'][0]) * ds[info.var_names['drC']]
    W = ds[info.var_names['W']]
    V = ds[info.var_names['V']]
    
    # Compute momVort1
    drC = drC[:-1]
    drC = drC.rename({'Zp1': 'Zl'})
    momVort1 = (info.grid.diff(W * drC, 'Y', boundary='fill', fill_value=float('nan')) -
                info.grid.diff(V * dyC, 'Z', to='right', boundary='fill', fill_value=float('nan'))
               ) / (dyC * drC)
    
    # Create DataArray
    momVort1.attrs['units']     = 's^-1'
    momVort1.attrs['long_name'] = '1st component of Vorticity'
    momVort1.attrs['history']   = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['momVort1'] = momVort1
    
    # Update var_names
    info.var_names['momVort1'] = 'momVort1'
    
    return _utils.reorder_ds(ds), info
    
def momVort2(ds, info,
             deep_copy = False):
    """
    Compute 2nd component of Vorticity and add to dataset.
    dU/dZ - dW/dX
    Use the same discretization of the 3rd component:
    http://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#relative-vorticity
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['dxC', 'drC', 'W', 'U']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing momVort2')
    
    # Variables
    dxC = _xr.ufuncs.sign(ds['X'][-1]-ds['X'][0]) * ds[info.var_names['dxC']]
    drC = _xr.ufuncs.sign(ds['Z'][-1]-ds['Z'][0]) * ds[info.var_names['drC']]
    W = ds[info.var_names['W']]
    U = ds[info.var_names['U']]
    
    # Compute momVort2
    drC = drC[:-1]
    drC = drC.rename({'Zp1': 'Zl'})
    momVort2 = (info.grid.diff(U * dxC, 'Z', to='right', boundary='fill', fill_value=float('nan')) -
                info.grid.diff(W * drC, 'X', boundary='fill', fill_value=float('nan'))
               ) / (dxC * drC)
    
    # Create DataArray
    momVort2.attrs['units']     = 's^-1'
    momVort2.attrs['long_name'] = '2nd component of Vorticity'
    momVort2.attrs['history']   = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['momVort2'] = momVort2
    
    # Update var_names
    info.var_names['momVort2'] = 'momVort2'
    
    return _utils.reorder_ds(ds), info
    
def momVort3(ds, info,
             deep_copy = False):
    """
    Compute 3rd component of Vorticity and add to dataset.
    dV/dX - dU/dY 
    http://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#relative-vorticity
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['rAz', 'dxC', 'dyC', 'U', 'V']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing momVort3')
    
    # Variables
    rAz = ds[info.var_names['rAz']]
    dxC = _xr.ufuncs.sign(ds['X'][-1]-ds['X'][0]) * ds[info.var_names['dxC']]
    dyC = _xr.ufuncs.sign(ds['Y'][-1]-ds['Y'][0]) * ds[info.var_names['dyC']]
    U   = ds[info.var_names['U']]
    V   = ds[info.var_names['V']]
    
    # Compute momVort3
    momVort3 = (info.grid.diff(V * dyC, 'X', boundary='fill', fill_value=float('nan')) -
                info.grid.diff(U * dxC, 'Y', boundary='fill', fill_value=float('nan'))
               ) /  rAz
    
    # Create DataArray
    momVort3.attrs['units']     = 's^-1'
    momVort3.attrs['long_name'] = '3rd component (vertical) of Vorticity'
    momVort3.attrs['history']   = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['momVort3'] = momVort3
    
    # Update var_names
    info.var_names['momVort3'] = 'momVort3'
    
    return _utils.reorder_ds(ds), info

def shear_strain(ds, info,
                 deep_copy = False):
    """
    Compute shear component of strain.
    dV/dX + dU/dY 
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['rAz', 'dxC', 'dyC', 'U', 'V']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing shear_strain')
    
    # Variables
    rAz = ds[info.var_names['rAz']]
    dxC = _xr.ufuncs.sign(ds['X'][-1]-ds['X'][0]) * ds[info.var_names['dxC']]
    dyC = _xr.ufuncs.sign(ds['Y'][-1]-ds['Y'][0]) * ds[info.var_names['dyC']]
    U   = ds[info.var_names['U']]
    V   = ds[info.var_names['V']]
    
    # Compute shear_strain
    shear_strain = (info.grid.diff(V * dyC, 'X', boundary='fill', fill_value=float('nan')) +
                    info.grid.diff(U * dxC, 'Y', boundary='fill', fill_value=float('nan'))
                    ) /  rAz
    
    # Create DataArray
    shear_strain.attrs['units']     = 's^-1'
    shear_strain.attrs['long_name'] = 'Shear component of Strain'
    shear_strain.attrs['history']   = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['shear_strain'] = shear_strain
    
    # Update var_names
    info.var_names['shear_strain'] = 'shear_strain'
    
    return _utils.reorder_ds(ds), info


def hor_div(ds, info,
            deep_copy = False):
    """
    Compute horizontal divergence.
    dU/dX + dV/dY 
    https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#horizontal-divergence
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['U', 'V', 'dyG', 'dxG', 'HFacW', 'HFacS', 'rA', 'HFacC']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing hor_div')
    
    # Variables
    U     = ds[info.var_names['U']]
    V     = ds[info.var_names['V']]
    dyG   = _xr.ufuncs.sign(ds['Yp1'][-1]-ds['Yp1'][0]) * ds[info.var_names['dyG']]
    dxG   = _xr.ufuncs.sign(ds['Xp1'][-1]-ds['Xp1'][0]) * ds[info.var_names['dxG']]
    HFacW = ds[info.var_names['HFacW']]
    HFacS = ds[info.var_names['HFacS']]
    HFacC = ds[info.var_names['HFacC']]
    rA    = ds[info.var_names['rA']]
    
    # Compute hor_div
    hor_div = (info.grid.diff(U * dyG * HFacW,'X') + 
               info.grid.diff(V * dxG * HFacS,'Y')) / (rA * HFacC)
    
    # Create DataArray
    hor_div.attrs['units']     = 's^-1'
    hor_div.attrs['long_name'] = 'Horizontal divergence'
    hor_div.attrs['history']   = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['hor_div'] = hor_div
    
    # Update var_names
    info.var_names['hor_div'] = 'hor_div'
    
    return _utils.reorder_ds(ds), info

def normal_strain(ds, info,
                  deep_copy = False):
    """
    Compute normal component of strain.
    dU/dX - dV/dY 
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['U', 'V', 'dyG', 'dxG', 'HFacW', 'HFacS', 'rA', 'HFacC']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing normal_strain')
    
    # Variables
    U     = ds[info.var_names['U']]
    V     = ds[info.var_names['V']]
    dyG   = _xr.ufuncs.sign(ds['Yp1'][-1]-ds['Yp1'][0]) * ds[info.var_names['dyG']]
    dxG   = _xr.ufuncs.sign(ds['Xp1'][-1]-ds['Xp1'][0]) * ds[info.var_names['dxG']]
    HFacW = ds[info.var_names['HFacW']]
    HFacS = ds[info.var_names['HFacS']]
    HFacC = ds[info.var_names['HFacC']]
    rA    = ds[info.var_names['rA']]
    
    # Compute normal_strain
    normal_strain = (info.grid.diff(U * dyG * HFacW,'X') - 
                     info.grid.diff(V * dxG * HFacS,'Y')) / (rA * HFacC)
    
    # Create DataArray
    normal_strain.attrs['units']     = 's^-1'
    normal_strain.attrs['long_name'] = 'Normal component of Strain'
    normal_strain.attrs['history']   = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['normal_strain'] = normal_strain
    
    # Update var_names
    info.var_names['normal_strain'] = 'normal_strain'
    
    return _utils.reorder_ds(ds), info

def Okubo_Weiss(ds, info,
                deep_copy = False):
    """
    Compute Okubo-Weiss parameter.
    OW = normal_strain^2 + shear_strain^2 - momVort3^2 
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['normal_strain', 'shear_strain', 'momVort3']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing Okubo_Weiss')
    
    # Variables
    normal_strain = ds[info.var_names['normal_strain']]
    shear_strain  = ds[info.var_names['shear_strain']]
    momVort3      = ds[info.var_names['momVort3']]
    
    # Interpolate vorticity and shear strain
    shear_strain = info.grid.interp(shear_strain, 'X', boundary='fill', fill_value=float('nan'))
    shear_strain = info.grid.interp(shear_strain, 'Y', boundary='fill', fill_value=float('nan'))
    momVort3 = info.grid.interp(momVort3, 'X', boundary='fill', fill_value=float('nan'))
    momVort3 = info.grid.interp(momVort3, 'Y', boundary='fill', fill_value=float('nan'))
    
    # Compute Okubo_Weiss
    Okubo_Weiss = (_xr.ufuncs.square(normal_strain) + 
                   _xr.ufuncs.square(shear_strain)  - 
                   _xr.ufuncs.square(momVort3)      )
    
    # Create DataArray
    Okubo_Weiss.attrs['units']     = 's^-2'
    Okubo_Weiss.attrs['long_name'] = 'Okubo-Weiss parameter'
    Okubo_Weiss.attrs['history']   = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['Okubo_Weiss'] = Okubo_Weiss
    
    # Update var_names
    info.var_names['Okubo_Weiss'] = 'Okubo_Weiss'
    
    return _utils.reorder_ds(ds), info


def Ertel_PV(ds, info,
             deep_copy = False):
    """
    Compute Ertel Potential Vorticity and add to dataset.
    Eq. 2.25 in OC3D, Klinger and Haine, 2018.
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['Y', 'fCori', 'dxC', 'dyC', 'Sigma0', 'N2', 'momVort1', 'momVort2', 'momVort3']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing Ertel_PV')
    
    # Variables
    Y        = ds[info.var_names['Y']]
    fCori    = ds[info.var_names['fCori']]
    dxC      = _xr.ufuncs.sign(ds['X'][-1]-ds['X'][0]) * ds[info.var_names['dxC']]
    dyC      = _xr.ufuncs.sign(ds['Y'][-1]-ds['Y'][0]) * ds[info.var_names['dyC']]
    Sigma0   = ds[info.var_names['Sigma0']]
    N2       = ds[info.var_names['N2']]
    momVort1 = ds[info.var_names['momVort1']]
    momVort2 = ds[info.var_names['momVort2']]
    momVort3 = ds[info.var_names['momVort3']]
    
    # Parameters
    omega = info.parameters['omega']
    g     = info.parameters['g']
    
    # Interpolate relative vorticity and N2
    N2       = info.grid.interp(N2, 'Z')
    
    momVort1 = info.grid.interp(momVort1, 'Y', boundary='fill', fill_value=float('nan'))
    momVort1 = info.grid.interp(momVort1, 'Z', boundary='fill', fill_value=float('nan'))
 
    momVort2 = info.grid.interp(momVort2, 'X', boundary='fill', fill_value=float('nan'))
    momVort2 = info.grid.interp(momVort2, 'Z', boundary='fill', fill_value=float('nan'))
    
    momVort3 = info.grid.interp(momVort3, 'X', boundary='fill', fill_value=float('nan'))
    momVort3 = info.grid.interp(momVort3, 'Y', boundary='fill', fill_value=float('nan'))
    
    # Compute Ertel PV
    e = 2 * omega * _xr.ufuncs.cos(_xr.ufuncs.deg2rad(Y))
    
    dS0dx = info.grid.diff(Sigma0, 'X', boundary='fill', fill_value=float('nan')) / dxC
    dS0dx = info.grid.interp(dS0dx,'X', boundary='fill', fill_value=float('nan'))
    dS0dy = info.grid.diff(Sigma0, 'Y', boundary='fill', fill_value=float('nan')) / dyC
    dS0dy = info.grid.interp(dS0dy,'Y', boundary='fill', fill_value=float('nan'))

    PV_ver = (momVort3 + fCori) * N2 / g
    PV_hor = (momVort1 * dS0dx  + (momVort2 + e ) * dS0dy) / info.parameters['rho0']
    Ertel_PV = PV_ver + PV_hor
    
    # Create DataArray
    Ertel_PV.attrs['units']     = '(m*s)^-1'
    Ertel_PV.attrs['long_name'] = 'Ertel Potential Vorticity'
    Ertel_PV.attrs['history']   = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['Ertel_PV'] = Ertel_PV
    
    # Update var_names
    info.var_names['Ertel_PV'] = 'Ertel_PV'
    
    return _utils.reorder_ds(ds), info

def KE(ds, info,
       deep_copy = False):
    """
    Compute Kinetic Energy and add to dataset.
    http://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#kinetic-energy
    Note: non-hydrostatic term is omitted (\epsilon_{nh}=0)
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """    
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['U', 'V', 'HFacW', 'HFacS']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing KE')
    
    # Variables
    U     = ds[info.var_names['U']]
    V     = ds[info.var_names['V']]
    HFacW = ds[info.var_names['HFacW']]
    HFacS = ds[info.var_names['HFacS']]
    
    # Compute Eddy Kinetic Energy
    KE = (info.grid.interp(_xr.ufuncs.square(U*HFacW), 'X') + 
          info.grid.interp(_xr.ufuncs.square(V*HFacS), 'Y')) / 2
    
    # Create DataArray
    KE.attrs['units']     = 'm^2/s^2'
    KE.attrs['long_name'] = 'Kinetic Energy'
    KE.attrs['history']   = 'Computed offline by OceanSpy'

    # Add to dataset
    ds['KE'] = KE
    
    # Update var_names
    info.var_names['KE'] = 'KE'
    
    return _utils.reorder_ds(ds), info

def EKE(ds, info,
        deep_copy = False):
    """
    Compute Eddy Kinetic Energy and add to dataset.
    http://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#kinetic-energy
    Note: non-hydrostatic term is omitted (\epsilon_{nh}=0)
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """    
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['U', 'V', 'HFacW', 'HFacS']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing EKE')
    
    # Variables
    U     = ds[info.var_names['U']]
    V     = ds[info.var_names['V']]
    HFacW = ds[info.var_names['HFacW']]
    HFacS = ds[info.var_names['HFacS']]
    
    # Compute Eddy Kinetic Energy
    u_mean = U.mean('time', skipna=True)
    v_mean = V.mean('time', skipna=True)
    u_prime = U - u_mean
    v_prime = V - v_mean
    EKE = (info.grid.interp(_xr.ufuncs.square(u_prime*HFacW), 'X') + 
           info.grid.interp(_xr.ufuncs.square(v_prime*HFacS), 'Y')) / 2
        
    # Create DataArray
    EKE.attrs['units']     = 'm^2/s^2'
    EKE.attrs['long_name'] = 'Eddy Kinetic Energy'
    EKE.attrs['history']   = 'Computed offline by OceanSpy'

    # Add to dataset
    ds['EKE'] = EKE
    
    # Update var_names
    info.var_names['EKE'] = 'EKE'
    
    return _utils.reorder_ds(ds), info

def tan_Vel(ds, info,
            deep_copy = False):
    """
    Compute velocity component tangential to a vertical section,
    and add to dataset.
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """    
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['U', 'V']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing tan_Vel')
    
    # Define dist, lats, lons and find rotation angle
    try:
        dist = ds[info.var_names['dist_VS']]
        lats = ds[info.var_names['lat_VS']]
        lons = ds[info.var_names['lon_VS']]
    except ValueError: raise ValueError("'ds' must be a vertical section. Info needed: dist_VS, lat_VS, lon_VS")
    rot_ang = _utils.rotation_angle(dist, lats, lons)
  
    # Rotate velocities
    U = ds[info.var_names['U']]
    V = ds[info.var_names['V']]
    tan_Vel =  U*_np.cos(rot_ang) + V*_np.sin(rot_ang)
    
    # Create DataArray
    if 'units' in U.attrs: tan_Vel.attrs['units'] = U.attrs['units']
    tan_Vel.attrs['long_name']      = 'tangential velocity'
    tan_Vel.attrs['direction']      = 'positive: flow towards larger distances'
    tan_Vel.attrs['rotation_angle'] = str(_np.rad2deg(rot_ang.values)) + ' deg (positive: counterclockwise)'
    tan_Vel.attrs['history']        = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['tan_Vel'] = tan_Vel
    
    # Update var_names
    info.var_names['tan_Vel'] = 'tan_Vel'
    
    return _utils.reorder_ds(ds), info

def ort_Vel(ds, info,
            deep_copy = False):
    """
    Compute velocity component orthogonal to a vertical section,
    and add to dataset.
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """    
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['U', 'V']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing ort_Vel')
    
    # Define dist, lats, lons and find rotation angle
    try:
        dist = ds[info.var_names['dist_VS']]
        lats = ds[info.var_names['lat_VS']]
        lons = ds[info.var_names['lon_VS']]
    except ValueError: raise ValueError("'ds' must be a vertical section. Info needed: dist_VS, lat_VS, lon_VS")
    rot_ang = _utils.rotation_angle(dist, lats, lons)
  
    # Rotate velocities
    U = ds[info.var_names['U']]
    V = ds[info.var_names['V']]
    ort_Vel =  V*_np.cos(rot_ang) - U*_np.sin(rot_ang)
    
    # Create DataArray
    if 'units' in U.attrs: ort_Vel.attrs['units'] = U.attrs['units']
    ort_Vel.attrs['long_name']      = 'orthogonal velocity'
    ort_Vel.attrs['direction']      = 'positive: flow keeps larger distances to the right'
    ort_Vel.attrs['rotation_angle'] = str(_np.rad2deg(rot_ang.values)) + ' deg (positive: counterclockwise)'
    ort_Vel.attrs['history']        = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['ort_Vel'] = ort_Vel
    
    # Update var_names
    info.var_names['ort_Vel'] = 'ort_Vel'
    
    return _utils.reorder_ds(ds), info

def heat_budget(ds, info,
                deep_copy = False):
    """
    Compute terms to close heat budget as explained in [1]_, and add to dataset.
    
    Terms:    
        | tendH: Heat total tendency    
        | adv_hConvH: Heat horizontal advective convergence  
        | adv_vConvH: Heat vertical advective convergence  
        | dif_vConvH: Heat vertical diffusive convergence  
        | kpp_vConvH: Heat vertical kpp convergence  
        | forcH: Heat surface forcing  
    
    Budget is closed if tendH = adv_hConvH + adv_vConvH + dif_vConvH + kpp_vConvH + forcH
    Vertical convergences cannot be estimated for the last vertical level (nans are returned)
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    
    REFERENCES
    ----------
    .. [1] Piecuch, 2017 ftp://ecco.jpl.nasa.gov/Version4/Release3/doc/evaluating_budgets_in_eccov4r3.pdf
    """
  
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['Temp', 'Eta', 'Depth', 'ADVx_TH', 'ADVy_TH', 'ADVr_TH', 'DFrI_TH', 'KPPg_TH', 'TFLUX', 'oceQsw_AVG', 
               'time', 'HFacC', 'HFacW', 'HFacS', 'drF', 'rA']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing heat budget terms')
    
    # Variables
    Temp       = ds[info.var_names['Temp']]
    Eta        = ds[info.var_names['Eta']]
    Depth      = ds[info.var_names['Depth']]
    ADVx_TH    = ds[info.var_names['ADVx_TH']]
    ADVy_TH    = ds[info.var_names['ADVy_TH']]
    ADVr_TH    = ds[info.var_names['ADVr_TH']]
    DFrI_TH    = ds[info.var_names['DFrI_TH']]
    KPPg_TH    = ds[info.var_names['KPPg_TH']]
    TFLUX      = ds[info.var_names['TFLUX']]
    oceQsw_AVG = ds[info.var_names['oceQsw_AVG']]
    HFacC     = ds[info.var_names['HFacC']]
    HFacW     = ds[info.var_names['HFacW']]
    HFacS     = ds[info.var_names['HFacS']]
    drF     = ds[info.var_names['drF']]
    rA      = ds[info.var_names['rA']]
    dt      = ds[info.var_names['dt']]
    
    # Parameters
    rho0 = info.parameters['rho0']
    c_p  = info.parameters['c_p']
    
    # Compute useful grid-factor variables
    HFacC_Zl  = info.grid.interp(HFacC,'Z', boundary='fill', fill_value=0, to='right')
    HFacC_Zp1 = info.grid.interp(HFacC,'Z', boundary='fill', fill_value=0, to='outer')
    dzMat   = drF * HFacC
    CellVol = rA  * dzMat
    
    # Total tendency
    z_star_scale = (1+Eta/Depth)
    tendH = info.grid.diff((Temp*z_star_scale).where(HFacC!=0),'time')/dt
    
    # Horizontal convergence
    adv_hConvH = -(info.grid.diff(ADVx_TH.where(HFacW!=0),'X') + 
                   info.grid.diff(ADVy_TH.where(HFacS!=0),'Y'))/CellVol
    
    # Vertical convergence
    for i in range(3):
        if   i==0: var_in = ADVr_TH
        elif i==1: var_in = DFrI_TH
        elif i==2: var_in = KPPg_TH

        var_out = var_in.where(HFacC_Zl!=0).diff('Zl')
        var_out = var_out.drop('Zl').rename({'Zl':'Z'}).assign_coords(Z=ds['Z'].isel(Z=slice(None,-1)))
        tmp     = _xr.DataArray(_np.zeros(var_out.isel(Z=-1).shape),
                               coords=var_out.isel(Z=-1).coords,
                               dims=var_out.isel(Z=-1).dims)
        tmp = tmp.assign_coords(Z=ds['Z'].isel(Z=-1))
        var_out = _xr.concat([var_out, tmp], 'Z')
        var_out = var_out.where(var_out['Z']!=var_out['Z'].isel(Z=-1))
        var_out = var_out/CellVol

        if   i==0: adv_vConvH = var_out
        elif i==1: dif_vConvH = var_out 
        elif i==2: kpp_vConvH = var_out 
    
    # Surface flux
    R       = 0.62
    zeta1   = 0.6
    zeta2   = 20
    q = (R * _xr.ufuncs.exp(ds['Zp1']/zeta1) + (1-R)*_xr.ufuncs.exp(ds['Zp1']/zeta2)).where(ds['Zp1']>=-200,0)
    forcH   = -info.grid.diff(q.where(HFacC_Zp1!=0),'Z')
    if ds['Zp1'].isel(Zp1=0)==0:
        forcH_surf = (TFLUX-(1-forcH.isel(Z=0))*oceQsw_AVG).expand_dims('Z')
        forcH_bott = forcH.isel(Z=slice(1,None))*oceQsw_AVG
        forcH = _xr.concat([forcH_surf, forcH_bott],dim='Z')
    else:
        forcH   = forcH * oceQsw_AVG
    forcH   = (forcH/(rho0*c_p*dzMat))
    
    
    # Create DataArrays
    tendH.attrs['units']     = 'degC/s'
    tendH.attrs['long_name'] = 'Heat total tendency'
    tendH.attrs['history']   = 'Computed offline by OceanSpy'
    
    adv_hConvH.attrs['units']     = 'degC/s'
    adv_hConvH.attrs['long_name'] = 'Heat horizontal advective convergence'
    adv_hConvH.attrs['history']   = 'Computed offline by OceanSpy'
    
    adv_vConvH.attrs['units']     = 'degC/s'
    adv_vConvH.attrs['long_name'] = 'Heat vertical advective convergence'
    adv_vConvH.attrs['history']   = 'Computed offline by OceanSpy'
    
    dif_vConvH.attrs['units']     = 'degC/s'
    dif_vConvH.attrs['long_name'] = 'Heat vertical diffusive convergence'
    dif_vConvH.attrs['history']   = 'Computed offline by OceanSpy'
    
    kpp_vConvH.attrs['units']     = 'degC/s'
    kpp_vConvH.attrs['long_name'] = 'Heat vertical kpp convergence'
    kpp_vConvH.attrs['history']   = 'Computed offline by OceanSpy'
    
    forcH.attrs['units']     = 'degC/s'
    forcH.attrs['long_name'] = 'Heat surface forcing'
    forcH.attrs['history']   = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['tendH']      = tendH
    ds['adv_hConvH'] = adv_hConvH
    ds['adv_vConvH'] = adv_vConvH
    ds['dif_vConvH'] = dif_vConvH
    ds['kpp_vConvH'] = kpp_vConvH
    ds['forcH']      = forcH
    
    # Update var_names
    info.var_names['tendH']      = 'tendH'
    info.var_names['adv_hConvH'] = 'adv_hConvH'
    info.var_names['adv_vConvH'] = 'adv_vConvH'
    info.var_names['dif_vConvH'] = 'dif_vConvH'
    info.var_names['kpp_vConvH'] = 'kpp_vConvH'
    info.var_names['forcH']      = 'forcH'
    
    return _utils.reorder_ds(ds), info

def salt_budget(ds, info,
                deep_copy = False):
    """
    Compute terms to close salt budget as explained in [1]_, and add to dataset.
    
    Terms:    
        | tendS: Salt total tendency    
        | adv_hConvS: Salt horizontal advective convergence  
        | adv_vConvS: Salt vertical advective convergence  
        | dif_vConvS: Salt vertical diffusive convergence  
        | kpp_vConvS: Salt vertical kpp convergence  
        | forcS: Salt surface forcing  
    
    Budget is closed if tendS = adv_hConvS + adv_vConvS + dif_vConvS + kpp_vConvS + forcS
    Vertical convergences cannot be estimated for the last vertical level (nans are returned)
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    
    REFERENCES
    ----------
    .. [1] Piecuch, 2017 ftp://ecco.jpl.nasa.gov/Version4/Release3/doc/evaluating_budgets_in_eccov4r3.pdf
    """

    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
        
    # Add missing variables
    varList = ['S', 'Eta', 'Depth', 'ADVx_SLT', 'ADVy_SLT', 'ADVr_SLT', 'DFrI_SLT', 'KPPg_SLT', 'SFLUX', 'oceSPtnd', 
               'time', 'HFacC', 'HFacW', 'HFacS', 'drF', 'rA']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing heat budget terms')
    
    # Variables
    S        = ds[info.var_names['S']]
    Eta      = ds[info.var_names['Eta']]
    Depth    = ds[info.var_names['Depth']]
    ADVx_SLT = ds[info.var_names['ADVx_SLT']]
    ADVy_SLT = ds[info.var_names['ADVy_SLT']]
    ADVr_SLT = ds[info.var_names['ADVr_SLT']]
    DFrI_SLT = ds[info.var_names['DFrI_SLT']]
    KPPg_SLT = ds[info.var_names['KPPg_SLT']]
    SFLUX    = ds[info.var_names['SFLUX']]
    oceSPtnd = ds[info.var_names['oceSPtnd']]
    HFacC    = ds[info.var_names['HFacC']]
    HFacW    = ds[info.var_names['HFacW']]
    HFacS    = ds[info.var_names['HFacS']]
    drF      = ds[info.var_names['drF']]
    rA       = ds[info.var_names['rA']]
    dt      = ds[info.var_names['dt']]
    
    # Parameters
    rho0 = info.parameters['rho0']
    
    # Compute useful grid-factor variables
    HFacC_Zl  = info.grid.interp(HFacC,'Z', boundary='fill', fill_value=0, to='right')
    dzMat   = drF * HFacC
    CellVol = rA  * dzMat
    
    # Total tendency
    z_star_scale = (1+Eta/Depth)
    tendS = info.grid.diff((S*z_star_scale).where(HFacC!=0),'time')/dt
    
    # Horizontal convergence
    adv_hConvS = -(info.grid.diff(ADVx_SLT.where(HFacW!=0),'X') + 
                   info.grid.diff(ADVy_SLT.where(HFacS!=0),'Y'))/CellVol
    
    # Vertical convergence
    for i in range(3):
        if   i==0: var_in = ADVr_SLT
        elif i==1: var_in = DFrI_SLT
        elif i==2: var_in = KPPg_SLT

        var_out = var_in.where(HFacC_Zl!=0).diff('Zl')
        var_out = var_out.drop('Zl').rename({'Zl':'Z'}).assign_coords(Z=ds['Z'].isel(Z=slice(None,-1)))
        tmp     = _xr.DataArray(_np.zeros(var_out.isel(Z=-1).shape),
                                  coords=var_out.isel(Z=-1).coords,
                                  dims=var_out.isel(Z=-1).dims)
        tmp = tmp.assign_coords(Z=ds['Z'].isel(Z=-1))
        var_out = _xr.concat([var_out, tmp], 'Z')
        var_out = var_out.where(var_out['Z']!=var_out['Z'].isel(Z=-1))
        var_out = var_out/CellVol

        if   i==0: adv_vConvS = var_out
        elif i==1: dif_vConvS = var_out 
        elif i==2: kpp_vConvS = var_out 
    
    # Surface flux
    forcS = oceSPtnd
    if ds['Zp1'].isel(Zp1=0)==0:
        forcS_surf = (SFLUX + forcS.isel(Z=0)).expand_dims('Z')
        forcS_bott = forcS.isel(Z=slice(1,None))
        forcS = _xr.concat([forcS_surf, forcS_bott],dim='Z')
    forcS = forcS /(dzMat*rho0)
    
    
    # Create DataArrays
    tendS.attrs['units']     = 'psu/s'
    tendS.attrs['long_name'] = 'Salt total tendency'
    tendS.attrs['history']   = 'Computed offline by OceanSpy'
    
    adv_hConvS.attrs['units']     = 'psu/s'
    adv_hConvS.attrs['long_name'] = 'Salt horizontal advective convergence'
    adv_hConvS.attrs['history']   = 'Computed offline by OceanSpy'
    
    adv_vConvS.attrs['units']     = 'psu/s'
    adv_vConvS.attrs['long_name'] = 'Salt vertical advective convergence'
    adv_vConvS.attrs['history']   = 'Computed offline by OceanSpy'
    
    dif_vConvS.attrs['units']     = 'psu/s'
    dif_vConvS.attrs['long_name'] = 'Salt vertical diffusive convergence'
    dif_vConvS.attrs['history']   = 'Computed offline by OceanSpy'
    
    kpp_vConvS.attrs['units']     = 'psu/s'
    kpp_vConvS.attrs['long_name'] = 'Salt vertical kpp convergence'
    kpp_vConvS.attrs['history']   = 'Computed offline by OceanSpy'
    
    forcS.attrs['units']     = 'psu/s'
    forcS.attrs['long_name'] = 'Salt surface forcing'
    forcS.attrs['history']   = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['tendS']      = tendS
    ds['adv_hConvS'] = adv_hConvS
    ds['adv_vConvS'] = adv_vConvS
    ds['dif_vConvS'] = dif_vConvS
    ds['kpp_vConvS'] = kpp_vConvS
    ds['forcS']      = forcS
    
    # Update var_names
    info.var_names['tendS']      = 'tendS'
    info.var_names['adv_hConvS'] = 'adv_hConvS'
    info.var_names['adv_vConvS'] = 'adv_vConvS'
    info.var_names['dif_vConvS'] = 'dif_vConvS'
    info.var_names['kpp_vConvS'] = 'kpp_vConvS'
    info.var_names['forcS']      = 'forcS'
    
    return _utils.reorder_ds(ds), info


def transport(ds, info,
              deep_copy = False):
    """
    Compute volume flux through a mooring array section (in/outflow), and add to dataset.
    If the array is closed, transport in the first cell is not computed.
    Otherwise, transport in both the first and last cells is not computed.
    Transport can be computed following two paths (ext and int side), so 'path'=[0,1] dimension is added.
    'zonal_dir_tran' and 'merid_dir_tran' indicate the direction of positive transport along the path.
    For example, merid_dir_tran=1 and zonal_dir_tran=-1 means positive transport towards NW.
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    deep_copy: bool
        If True, deep copy ds and infod
    
    Returns
    -------
    ds: xarray.Dataset 
    info: oceanspy.open_dataset._info
    """    
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
    
    # Add missing variables
    varList  = ['Xc', 'Yc', 'dyG', 'dxG', 'drF']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing transport')
    
    # Variables
    X    = ds[info.var_names['Xc']]
    Y    = ds[info.var_names['Yc']]
    dxG  = ds[info.var_names['dxG']]
    dyG  = ds[info.var_names['dyG']]
    drF  = ds[info.var_names['drF']]
    cell = ds[info.var_names['cell']]
    
    # Check if mass weighted variables are available, otherwise compute
    try:
        varList = ['UVELMASS', 'VVELMASS']
        ds, info = _utils.compute_missing_variables(ds, info, varList)
        Umass = ds[info.var_names['UVELMASS']]; 
        Vmass = ds[info.var_names['VVELMASS']];
    except:
        varList = ['U', 'V', 'HFacW', 'HFacS']
        ds, info = _utils.compute_missing_variables(ds, info, varList)
        U     = ds[info.var_names['U']]
        V     = ds[info.var_names['V']]
        HFacW = ds[info.var_names['HFacW']]
        HFacS = ds[info.var_names['HFacS']]
        Umass = U*HFacW; 
        Vmass = V*HFacS;
    
    # Extract bfill-ffill values: m^2/s
    U = Umass * ds['dyG'] 
    V = Vmass * ds['dxG'] 
    Ub=U.sel(Xu='Xb'); Uf=U.sel(Xu='Xf')
    Vb=V.sel(Yv='Yb'); Vf=V.sel(Yv='Yf')
    
    # Initialize direction
    Uf_dir = _np.zeros((len(X),2)); Ub_dir = _np.zeros((len(X),2))
    Vf_dir = _np.zeros((len(Y),2)); Vb_dir = _np.zeros((len(Y),2))
    
    # Steps
    diffX = _np.diff(X); diffY = _np.diff(Y)
    
    # Closed array?
    if X[0]==X[-1] and Y[0]==Y[-1]:
        closed = True
        diffX  = _np.append(diffX,diffX[0])
        diffY  = _np.append(diffY,diffY[0])
    else: closed = False
        
    # Loop
    Usign = 1; Vsign = 1
    keepXf = False; keepYf = False
    for i in range(len(diffX)-1):
        if diffY[i]==0 and diffY[i+1]==0:   # Zonal
            Vb_dir[i+1,:]=Vsign; Vf_dir[i+1,:]=Vsign
        elif diffX[i]==0 and diffX[i+1]==0: # Meridional
            Ub_dir[i+1,:]=Usign; Uf_dir[i+1,:]=Usign
            
        # Corners
        elif (diffY[i]<0  and diffX[i+1]>0):  # |_
            Vsign=Usign; keepYf=keepXf
            Uf_dir[i+1,:]=Usign; Vf_dir[i+1,:]=Vsign  
        elif (diffY[i+1]>0  and diffX[i]<0):  
            Usign=Vsign; keepXf=keepYf
            Uf_dir[i+1,:]=Usign; Vf_dir[i+1,:]=Vsign 

        elif (diffY[i]>0  and diffX[i+1]>0): # |‾
            Vsign=-Usign; keepYf=not keepXf
            Uf_dir[i+1,:]=Usign; Vb_dir[i+1,:]=Vsign
        elif (diffY[i+1]<0  and diffX[i]<0):
            Usign=-Vsign; keepXf=not keepYf
            Uf_dir[i+1,:]=Usign; Vb_dir[i+1,:]=Vsign    

        elif (diffX[i]>0  and diffY[i+1]<0): # ‾|  
            Usign=Vsign; keepXf=keepYf
            Vb_dir[i+1,:]=Vsign; Ub_dir[i+1,:]=Usign
        elif (diffX[i+1]<0  and diffY[i]>0):  
            Vsign=Usign; keepYf=keepXf
            Vb_dir[i+1,:]=Vsign; Ub_dir[i+1,:]=Usign

        elif (diffX[i]>0  and diffY[i+1]>0): # _| 
            Usign=-Vsign; keepXf=not keepYf
            Vf_dir[i+1,:]= Vsign; Ub_dir[i+1,:]=Usign
        elif (diffX[i+1]<0  and diffY[i]<0):  
            Vsign=-Usign; keepYf=not keepXf
            Vf_dir[i+1,:]= Vsign; Ub_dir[i+1,:]=Usign 
    
        if keepXf: Ub_dir[i+1,0]=0; Uf_dir[i+1,1]=0
        else:      Uf_dir[i+1,0]=0; Ub_dir[i+1,1]=0
        if keepYf: Vb_dir[i+1,0]=0; Vf_dir[i+1,1]=0
        else:      Vf_dir[i+1,0]=0; Vb_dir[i+1,1]=0
        
    # Create direction DataArrays. 
    # Add a switch to return this? Useful to debug and/or plot velocities.
    Ub_dir  = _xr.DataArray(Ub_dir,  coords={'cell': cell, 'path':[0,1]}, dims=('cell', 'path'))
    Uf_dir  = _xr.DataArray(Uf_dir,  coords={'cell': cell, 'path':[0,1]}, dims=('cell', 'path'))
    Vb_dir  = _xr.DataArray(Vb_dir,  coords={'cell': cell, 'path':[0,1]}, dims=('cell', 'path'))
    Vf_dir  = _xr.DataArray(Vf_dir,  coords={'cell': cell, 'path':[0,1]}, dims=('cell', 'path'))
    
    # Compute transport
    transport = (Ub*Ub_dir+Uf*Uf_dir+Vb*Vb_dir+Vf*Vf_dir)*drF*1.E-6
    if closed: transport = transport.where(ds['cell']!=ds['cell'].isel(cell=0))
    else:      transport = transport.where(ds['cell']==ds['cell'].isel(cell=slice(1,-1)))
    transport.attrs.update({'long_name': 'Volume flux', 
                            'units': 'Sv',
                            'history': 'Computed offline by OceanSpy'}) 
    
    # Return directions
    zonal_dir_tran = Ub_dir+Uf_dir
    zonal_dir_tran.attrs.update({'long_name': 'Zonal direction of the transport', 
                                 'units': '0: No zonal contribution; 1:Eastward; -1:Westward',
                                 'history': 'Computed offline by OceanSpy'}) 
    merid_dir_tran = Vb_dir+Vf_dir
    merid_dir_tran.attrs.update({'long_name': 'Meridional direction of the transport', 
                                 'units': '0: No meridional contribution; 1:Northward; -1:Southward',
                                 'history': 'Computed offline by OceanSpy'}) 
    
    # Add to dataset
    ds['transport'] = transport
    ds['zonal_dir_tran'] = zonal_dir_tran
    ds['merid_dir_tran'] = merid_dir_tran
    
    # Update var_names
    info.var_names['transport']      = 'transport'
    info.var_names['zonal_dir_tran'] = 'zonal_dir_tran'
    info.var_names['merid_dir_tran'] = 'merid_dir_tran'
    
    return _utils.reorder_ds(ds), info

    

    
    
