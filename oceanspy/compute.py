"""
Compute: add new variables to the dataset
"""

# Comments for developers:

# 1) Use the same structure for each function:
#    - Input and output ds and info (add new variables to ds and info)
#    - Always add deep_copy option, check missing variables, and print a message
#    - Always add the following attribute da.attrs['history'] = 'Computed offline by OceanSpy'

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
    
    return ds, info

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
    varList = ['Sigma0', 'HFacC', 'drC']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing N2')
    
    # Variables
    Sigma0 = ds[info.var_names['Sigma0']]
    HFacC  = ds[info.var_names['HFacC']]
    drC    = ds[info.var_names['drC']]
    
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
    
    return ds, info

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
    dyC = ds[info.var_names['dyC']]
    drC = ds[info.var_names['drC']]
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
    
    return ds, info
    
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
    dxC = ds[info.var_names['dxC']]
    drC = ds[info.var_names['drC']]
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
    
    return ds, info
    
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
    dxC = ds[info.var_names['dxC']]
    dyC = ds[info.var_names['dyC']]
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
    
    return ds, info

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
    dxC = ds[info.var_names['dxC']]
    dyC = ds[info.var_names['dyC']]
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
    
    return ds, info


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
    dyG   = ds[info.var_names['dyG']]
    dxG   = ds[info.var_names['dxG']]
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
    
    return ds, info

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
    dyG   = ds[info.var_names['dyG']]
    dxG   = ds[info.var_names['dxG']]
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
    
    return ds, info

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
    
    return ds, info


def Ertel_PV(ds, info,
             deep_copy = False):
    """
    Compute Ertel Potential Vorticity and add to dataset.
    Eq. 2.2 in OC3D, Klinger and Haine, 2018.
    
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
    varList = ['fCori', 'dxC', 'dyC', 'Sigma0', 'N2', 'momVort1', 'momVort2', 'momVort3']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing Ertel_PV')
    
    # Variables
    fCori    = ds[info.var_names['fCori']]
    dxC      = ds[info.var_names['dxC']]
    dyC      = ds[info.var_names['dyC']]
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
    
    return ds, info

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
    
    return ds, info

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
    
    return ds, info

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
    
    return ds, info

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
    
    return ds, info


def transport(ds, info,
              deep_copy = False):
    """
    Compute transport through a mooring array section, and add to dataset.
    Transport at the extremities is set to 0: extremities should be on land.
    Transport can be computed using two paths, so the dimension 'path' is added
    
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
    varList = ['U', 'V', 'dyG', 'dxG', 'HFacW', 'HFacS', 'drF']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing transport')
    
    # Define dist, lats, lons 
    try:
        dist = ds[info.var_names['dist_array']]
        X    = ds[info.var_names['X_array']]
        Y    = ds[info.var_names['Y_array']]
    except ValueError: 
        raise ValueError("\n 'ds' must be a mooring array section. Info needed: dist_array, X_array, Y_array")
    
    # Check extremities
    from warnings import warn as _warn
    if 'Depth' not in ds.variables:
        _warn("\nds['Depth'] not available: ospy.compute.transport() can not check if extremities are on land",stacklevel=2)
    elif (ds['Depth'][0]!=0 or ds['Depth'][-1]!=0):
        _warn('\nThe extremities of the array are not on land. ospy.compute.transport() always returns zero transport at the extremities.',stacklevel=2)
    
    
    # Variables (define 2 paths)
    a    = 0; b    = 1
    u_in = 1; v_in = 1
    if X[0] == X[-1]:
        direction = 'Positive: Toward East'
        v_in = 0
    elif Y[0] == Y[-1]:
        direction = 'Positive: Toward North'
        u_in = 0
    elif _np.sign(Y[0]-Y[-1])*_np.sign(X[0]-X[-1])==1:
        a = 1; b = 0
        direction = 'Positive: Toward North-West'
        u_in = -1
    else:
        direction = 'Positive: Toward North-East'
        u_in = 1

    drF  = ds['drF']
    
    Ua   = ds['U'].isel(Xpath=a)
    Ub   = ds['U'].isel(Xpath=b)
    Va   = ds['V'].isel(Ypath=a)
    Vb   = ds['V'].isel(Ypath=b)

    dyGa   = ds['dyG'].isel(Xpath=a)
    dyGb   = ds['dyG'].isel(Xpath=b)
    HFacWa = ds['HFacW'].isel(Xpath=a)
    HFacWb = ds['HFacW'].isel(Xpath=b)

    dxGa   = ds['dxG'].isel(Ypath=0)
    dxGb   = ds['dxG'].isel(Ypath=1)
    HFacSa = ds['HFacS'].isel(Ypath=0)
    HFacSb = ds['HFacS'].isel(Ypath=1)

    mask_Va  = _np.zeros(dist.shape)
    mask_Vb  = _np.zeros(dist.shape)
    mask_Ua  = _np.zeros(dist.shape)
    mask_Ub  = _np.zeros(dist.shape)
    U_Y  = Y 
    U_Xa = ds['Xp1_array'].isel(Xpath=0)
    U_Xb = ds['Xp1_array'].isel(Xpath=1)
    V_X  = X
    V_Ya = ds['Yp1_array'].isel(Ypath=0)
    V_Yb = ds['Yp1_array'].isel(Ypath=1)

    if X[0] == X[-1]:
        direction = 'Positive: Toward East'
        v_in = 0
    elif Y[0] == Y[-1]:
        direction = 'Positive: Toward North'
        u_in = 0
    elif _np.sign(Y[0]-Y[-1])*_np.sign(X[0]-X[-1])==1:
        Ua   = ds['U'].isel(Xpath=1)
        Ub   = ds['U'].isel(Xpath=0)
        U_Xa = ds['Xp1_array'].isel(Xpath=1)
        U_Xb = ds['Xp1_array'].isel(Xpath=0)

        dyGa   = ds['dyG'].isel(Xpath=1)
        dyGb   = ds['dyG'].isel(Xpath=0)
        HFacWa = ds['HFacW'].isel(Xpath=1)
        HFacWb = ds['HFacW'].isel(Xpath=0)

        dir_inflow = 'Positive: Toward North-West'
        u_in = -1
    else:
        dir_inflow = 'Positive: Toward North-East'
        u_in = 1
            
    # Mask velocities
    for i in range(1,len(dist)-1):
        if X[i-1]==X[i]==X[i+1]:
            mask_Ua[i] = u_in
            mask_Ub[i] = u_in
        elif Y[i-1]==Y[i]==Y[i+1]:
            mask_Va[i] = v_in
            mask_Vb[i] = v_in
        elif (X[i-1]<X[i] and Y[i]<Y[i+1]) or (Y[i-1]>Y[i] and X[i]>X[i+1]):
            mask_Ua[i] = u_in
            mask_Va[i] = v_in
        elif (X[i-1]<X[i] and Y[i]>Y[i+1]) or (Y[i-1]<Y[i] and X[i]>X[i+1]):
            mask_Ub[i] = u_in
            mask_Vb[i] = v_in
        elif (Y[i-1]<Y[i] and X[i]<X[i+1]) or (X[i-1]>X[i] and Y[i]>Y[i+1]):
            mask_Ub[i] = u_in
            mask_Vb[i] = v_in    
        elif (Y[i-1]>Y[i] and X[i]<X[i+1]) or (X[i-1]>X[i] and Y[i]<Y[i+1]):
            mask_Ua[i] = u_in
            mask_Va[i] = v_in
    
    # Compute transport
    transport_a = (Ua * mask_Ua * dyGa * HFacWa +
                   Va * mask_Va * dxGa * HFacSa ).expand_dims('path')
    transport_b = (Ub * mask_Ub * dyGb * HFacWb +
                   Vb * mask_Vb * dxGb * HFacSb ).expand_dims('path')
    transport   = _xr.concat([transport_a, transport_b], dim='path') * _xr.ufuncs.fabs(drF) * 1.e-6
    
    
    # Create DataArray
    transport.attrs['long_name'] = 'Transport (volume flux)'
    transport.attrs['direction'] = direction
    transport.attrs['units']     = 'Sv (10^6 m^3/s)'
    transport.attrs['history']   = 'Computed offline by OceanSpy'
    
    # Add to dataset
    ds['transport'] = transport
    
    # Update var_names
    info.var_names['transport'] = 'transport'
    
    return ds, info
            
    

    
    