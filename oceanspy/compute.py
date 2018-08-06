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
    momVort1 = info.grid.interp(momVort1, 'Y', boundary='fill', fill_value=float('nan'))
    momVort1 = info.grid.interp(momVort1, 'Z', boundary='fill', fill_value=float('nan'))
    
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
    momVort2 = info.grid.interp(momVort2, 'X', boundary='fill', fill_value=float('nan'))
    momVort2 = info.grid.interp(momVort2, 'Z', boundary='fill', fill_value=float('nan'))
    
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
    varList = ['Y', 'fCori', 'dxC', 'dyC', 'Sigma0', 'N2', 'momVort1', 'momVort2', 'momVort3']
    ds, info = _utils.compute_missing_variables(ds, info, varList)
    
    # Message
    print('Computing Ertel_PV')
    
    # Variables
    Y        = ds[info.var_names['Y']]
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


