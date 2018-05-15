import numpy as np
import pandas as pd
import xarray as xr
import xgcm as xgcm
import time

class Sample:
    """
    Write something here
    """
    
    def __init__(self,
                 ds         = None, 
                 lonRange   = [-180, 180],
                 latRange   = [-90, 90],
                 depthRange = [0, float("-inf")],
                 timeRange  = ['2007-09-01T00', '2008-09-01T00'],
                 timeFreq   = '6H',
                 sampMethod = 'snapshot'):
        """
        Subsample a dataset in time and space.
        The new dataset is stored in self.CUTOFF.
        

        Parameters
        ----------
        ds: xarray.Dataset or None
           Dataset with all the available variables.
           If None, autogenerate xarray.
        lonRange: list
                 Longitude limits (based on Xp1 dimension)
        latRange: list
                 Latitude limits  (based on Yp1 dimension)
        depthRange: list
                   Depth limits   (based on Zp1 dimension)
        timeRange: list
                  Time limits
        timeFreq: str
                 Time frequency. Available optionts are pandas Offset Aliases (e.g., '6H'):
                 http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        sampMethod: str
                   Sampling method: 'snapshot' or 'mean'
        """

        # Check parameters
        if not isinstance(ds, xr.Dataset) and ds!=None: 
            raise RuntimeError("'ds' needs to be a xarray.Dataset or None")
        if not isinstance(lonRange, list): raise RuntimeError("'lonRange' needs to be a list")
        if not isinstance(latRange, list): raise RuntimeError("'latRange' needs to be a list")
        if not isinstance(depthRange, list): raise RuntimeError("'depthRange' needs to be a list")
        if not isinstance(timeRange, list): raise RuntimeError("'timeRange' needs to be a list")
        if not isinstance(timeFreq, str): raise RuntimeError("'timeFreq' needs to be a string")
        if not isinstance(sampMethod, str): raise RuntimeError("'sampMethod' needs to be a string")

        # Load dataset if not provided
        if ds is None: ds, grid = generate_ds_grid()    
            
        # Store input
        self._ds = ds
        self._lonRange   = lonRange
        self._latRange   = latRange
        self._depthRange = depthRange
        self._timeRange  = timeRange
        self._timeFreq   = timeFreq
        self._sampMethod = sampMethod

        # Cut array (Space)
        ds = ds.sel(time = slice(min(timeRange),  max(timeRange)),
                    Xp1  = slice(min(lonRange),   max(lonRange)),
                    Yp1  = slice(min(latRange),   max(latRange)),
                    Zp1  = slice(max(depthRange), min(depthRange)))
        ds = ds.sel(X    = slice(min(ds['Xp1'].values), max(ds['Xp1'].values)),
                    Y    = slice(min(ds['Yp1'].values), max(ds['Yp1'].values)),
                    Z    = slice(max(ds['Zp1'].values), min(ds['Zp1'].values)),
                    Zu   = slice(max(ds['Zp1'].values), min(ds['Zp1'].values)),
                    Zl   = slice(max(ds['Zp1'].values), min(ds['Zp1'].values)))
        
        # Resample (Time)
        from pandas.tseries.frequencies import to_offset
        if to_offset(timeFreq) < to_offset(pd.infer_freq(ds.time.values)):
            raise RuntimeError("Upsampling not supported. Maximum frequency available: "
                               +pd.infer_freq(ds.time.values))
        elif sampMethod=='snapshot' and timeFreq!=pd.infer_freq(ds.time.values):
            ds = ds.resample(time=timeFreq).first(skipna=False)
        elif sampMethod=='mean':
            ds = ds.resample(time=timeFreq).mean()
            
        # Store cutted array
        self.CUTOFF = ds
    
    
    def compute_Sigma0(self):
        """
        Compute potential density anomaly and add it to self.CUTOFF.
        
        Adapted from jmd95.py:
        Density of Sea Water using Jackett and McDougall 1995 (JAOT 12) polynomial
        created by mlosch on 2002-08-09
        converted to python by jahn on 2010-04-29
        """
        
        # Compute potential density only if not available
        if any(d == 'Sigma0' for d in self.CUTOFF.variables):
            print('Sigma0 has been already added to CUTOFF')
            return
        
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
        t = self.CUTOFF['Temp']
        s = self.CUTOFF['S']  
        p = 0.
        # Useful stuff
        t2   = t*t
        t3   = t2*t
        t4   = t3*t
        s3o2 = s*xr.ufuncs.sqrt(s)
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
        rho    = rho / (1. - p/bulkmod)
        Sigma0 = rho - 1000
        
        # Store results in CUTOFF
        Sigma0   = Sigma0.to_dataset(name='Sigma0')
        Sigma0['Sigma0'].attrs['units']     = 'kg/m^3'
        Sigma0['Sigma0'].attrs['long_name'] = 'potential density anomaly'
        self.CUTOFF = xr.merge([self.CUTOFF,Sigma0])
        
        print('Sigma0 added to CUTOFF')
        
    def compute_N2(self):
        """
        Compute potential density anomaly and add it to self.CUTOFF.
        N2 = -(g/rho0)(drho/dz)
        """
        
        # Compute N2 only if not available
        if any(d == 'N2' for d in self.CUTOFF.variables):
            print('N2 has been already added to CUTOFF')
            return

        # Compute potential density if not available
        if all(d != 'Sigma0' for d in self.CUTOFF.variables): self.compute_Sigma0()
            
        # Compute Brunt-Vaisala
        g    = 9.81 # m/s^2
        rho0 = 1027 # kg/m^3 
        N2   = - (g / rho0) * (self.CUTOFF['Sigma0'].diff('Z') / self.CUTOFF['Z'].diff('Z'))
        
        # Store results in CUTOFF
        N2   = N2.to_dataset(name='N2')
        N2   = N2.rename({'Z': 'Zm1'})
        N2['Zm1'].values = (self.CUTOFF['Z'][:-1].values + self.CUTOFF['Z'][1:].values) / 2
        N2['Zm1'].attrs.update({'c_grid_axis_shift': -0.5})
        self.CUTOFF = xr.merge([self.CUTOFF, N2])
        
        # Interpolte from Zm1 to Z
        grid = xgcm.Grid(self.CUTOFF,periodic=False)
        self.CUTOFF['N2'] = grid.interp(self.CUTOFF['N2'], axis='Z', boundary='fill', fill_value=float('nan'))
        self.CUTOFF = self.CUTOFF.drop('Zm1')
        
        print('N2 added to CUTOFF')