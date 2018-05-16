import numpy as np
import pandas as pd
import xarray as xr
import xgcm as xgcm
import time

class Subsample:
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
        The new dataset is stored in self.ds
        

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
        if ds is None: ds = generate_ds()    
            
        # Store input
        self._INds         = ds
        self._INlonRange   = lonRange
        self._INlatRange   = latRange
        self._INdepthRange = depthRange
        self._INtimeRange  = timeRange
        self._INtimeFreq   = timeFreq
        self._INsampMethod = sampMethod

        # Cut array (Space)
        ds = ds.sel(time = slice(min(timeRange),  max(timeRange)),
                    Xp1  = slice(min(lonRange),   max(lonRange)),
                    Yp1  = slice(min(latRange),   max(latRange)),
                    Zp1  = slice(max(depthRange), min(depthRange)))
        ds = ds.sel(X    = slice(min(ds['Xp1'].values), max(ds['Xp1'].values)),
                    Y    = slice(min(ds['Yp1'].values), max(ds['Yp1'].values)),
                    Z    = slice(max(ds['Zp1'].values), min(ds['Zp1'].values)),
                    Zu   = slice(max(ds['Zp1'].values), min(ds['Zp1'].values)))
        ds = ds.sel(Zl   = slice(max(ds['Zp1'].values), min(ds['Z'].values)))
        
        # Resample (Time)
        from pandas.tseries.frequencies import to_offset
        start_time = time.time()
        if to_offset(timeFreq) < to_offset(pd.infer_freq(ds.time.values)):
            raise RuntimeError("Upsampling not supported. Maximum frequency available: "
                               +pd.infer_freq(ds.time.values))
        elif sampMethod=='snapshot' and timeFreq!=pd.infer_freq(ds.time.values):
            # Is there a way more efficient for this?
            print('Resampling in time triggers some computation:',end=' ')
            ds = ds.resample(time=timeFreq).first(skipna=False)
            elapsed_time = time.time() - start_time
            print(time.strftime('done in %H:%M:%S', time.gmtime(elapsed_time)))
        elif sampMethod=='mean':
            print('Resampling in time triggers some computation:',end=' ')
            ds = ds.resample(time=timeFreq).mean()
            elapsed_time = time.time() - start_time
            print(time.strftime('done in %H:%M:%S', time.gmtime(elapsed_time)))
            
        # Store cutted array
        self.ds   = ds
        self.grid = xgcm.Grid(ds, periodic=False)
    
    def compute_Sigma0(self):
        """
        Compute potential density anomaly
        
        Adapted from jmd95.py:
        Density of Sea Water using Jackett and McDougall 1995 (JAOT 12) polynomial
        created by mlosch on 2002-08-09
        converted to python by jahn on 2010-04-29
        """
        
        # Compute potential density only if not available
        if 'Sigma0' in self.ds.variables: 
            print('Sigma0 has been already added to ds')
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
        t = self.ds['Temp']
        s = self.ds['S']  
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
        
        # Store results in ds
        Sigma0   = Sigma0.to_dataset(name='Sigma0')
        Sigma0['Sigma0'].attrs['units']     = 'kg/m^3'
        Sigma0['Sigma0'].attrs['long_name'] = 'potential density anomaly'
        Sigma0['Sigma0'].attrs['history']   = 'Computed offline by OceanSpy'
        self.ds = xr.merge([self.ds,Sigma0])
        
        print('Sigma0 added to ds')
        
    def compute_N2(self):
        """
        Compute potential density anomaly
        N2 = -(g/rho0)(drho/dz)
        """
        
        # Compute N2 only if not available
        if 'N2' in self.ds.variables:
            print('N2 has been already added to ds')
            return

        # Compute potential density
        self.compute_Sigma0()
        
        # Parameters
        g    = 9.81 # m/s^2
        rho0 = 1027 # kg/m^3 
        
        # Compute Brunt-Vaisala   
        N2 =  ( - g / rho0   
                * self.grid.diff(self.ds['Sigma0'], 'Z', to='outer',
                                                         boundary='fill', 
                                                         fill_value=float('nan'))
                * self.grid.interp(self.ds['HFacC'], 'Z', to='outer',
                                                          boundary='fill', 
                                                          fill_value=float('nan'))
                / (self.ds.drC)
              )

        # Store results in ds
        N2   = N2.to_dataset(name='N2')
        N2['N2'].attrs['units']     = 's^-2'
        N2['N2'].attrs['long_name'] = 'Brunt-Vaisala Frequency'
        N2['N2'].attrs['history']   = 'Computed offline by OceanSpy'
        self.ds = xr.merge([self.ds,N2])

        print('N2 added to ds')
        
    def compute_vorticity(self):
        """
        Compute horizontal and vertical components of vorticity
        
        momVort1 = dw/dy-dv/dz
        momVort2 = du/dz-dw/dx
        momVort3 = dv/dx-du/dy
        """
        
        # Compute vorticity only if not available
        if all(x in self.ds.variables for x in ['momVort1', 'momVort2', 'momVort3']):
            print('momVort1, momVort2, and momVort3 have been already added to ds')
            return
        
        # Adapt drC to Zl
        drC = self.ds['drC'][:-1]
        drC = drC.rename({'Zp1': 'Zl'})
        
        # ============================
        # dw/dy-dv/dz
        # ============================
        momVort1 = (self.grid.diff(self.ds['W'] * drC, 'Y', boundary='fill', 
                                                            fill_value=float('nan')) -
                    self.grid.diff(self.ds['V'] * self.ds['dyC'], 'Z', to='right', 
                                                                       boundary='fill', 
                                                                       fill_value=float('nan')) 
                   )/ (self.ds['dyC'] * drC)
        # Store results in ds
        momVort1   = momVort1.to_dataset(name='momVort1')
        momVort1['momVort1'].attrs['units']     = 's^-1'
        momVort1['momVort1'].attrs['long_name'] = '1st component (horizontal) of Vorticity'
        momVort1['momVort1'].attrs['history']   = 'Computed offline by OceanSpy'
        self.ds = xr.merge([self.ds,momVort1])
        print('momVort1 added to ds')
        # ============================
        
        # ============================
        # du/dz-dw/dx
        # ============================
        momVort2 = (self.grid.diff(self.ds['U'] * self.ds['dxC'], 'Z', to='right',
                                                                       boundary='fill', 
                                                                       fill_value=float('nan')) -
                    self.grid.diff(self.ds['W'] * drC, 'X', boundary='fill', 
                                                            fill_value=float('nan')) 
                   )/ (self.ds['dxC'] * drC)
        # Store results in ds
        momVort2   = momVort2.to_dataset(name='momVort2')
        momVort2['momVort2'].attrs['units']     = 's^-1'
        momVort2['momVort2'].attrs['long_name'] = '2nd component (horizontal) of Vorticity'
        momVort2['momVort2'].attrs['history']   = 'Computed offline by OceanSpy'
        self.ds = xr.merge([self.ds,momVort2])
        print('momVort2 added to ds')
        # ============================
        
        # ============================
        # dv/dx-du/dy
        # ============================
        if 'momVort3' in self.ds.variables:
            print('momVort3 has been computed online')
        else:
            momVort3 = (self.grid.diff(self.ds['V'] * self.ds['dyC'], 'X', boundary='fill', 
                                                                           fill_value=float('nan')) -
                        self.grid.diff(self.ds['U'] * self.ds['dxC'], 'Y', boundary='fill', 
                                                                           fill_value=float('nan')) 
                        )/ self.ds['rAz']
            # Store results in ds
            momVort3   = momVort3.to_dataset(name='momVort3')
            momVort3['momVort3'].attrs['units']     = 's^-1'
            momVort3['momVort3'].attrs['long_name'] = '3rd component (vertical) of Vorticity'
            momVort3['momVort3'].attrs['history']   = 'Computed offline by OceanSpy'
            self.ds = xr.merge([self.ds,momVort3])
            print('momVort3 added to ds')
        # ============================
        
    def compute_PV(self):
        """
        Compute Potential Vorticity and its horizontal and vertical components. 
        Ertel PV ant its terms ate computed using eq. 2.25 in Klinger and Haine 
        """
        
        # Compute vorticity only if not available
        if all(x in self.ds.variables for x in ['PV', 'PV1', 'PV2', 'PV3']):
            print('PV, PV1, PV2, PV3 have been already added to ds')
            return
        
        # Compute Sigma0, N2 and vorticity  
        self.compute_Sigma0()
        self.compute_N2()
        self.compute_vorticity()   
        
        # Parameters
        g    = 9.81 # m/s^2
        rho0 = 1027 # kg/m^3
        e    = 0    # e = 2|Omega|cos(theta) is 0 for beta-plane
        
        # dSigma0/dx
        dS0dx = (  self.grid.diff(self.ds['Sigma0'], 'X', boundary='fill', 
                                                          fill_value=float('nan'))
                  / self.ds.dxC)
        # dSigma0/dy
        dS0dy = (  self.grid.diff(self.ds['Sigma0'], 'Y', boundary='fill', 
                                                          fill_value=float('nan'))
                  / self.ds.dyC)
        
        # Move everything on X, Y, Z
        momVort1 = self.grid.interp(self.ds.momVort1,'Y', boundary='fill', 
                                                          fill_value=float('nan'))
        momVort1 = self.grid.interp(momVort1,'Z', to='center', 
                                                  boundary='fill', 
                                                  fill_value=float('nan'))
        
        momVort2 = self.grid.interp(self.ds.momVort2,'X', boundary='fill', 
                                                          fill_value=float('nan'))
        momVort2 = self.grid.interp(momVort2,'Z', to='center', boundary='fill', 
                                                               fill_value=float('nan'))
        
        momVort3 = self.grid.interp(self.ds.momVort3,'X', boundary='fill', 
                                                          fill_value=float('nan'))
        momVort3 = self.grid.interp(momVort3,'Y', boundary='fill', 
                                                  fill_value=float('nan'))
        
        N2 = self.grid.interp(self.ds.N2,'Z', to='center', 
                                              boundary='fill', 
                                              fill_value=float('nan'))
        
        dS0dx = self.grid.interp(dS0dx,'X', boundary='fill', 
                                            fill_value=float('nan'))
        
        dS0dy = self.grid.interp(dS0dy,'Y', boundary='fill', 
                                            fill_value=float('nan'))
        
        # Compute first, second and third components of PV, and sum them up
        PV3 = (momVort3 + self.ds.fCori) * N2 / g
        PV1 = momVort1 * dS0dx / rho0
        PV2 = (momVort2 + e ) * dS0dy / rho0  
        PV  = PV3 + PV1 + PV2
        
        # Store results in ds
        PV1   = PV1.to_dataset(name='PV1')
        PV1['PV1'].attrs['units']     = 'm^-1s^-1'
        PV1['PV1'].attrs['long_name'] = '1st component (horizontal) of Potential Vorticity'
        PV1['PV1'].attrs['history']   = 'Computed offline by OceanSpy'
        self.ds = xr.merge([self.ds,PV1])
        print('PV1 added to ds')
        
        PV2   = PV2.to_dataset(name='PV2')
        PV2['PV2'].attrs['units']     = 'm^-1s^-1'
        PV2['PV2'].attrs['PV3long_name'] = '2nd component (horizontal) of Potential Vorticity'
        PV2['PV2'].attrs['history']   = 'Computed offline by OceanSpy'
        self.ds = xr.merge([self.ds,PV2])
        print('PV2 added to ds')
        
        PV3   = PV3.to_dataset(name='PV3')
        PV3['PV3'].attrs['units']     = 'm^-1s^-1'
        PV3['PV3'].attrs['long_name'] = '3rd component (vertical) of Potential Vorticity'
        PV3['PV3'].attrs['history']   = 'Computed offline by OceanSpy'
        self.ds = xr.merge([self.ds,PV3])
        print('PV3 added to ds')
        
        PV   = PV.to_dataset(name='PV')
        PV['PV'].attrs['units']     = 'm^-1s^-1'
        PV['PV'].attrs['long_name'] = 'Potential Vorticity = PV3 + (PV1 + PV2)'
        PV['PV'].attrs['history']   = 'Computed offline by OceanSpy'
        self.ds = xr.merge([self.ds,PV])
        print('PV added to ds')
        
        