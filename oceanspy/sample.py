import numpy as np
import pandas as pd
import xarray as xr
import xgcm as xgcm
import time

class Sample:
    """
    An object that extract a subsample in time and space, and add new variables.
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
        Create a new Sample from an input dataset.

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
        self.sample = ds
    
    