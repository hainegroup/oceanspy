import pytest
import xarray as xr
import numpy as np
import pandas as pd
from oceanspy import OceanDataset

class Datasets:
    def __init__(self):
        # Create a small datasets
        self.NX = 9
        self.NY = 10
        self.NZ = 11
        self.NT = 12
        
    def MITgcm_rect_nc(self):
        """
        Similar to exp_ASR and exp_ERAI
        """
        
        # Horizontal Dimensions
        X      = xr.DataArray( np.arange(self.NX),     dims = 'X')
        Xp1    = xr.DataArray( np.arange(self.NX+1),   dims = 'Xp1')
        Y      = xr.DataArray( np.arange(self.NY),     dims = 'Y')
        Yp1    = xr.DataArray( np.arange(self.NY+1),   dims = 'Yp1')
        
        # Vertical Dimensions
        Z      = xr.DataArray(-np.arange(self.NZ)-0.5, dims = 'Z')
        Zp1    = xr.DataArray(-np.arange(self.NZ+1),   dims = 'Zp1')
        Zu     = xr.DataArray(-np.arange(self.NZ)-1,   dims = 'Zu')
        Zl     = xr.DataArray(-np.arange(self.NZ),     dims = 'Zl')

        # Space Coordinates
        YC, XC = xr.broadcast(Y+0.5, X+0.5)
        YG, XG = xr.broadcast(Yp1  , Xp1)
        YU, XU = xr.broadcast(Y+0.5, Xp1)
        YV, XV = xr.broadcast(Yp1  , X+0.5)

        # Time Dimension
        time   = xr.DataArray(pd.date_range('2000-01-01', freq='M', periods=self.NT), dims = 'time')
        
        # Add some nan due to exch2
        rX = np.random.randint(self.NX)
        rY = np.random.randint(self.NY)
        maskC = xr.where(np.logical_or(XC!=XC.isel(X=rX, Y=rY), YC!=YC.isel(X=rX, Y=rY)), 1, 0)
        maskG = xr.where(np.logical_or(XG!=XG.isel(Xp1=rX, Yp1=rY), YG!=YG.isel(Xp1=rX, Yp1=rY)), 1, 0)
        maskU = xr.where(np.logical_or(XU!=XU.isel(Xp1=rX, Y=rY), YU!=YU.isel(Xp1=rX, Y=rY)), 1, 0)
        maskV = xr.where(np.logical_or(XV!=XV.isel(X=rX, Yp1=rY), YV!=YV.isel(X=rX, Yp1=rY)), 1, 0)
        XC = XC.where(maskC); YC = YC.where(maskC)
        XG = XG.where(maskG); YG = YG.where(maskG)
        XU = XU.where(maskU); YU = YU.where(maskU)
        XV = XV.where(maskV); YV = YV.where(maskV)
        
        return xr.Dataset({'X'   : X,    'Xp1': Xp1, 
                           'Y'   : Y,    'Yp1': Yp1,
                           'Z'   : Z,    'Zp1': Zp1, 'Zu': Zu, 'Zl': Zl,
                           'YC'  : YC,   'XC' : XC, 
                           'YG'  : YG,   'XG' : XG, 
                           'YU'  : YU,   'XU' : XU, 
                           'YV'  : YV,   'XV' : XV,
                           'time': time, })
    
    def MITgcm_rect_bin(self):
        """
        Similar to exp_ASR and exp_ERAI
        """
        
        # Horizontal Dimensions
        X      = xr.DataArray( np.arange(self.NX),  dims = 'X')
        Xp1    = xr.DataArray( np.arange(self.NX),  dims = 'Xp1')
        Y      = xr.DataArray( np.arange(self.NY),  dims = 'Y')
        Yp1    = xr.DataArray( np.arange(self.NY),  dims = 'Yp1')
        
        # Vertical Dimensions
        Z      = xr.DataArray(-np.arange(self.NZ)-0.5, dims = 'Z')
        Zp1    = xr.DataArray(-np.arange(self.NZ+1),   dims = 'Zp1')
        Zu     = xr.DataArray(-np.arange(self.NZ)-1,   dims = 'Zu')
        Zl     = xr.DataArray(-np.arange(self.NZ),     dims = 'Zl')

        # Space Coordinates
        YC, XC = xr.broadcast(Y+0.5, X+0.5)
        YG, XG = xr.broadcast(Yp1  , Xp1)
        YU, XU = xr.broadcast(Y+0.5, Xp1)
        YV, XV = xr.broadcast(Yp1  , X+0.5)

        # Time Dimension
        time   = xr.DataArray(pd.date_range('2000-01-01', freq='M', periods=self.NT), dims = 'time')
        
        # Add some nan due to exch2
        rX = np.random.randint(self.NX)
        rY = np.random.randint(self.NY)
        maskC = xr.where(np.logical_or(XC!=XC.isel(X=rX, Y=rY), YC!=YC.isel(X=rX, Y=rY)), 1, 0)
        maskG = xr.where(np.logical_or(XG!=XG.isel(Xp1=rX, Yp1=rY), YG!=YG.isel(Xp1=rX, Yp1=rY)), 1, 0)
        maskU = xr.where(np.logical_or(XU!=XU.isel(Xp1=rX, Y=rY), YU!=YU.isel(Xp1=rX, Y=rY)), 1, 0)
        maskV = xr.where(np.logical_or(XV!=XV.isel(X=rX, Yp1=rY), YV!=YV.isel(X=rX, Yp1=rY)), 1, 0)
        XC = XC.where(maskC); YC = YC.where(maskC)
        XG = XG.where(maskG); YG = YG.where(maskG)
        XU = XU.where(maskU); YU = YU.where(maskU)
        XV = XV.where(maskV); YV = YV.where(maskV)
        
        return xr.Dataset({'X'   : X,    'Xp1': Xp1, 
                           'Y'   : Y,    'Yp1': Yp1,
                           'Z'   : Z,    'Zp1': Zp1, 'Zu': Zu, 'Zl': Zl,
                           'YC'  : YC,   'XC' : XC, 
                           'YG'  : YG,   'XG' : XG, 
                           'YU'  : YU,   'XU' : XU, 
                           'YV'  : YV,   'XV' : XV,
                           'time': time, })
    
    def MITgcm_curv_nc(self):
        """
        Similar to exp_Arctic_Control
        """
        
        # Horizontal Dimensions (use xarray tutorial)
        ds  = xr.tutorial.open_dataset('rasm')
        ds['xc'] = xr.where(ds['xc']>180, ds['xc']-180, ds['xc'])
        X   = ds['x'].isel(x=slice(self.NX)).values
        Xp1 = ds['x'].isel(x=slice(self.NX+1)).values
        Y   = ds['y'].isel(y=slice(self.NY)).values
        Yp1 = ds['y'].isel(y=slice(self.NY+1)).values
        XG  = ds['xc'].isel(x=slice(self.NX+1), y=slice(self.NY+1)).values
        YG  = ds['yc'].isel(x=slice(self.NX+1), y=slice(self.NY+1)).values
        XC  = (XG[:-1, :-1] + XG[1:, 1:])/2
        YC  = (YG[:-1, :-1] + YG[1:, 1:])/2
        
        # Set DataArray
        X      = xr.DataArray( X,   dims = 'X')
        Xp1    = xr.DataArray( Xp1, dims = 'Xp1')
        Y      = xr.DataArray( Y,   dims = 'Y')
        Yp1    = xr.DataArray( Yp1, dims = 'Yp1')
        XC     = xr.DataArray( XC,  dims = ('Y'  , 'X'))
        XG     = xr.DataArray( XG,  dims = ('Yp1', 'Xp1'))
        YC     = xr.DataArray( YC,  dims = ('Y'  , 'X'))
        YG     = xr.DataArray( YG,  dims = ('Yp1', 'Xp1'))
        
        # Vertical Dimensions
        Z      = xr.DataArray(-np.arange(self.NZ)-0.5, dims = 'Z')
        Zp1    = xr.DataArray(-np.arange(self.NZ+1),   dims = 'Zp1')
        Zu     = xr.DataArray(-np.arange(self.NZ)-1,   dims = 'Zu')
        Zl     = xr.DataArray(-np.arange(self.NZ),     dims = 'Zl')
        
        # Time Dimension
        time   = xr.DataArray(pd.date_range('2000-01-01', freq='M', periods=self.NT), dims = 'time')
        
        return xr.Dataset({'X'   : X,    'Xp1': Xp1, 
                           'Y'   : Y,    'Yp1': Yp1,
                           'Z'   : Z,    'Zp1': Zp1, 'Zu': Zu, 'Zl': Zl,
                           'YC'  : YC,   'XC' : XC, 
                           'YG'  : YG,   'XG' : XG, 
                           'time': time, })
    
    
datasets = {'MITgcm_rect_nc' : Datasets().MITgcm_rect_nc(),
            'MITgcm_rect_bin': Datasets().MITgcm_rect_bin(),
            'MITgcm_curv_nc' : Datasets().MITgcm_curv_nc()}

oceandatasets = {'MITgcm_rect_nc' : OceanDataset(datasets['MITgcm_rect_nc']).import_MITgcm_rect_nc(),
                 'MITgcm_rect_bin': OceanDataset(datasets['MITgcm_rect_bin']).import_MITgcm_rect_bin(),
                 'MITgcm_curv_nc' : OceanDataset(datasets['MITgcm_curv_nc']).import_MITgcm_curv_nc()}

aliased_ods = {}
for od_name in oceandatasets:
    dataset = oceandatasets[od_name].dataset
    aliases = {var: 'alias_'+var for var in dataset.variables}
    dataset = dataset.rename(aliases)
    aliased_ods[od_name] = OceanDataset(dataset).set_aliases(aliases)
