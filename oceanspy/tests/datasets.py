import pytest
import xarray as xr
import numpy as np

class Datasets:
    def __init__(self):
        # Create a small datasets
        self.NX = 9
        self.NY = 10
        self.NZ = 11
        self.NT = 12
        
    def MITgcm_rect_nc(self):
        
        # Space Dimensions
        X      = xr.DataArray( np.arange(self.NX),     dims = 'X')
        Xp1    = xr.DataArray( np.arange(self.NX+1),   dims = 'Xp1')
        Y      = xr.DataArray( np.arange(self.NY),     dims = 'Y')
        Yp1    = xr.DataArray( np.arange(self.NY+1),   dims = 'Yp1')
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
        time   = xr.DataArray(np.arange(0, self.NT, dtype='datetime64[D]'), dims = 'time')
        
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
    
    
datasets = {'MITgcm_rect_nc': Datasets().MITgcm_rect_nc()}

@pytest.fixture(scope="module", params=datasets.keys())
def all_grids(request):
    ds = datasets[request.param]
    return ds

@pytest.fixture(scope="module", params=[key for key in datasets.keys() if 'MITgcm' in key])
def MITgcm_grids(request):
    ds = datasets[request.param]
    return ds

@pytest.fixture(scope="module", params=[key for key in datasets.keys() if 'rect' in key])
def rect_grids(request):
    ds = datasets[request.param]
    return ds

@pytest.fixture(scope="module", params=[key for key in datasets.keys() if 'nc' in key])
def nc_grids(request):
    ds = datasets[request.param]
    return ds