import pytest
import numpy  as np
import pandas as pd

from oceanspy import OceanDataset
from . datasets import datasets

def test_import_MITgcm_rect_nc():
    
    # Dataset
    ds = datasets['MITgcm_rect_nc']
    
    # From open_oceandataset
    od = OceanDataset(ds).import_MITgcm_rect_nc()
    check_coords = {'Y':    'center', 'Yp1': 'outer', 
                    'X':    'center', 'Xp1': 'outer', 
                    'Z':    'center', 'Zp1': 'outer', 'Zu': 'right', 'Zl': 'left',
                    'time': 'outer', 'time_midp': 'center'}
    
    # Check NaNs
    assert any(od.dataset.isnull()) == False    
    
    # Check new dimensions
    assert np.array_equal(od.dataset['XC'].isel(Y=0).values,   od.dataset['X'].values)
    assert np.array_equal(od.dataset['XG'].isel(Yp1=0).values, od.dataset['Xp1'].values)
    assert np.array_equal(od.dataset['XU'].isel(Y=0).values,   od.dataset['Xp1'].values)
    assert np.array_equal(od.dataset['XV'].isel(Yp1=0).values, od.dataset['X'].values)
    assert np.array_equal(od.dataset['YC'].isel(X=0).values,   od.dataset['Y'].values)
    assert np.array_equal(od.dataset['YG'].isel(Xp1=0).values, od.dataset['Yp1'].values)
    assert np.array_equal(od.dataset['YU'].isel(Xp1=0).values, od.dataset['Y'].values)
    assert np.array_equal(od.dataset['YV'].isel(X=0).values,   od.dataset['Yp1'].values)
    
    # Check grid
    for axis in od.grid.axes.keys():
        for pos in od.grid.axes[axis].coords.keys():
            coord = od.grid.axes[axis].coords[pos].name
            assert pos == check_coords[coord]
    
    # Check time
    od_time      = pd.to_datetime(od.dataset['time'].values).to_julian_date() 
    od_time_midp = pd.to_datetime(od.dataset['time_midp'].values).to_julian_date() 
    my_time      = pd.to_datetime(ds['time'].values).to_julian_date() 
    my_time_midp = (my_time[:-1] + my_time[1:])/2
    assert np.array_equal(od_time, my_time)
    assert np.array_equal(od_time_midp, my_time_midp)
    
    
    
