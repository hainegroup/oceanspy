import pytest
import numpy  as np
import pandas as pd

from oceanspy import OceanDataset
from oceanspy.open_oceandataset import from_netcdf
from .datasets import (datasets, oceandatasets)


def test_import_MITgcm_rect_nc():
    """
    Testing Almansi et al. 2017 (ASR and ERAI)
    """
    
    # Dataset
    ds = datasets['MITgcm_rect_nc']
    
    # From open_oceandataset
    od = OceanDataset(ds).import_MITgcm_rect_nc()
    check_coords = {'Y':    'center', 'Yp1': 'outer', 
                    'X':    'center', 'Xp1': 'outer', 
                    'Z':    'center', 'Zp1': 'outer', 'Zu': 'right', 'Zl': 'left',
                    'time': 'outer', 'time_midp': 'center'}
    
    # All coordinates
    assert set(od.dataset.variables).issubset(od.dataset.coords)
    
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
    
def test_import_MITgcm_rect_bin():
    """
    Testing Magaldi and Haine, 2015
    """
    
    # Dataset
    ds = datasets['MITgcm_rect_bin']
    
    # From open_oceandataset
    od = OceanDataset(ds).import_MITgcm_rect_bin()
    check_coords = {'Y':    'center', 'Yp1': 'right', 
                    'X':    'center', 'Xp1': 'right', 
                    'Z':    'center', 'Zp1': 'outer', 'Zu': 'right', 'Zl': 'left',
                    'time': 'outer', 'time_midp': 'center'} 
    
    # All coordinates
    assert set(od.dataset.variables).issubset(od.dataset.coords)
    
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
    
def test_import_MITgcm_curv():
    """
    Testing exp_Arctic_Control
    """
    
    # Dataset
    ds = datasets['MITgcm_curv_nc']
    
    # From open_oceandataset
    od = OceanDataset(ds).import_MITgcm_curv_nc()
    check_coords = {'Y':    'center', 'Yp1': 'outer', 
                    'X':    'center', 'Xp1': 'outer', 
                    'Z':    'center', 'Zp1': 'outer', 'Zu': 'right', 'Zl': 'left',
                    'time': 'outer', 'time_midp': 'center'}   
    
    # All coordinates
    assert set(od.dataset.variables).issubset(od.dataset.coords)
    
    # Check grid
    for axis in od.grid.axes.keys():
        for pos in od.grid.axes[axis].coords.keys():
            coord = od.grid.axes[axis].coords[pos].name
            assert pos == check_coords[coord]
            
    # Check new dimensions
    assert np.array_equal(np.float32(od.dataset['XU'].values), 
                          np.float32((od.dataset['XG'].values[:-1, :]+od.dataset['XG'].values[1:, :])/2))
    assert np.array_equal(np.float32(od.dataset['YU'].values), 
                          np.float32((od.dataset['YG'].values[:-1, :]+od.dataset['YG'].values[1:, :])/2))
    assert np.array_equal(np.float32(od.dataset['XV'].values), 
                          np.float32((od.dataset['XG'].values[:, :-1]+od.dataset['XG'].values[:, 1:])/2))
    assert np.array_equal(np.float32(od.dataset['YV'].values), 
                          np.float32((od.dataset['YG'].values[:, :-1]+od.dataset['YG'].values[:, 1:])/2))

    # Check time
    od_time      = pd.to_datetime(od.dataset['time'].values).to_julian_date() 
    od_time_midp = pd.to_datetime(od.dataset['time_midp'].values).to_julian_date() 
    my_time      = pd.to_datetime(ds['time'].values).to_julian_date() 
    my_time_midp = (my_time[:-1] + my_time[1:])/2
    assert np.array_equal(od_time, my_time)
    assert np.array_equal(od_time_midp, my_time_midp)
    
def test_from_netcdf():  
    
    # Dataset
    orig_od = oceandatasets['MITgcm_rect_nc']

    # Save
    filename = 'tmp_oceandataset.nc'
    orig_od.to_netcdf(filename)
    
    # Open
    open_od = from_netcdf(filename)
    
    assert orig_od.dataset.equals(open_od.dataset)
    assert orig_od._ds.equals(open_od._ds)
    
    # Clean up
    import os
    os.remove(filename)
    

