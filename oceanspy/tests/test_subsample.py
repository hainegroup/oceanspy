import pytest
import pandas as pd
import numpy as np
import xarray as xr
from . datasets import (aliased_ods, oceandatasets)






@pytest.mark.parametrize("varList", [None,
                                     lambda od_in: np.random.choice(od_in.dataset.data_vars, 1),
                                     lambda od_in: np.random.choice(od_in.dataset.dims, 1),
                                     lambda od_in: np.random.choice(od_in.dataset.coords, 1)])
def test_cutout_varList(varList):
    # Add some variable
    od_in = oceandatasets['MITgcm_rect_nc']
    ds    = xr.Dataset({'zeros': xr.zeros_like(od_in.dataset['XC']),
                        'ones':  xr.ones_like(od_in.dataset['XC'])})
    od_in = od_in.merge_into_oceandataset(ds)
    
    # Create varList and run cutout
    if varList is not None  : varList = varList(od_in)
    od_out = od_in.subsample.cutout(varList=varList)
    
    # Variables
    assert set(od_out.dataset.dims) == set(od_in.dataset.dims)
    assert set(od_out.dataset.coords) == set(od_in.dataset.coords)
    if varList is not None and set(varList).issubset(od_in.dataset.data_vars):
        assert list(od_out.dataset.data_vars) != list(od_in.dataset.data_vars)
    
    
    
    
    
@pytest.mark.parametrize("timeRange", [None,
                                       lambda od_in: np.random.choice(od_in.dataset['time_midp'].values.ravel(), 1),
                                       lambda od_in: np.random.choice(od_in.dataset['time_midp'].values.ravel(), 3, replace=False)])
@pytest.mark.parametrize("timeFreq", [None, '2M'])
@pytest.mark.parametrize("sampMethod", ['snapshot', 'mean'])
def test_cutout_time(timeRange, timeFreq, sampMethod):
    # Extract od
    od_in = oceandatasets['MITgcm_rect_nc']
    
    # Create timeRange
    if timeRange is not None: timeRange = timeRange(od_in)
    
    # Run cutout
    if timeRange is not None and len(timeRange)==1 and timeFreq is not None:
            pytest.skip('Pandas needs at least 3 points to infer frequency')
    elif timeFreq is not None:
        with pytest.warns(UserWarning):
            od_out = od_in.subsample.cutout(timeRange=timeRange, timeFreq=timeFreq, sampMethod=sampMethod)
    else:
        od_out = od_in.subsample.cutout(timeRange=timeRange, timeFreq=timeFreq, sampMethod=sampMethod)
    
    # Check dimensions
    for dim in ['time', 'time_midp']:
        if timeFreq is not None:
            assert not np.array_equal(np.unique(od_in.dataset[dim].diff(dim).values), 
                                      np.unique(od_out.dataset[dim].diff(dim).values))
        elif timeRange is None:
            assert len(od_in.dataset[dim])==len(od_out.dataset[dim])
        else:
            assert len(od_in.dataset[dim])!=len(od_out.dataset[dim])
            
    # Relative size dimensions unchanged
    assert len(od_in.dataset['time']) - len(od_in.dataset['time_midp']) == len(od_out.dataset['time']) - len(od_out.dataset['time_midp'])


    
    
@pytest.mark.parametrize("ZRange", [None,
                                    lambda od_in: np.random.choice(od_in.dataset['Z'].values.ravel(), 2, replace=False)])
def test_cutout_Z(ZRange):
    # Extract od
    od_in = oceandatasets['MITgcm_rect_nc']
    
    # Create ZRange
    if ZRange is not None: ZRange = ZRange(od_in)
    
    # Run cutout
    od_out = od_in.subsample.cutout(ZRange=ZRange)
    
    for dim in [dim for dim in od_in.dataset.dims if dim[0]=='Z' ]:
        if ZRange is None:
            assert len(od_in.dataset[dim])==len(od_out.dataset[dim])
        else:
            assert len(od_in.dataset[dim])!=len(od_out.dataset[dim])
            
    # Relative size dimensions unchanged
    assert len(od_in.dataset['Zp1']) - len(od_in.dataset['Z']) == len(od_out.dataset['Zp1']) - len(od_out.dataset['Z'])    
    
    
    
    
@pytest.mark.parametrize("od_name", list(oceandatasets.keys()))
@pytest.mark.parametrize("XRange", [None,
                                    lambda od_in: [od_in.dataset['XC'].mean().values],
                                    lambda od_in: [od_in.dataset['XC'].mean().values-od_in.dataset['XC'].std().values,
                                                   od_in.dataset['XC'].mean().values+od_in.dataset['XC'].std().values]])
@pytest.mark.parametrize("YRange", [None,
                                    lambda od_in: [od_in.dataset['YC'].mean().values],
                                    lambda od_in: [od_in.dataset['YC'].mean().values-od_in.dataset['YC'].std().values,
                                                   od_in.dataset['YC'].mean().values+od_in.dataset['YC'].std().values]])
@pytest.mark.parametrize("mask_outside", [True, False])
def test_cutout_horizontal(od_name, XRange, YRange, mask_outside):
    # Extract od
    od_in = oceandatasets[od_name]
    od_in = od_in.merge_into_oceandataset(xr.zeros_like(od_in.dataset['XC']).rename('zeros'))
    
    
    # Extract ranges
    if XRange is not None   : XRange = XRange(od_in)
    if YRange is not None   : YRange = YRange(od_in)
        
    # Curvilinear case
    if 'curv' in od_name:
        if (XRange is not None and len(XRange)==1) or (YRange is not None and len(YRange)==1):
            pytest.skip('Might hit zero grid points in the horizontal range')
    od_out = od_in.subsample.cutout(XRange=XRange, YRange=YRange, mask_outside=mask_outside)
    
    
    if 'rect' in od_name and 'nc' in od_name:
        for dim in ['X', 'Y']:
            if eval(dim+'Range') is None:
                assert len(od_in.dataset[dim])==len(od_out.dataset[dim])
                assert len(od_in.dataset[dim+'p1'])==len(od_out.dataset[dim+'p1'])
            else:
                assert len(od_in.dataset[dim])!=len(od_out.dataset[dim])
                assert len(od_in.dataset[dim+'p1'])!=len(od_out.dataset[dim+'p1'])
                
    elif 'rect' in od_name and 'bin' in od_name:
        if XRange is not None or YRange is not None:
            assert len(od_out.dataset['Xp1'])>len(od_out.dataset['X'])!=len(od_in.dataset['X'])
            assert len(od_out.dataset['Yp1'])>len(od_out.dataset['Y'])!=len(od_in.dataset['Y'])
        else:
            for dim in ['X', 'Y']:
                assert len(od_in.dataset[dim])==len(od_out.dataset[dim])
                assert len(od_in.dataset[dim+'p1'])==len(od_out.dataset[dim+'p1'])
    elif 'curv' in od_name:
        if (XRange is not None or YRange is not None) and mask_outside is True:
            assert np.isnan(od_out.dataset['zeros'].values).any()
                    
            
            
            
        
@pytest.mark.parametrize("XRange", [None, lambda od_in: [od_in.dataset['XC'].mean().values]])        
@pytest.mark.parametrize("dropAxes", [True, False, ['X'], ['Y']])    
def test_cutout_dropAxes(XRange, dropAxes):  
    # Extract od
    od_in = oceandatasets['MITgcm_rect_nc']
    
    # Extract range
    if XRange is not None   : XRange = XRange(od_in)
        
    # Run cutout
    od_out = od_in.subsample.cutout(XRange=XRange, dropAxes=dropAxes)
    
    # Check cut
    if XRange is not None:
        assert len(od_out.dataset['Xp1']) != len(od_in.dataset['Xp1'])
        
    # Drop axis for X
    if (dropAxes is True or (isinstance(dropAxes, list) and 'X' in dropAxes)) and (XRange is not None and len(XRange)==1):
        assert len(od_out.dataset['X'])==len(od_out.dataset['Xp1'])==1
        assert 'X' not in od_out.grid_coords
    else:
        assert len(od_out.dataset['Xp1']) - len(od_out.dataset['X']) == len(od_in.dataset['Xp1']) - len(od_in.dataset['X'])
        
    # Y is unchanged
    assert len(od_out.dataset['Y'])==len(od_in.dataset['Y'])
    assert len(od_out.dataset['Yp1'])==len(od_in.dataset['Yp1'])  
    assert len(od_out.dataset['Yp1']) - len(od_out.dataset['Y']) == len(od_in.dataset['Yp1']) - len(od_in.dataset['Y'])
        
        
        

    
    
    
    
