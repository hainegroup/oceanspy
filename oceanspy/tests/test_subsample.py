import pytest
import pandas as pd
import numpy as np
import xarray as xr
from oceanspy.subsample import *
from oceanspy.utils import great_circle_path
from .datasets import (aliased_ods, oceandatasets)



# =======
# CUTOUT
# =======

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
            od_out = cutout(od_in, timeRange=timeRange, timeFreq=timeFreq, sampMethod=sampMethod)
    else:
        od_out = od_in.subsample.cutout(timeRange=timeRange, timeFreq=timeFreq, sampMethod=sampMethod)
        od_out = cutout(od_in, timeRange=timeRange, timeFreq=timeFreq, sampMethod=sampMethod)
    
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
    od_out = cutout(od_in, ZRange=ZRange)
    
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
    od_out = cutout(od_in, XRange=XRange, YRange=YRange, mask_outside=mask_outside)
    
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
    od_out = cutout(od_in, XRange=XRange, dropAxes=dropAxes)
    
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
        
        
        
        
        
# =======
# MOORING
# =======

@pytest.mark.parametrize("od_name", ['MITgcm_curv_nc'])
def test_mooring_array(od_name):
    
    # Extract od
    od_in = oceandatasets[od_name]
    indY, indX = xr.broadcast(od_in.dataset['Y'], od_in.dataset['X'])
    od_in = od_in.merge_into_oceandataset(xr.Dataset({'indX': indX, 'indY': indY}))
    
    # Get random coords
    X = od_in.dataset['XC'].stack(XY=('X', 'Y')).values
    Y = od_in.dataset['YC'].stack(XY=('X', 'Y')).values
    inds = np.random.choice(len(X), 3)
    Xmoor = X[inds]
    Ymoor = Y[inds]
    
    # Run mooring
    od_out = od_in.subsample.mooring_array(Xmoor=Xmoor, Ymoor=Ymoor)
    od_out = mooring_array(od_in, Xmoor=Xmoor, Ymoor=Ymoor)
    
    # Check dimensions
    for dim in ['mooring', 'mooring_midp']:
        assert dim in od_out.dataset.dims and dim in od_out.grid_coords['mooring']
    assert len(od_out.dataset['X'])==len(od_out.dataset['Y'])==1
    assert len(od_out.dataset['Xp1'])==len(od_out.dataset['Yp1'])==2
    
    # Check verteces
    assert od_out.dataset['XC'].isel(mooring=0).values==Xmoor[0] and od_out.dataset['XC'].isel(mooring=-1).values==Xmoor[-1]
    assert od_out.dataset['YC'].isel(mooring=0).values==Ymoor[0] and od_out.dataset['YC'].isel(mooring=-1).values==Ymoor[-1]
    
    # Check path
    diffX = od_out.grid.diff(od_out.dataset['indX'], 'mooring')
    diffY = od_out.grid.diff(od_out.dataset['indY'], 'mooring')
    assert np.count_nonzero(diffX.values*diffY.values)==0
    assert np.count_nonzero(diffX.values+diffY.values)==len(od_out.dataset['mooring_midp'])
    

# =======
# SURVEY
# =======
    
@pytest.mark.parametrize("od_name", ['MITgcm_curv_nc'])
@pytest.mark.parametrize("delta",   [None, True])
def test_survey_stations(od_name, delta):
    
    # Extract od
    od_in = oceandatasets[od_name]
    
    # Get random coords
    Xsurv = [od_in.dataset['XC'].isel(X=0,  Y=0).values,
             od_in.dataset['XC'].isel(X=-1, Y=-1).values]
    Ysurv = [od_in.dataset['YC'].isel(X=0,  Y=0).values,
             od_in.dataset['YC'].isel(X=-1, Y=-1).values]
    
    if delta is True:
        _, _, delta = great_circle_path(float(Ysurv[0]), float(Xsurv[0]), float(Ysurv[1]), float(Xsurv[1]))
        delta = float(delta[1]/10)
    
    # Run survey
    with pytest.warns(UserWarning):
        od_out = od_in.subsample.survey_stations(Xsurv=Xsurv, Ysurv=Ysurv, delta=delta)
        od_out = survey_stations(od_in, Xsurv=Xsurv, Ysurv=Ysurv, delta=delta)

    # Check dimensions
    for dim in ['station', 'station_midp']:
        assert dim in od_out.dataset.dims and dim in od_out.grid_coords['station']
    for dim in ['X', 'Y', 'Xp1', 'Yp1']:
        assert dim not in od_out.dataset.dims 
    
    # Check verteces
    if delta is None:
        assert od_out.dataset['XC'].isel(station=0).values==Xsurv[0] and od_out.dataset['XC'].isel(station=-1).values==Xsurv[-1]
        assert od_out.dataset['YC'].isel(station=0).values==Ysurv[0] and od_out.dataset['YC'].isel(station=-1).values==Ysurv[-1]
        assert len(od_out.dataset['station'])==2
    else:
        assert len(od_out.dataset['station'])>2

        
# ==========
# PARTICLES
# ==========

od_in = oceandatasets['MITgcm_curv_nc']
points =  ['C', 'G', 'U', 'V']
for point in points:
    od_in = od_in.merge_into_oceandataset(od_in.dataset['X'+point].drop(od_in.dataset['X'+point].coords).rename('test'+point))
data = xr.DataArray(np.random.randn(2, 3), coords={'x': ['a', 'b']}, dims=('x', 'y'))
times = od_in.dataset['time'].isel(time=slice(2,-2))
npart = 10
X = od_in.dataset['XC'].stack(XY=('X', 'Y')).values
Y = od_in.dataset['YC'].stack(XY=('X', 'Y')).values
Xpart = np.empty((len(times), npart))
Ypart = np.empty((len(times), npart))
Zpart = np.empty((len(times), npart))
for t in range(len(times)):
    inds = np.random.choice(len(X), npart)
    Ypart[t, :] = Y[inds]
    Xpart[t, :] = X[inds]
    Zpart[t, :] = np.random.choice(od_in.dataset['Z'], npart)
    
def test_particle_properties():
    
    # Warning due to midp
    with pytest.warns(UserWarning):
        od_out = od_in.subsample.particle_properties(times=times, Xpart=Xpart, Ypart=Ypart, Zpart=Zpart)
        od_out = particle_properties(od_in, times=times, Xpart=Xpart, Ypart=Ypart, Zpart=Zpart)
    
    # Check dimensions
    assert 'particle' in od_out.dataset.dims
    for dim in ['X', 'Y', 'Xp1', 'Yp1']:
        assert dim not in od_out.dataset.dims 
    assert len(od_out.dataset['time'])==len(times)
    assert len(od_out.dataset['particle'])==npart
    
    # Check variables
    assert od_out.dataset ==0
    assert np.array_equal(od_out.dataset['time'].values, times) 
    assert np.array_equal(od_out.dataset['XC'].values, Xpart) 
    assert np.array_equal(od_out.dataset['YC'].values, Ypart) 
    assert np.array_equal(od_out.dataset['Z'].values, Zpart) 
    for var in od_out.dataset.variables:
        if var not in ['time', 'time_midp', 'particle']:
            assert od_out.dataset[var].shape == Xpart.shape