import pytest
from . datasets import sin_od
from oceanspy.compute import *
import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

def test_gradient():
    
    varNameList = ['sinZ', 'sinY', 'sinX']
    grad_ds = gradient(sin_od, varNameList=varNameList) 
    
    # Test shortcut
    od = sin_od.compute.gradient(varNameList=varNameList)
    assert set(grad_ds.data_vars).issubset(od.dataset.data_vars)
    
    # sin' = cos
    for varName in varNameList:
        for axis in sin_od.grid_coords.keys():
            gradName = 'd'+varName+'_'+'d'+axis
            var = grad_ds[gradName]
            if varName[-1]!=axis:
                assert var.min().values==grad_ds[gradName].max().values==0
            else:
                coords  = {coord[0]: var[coord] for coord in var.coords}
                coords['Z'], coords['Y'], coords['X'] = xr.broadcast(coords['Z'], coords['Y'], coords['X'])
                check = np.cos(coords[axis])
                mask = xr.where(np.logical_or(check.isnull(), var.isnull()), 0, 1)
                # Assert using numpy
                assert_allclose(var.where(mask).values, check.where(mask).values, 1.E-3)
                
    
                
                
def test_laplacian():
    
    varNameList = ['sinZ', 'sinY', 'sinX']
    lapl_ds = laplacian(sin_od, varNameList=varNameList) 
    
    # Test shortcut
    od = sin_od.compute.laplacian(varNameList=varNameList)
    assert set(lapl_ds.data_vars).issubset(od.dataset.data_vars)
    
    # sin' = cos
    for varName in varNameList:
        for axis in sin_od.grid_coords.keys():
            laplName = 'dd'+varName+'_'+'d'+axis+'_'+'d'+axis
            var = lapl_ds[laplName]
            if varName[-1]!=axis:
                assert var.min().values==lapl_ds[laplName].max().values==0
            else:
                coords  = {coord[0]: var[coord] for coord in var.coords}
                coords['Z'], coords['Y'], coords['X'] = xr.broadcast(coords['Z'], coords['Y'], coords['X'])
                check = -np.sin(coords[axis])
                mask = xr.where(np.logical_or(check.isnull(), var.isnull()), 0, 1)
                
                # Assert using numpy
                assert_allclose(var.where(mask).values, check.where(mask).values, 1.E-3)
                
def test_divergence():
    
    varNameList = ['sinUX', 'sinVY', 'sinWZ']
    dive_ds = divergence(sin_od, iName=varNameList[0], jName=varNameList[1], kName=varNameList[2]) 
    
    # Test shortcut
    od = sin_od.compute.divergence(iName=varNameList[0], jName=varNameList[1], kName=varNameList[2])
    assert set(dive_ds.data_vars).issubset(od.dataset.data_vars)
          
    # sin' = cos
    for varName in varNameList:
        axis = varName[-1]
        diveName = 'd'+varName+'_'+'d'+axis
        var = dive_ds[diveName]
        
        coords  = {coord[0]: var[coord] for coord in var.coords}
        coords['Z'], coords['Y'], coords['X'] = xr.broadcast(coords['Z'], coords['Y'], coords['X'])
        check = np.cos(coords[axis])
        mask = xr.where(np.logical_or(check.isnull(), var.isnull()), 0, 1)
        
        # Assert using numpy
        assert_allclose(var.where(mask).values, check.where(mask).values, 1.E-3)
        
def test_curl():
    
    velocities = [[None   , 'sinVZ', 'sinWY'],
                  ['sinUZ',    None, 'sinWX'],
                  ['sinUY', 'sinVX',    None]]
    
    for _, vels in enumerate(velocities):
        curl_ds = curl(sin_od, iName=vels[0], jName=vels[1], kName=vels[2]) 
                                           
        # Test shortcut
        od = sin_od.compute.curl(iName=vels[0], jName=vels[1], kName=vels[2])
        assert set(curl_ds.data_vars).issubset(od.dataset.data_vars)
                                               
        # sin' = cos                                       
        for var in curl_ds.data_vars: var = curl_ds[var]
            
        coords  = {coord[0]: var[coord] for coord in var.coords}
        coords['Z'], coords['Y'], coords['X'] = xr.broadcast(coords['Z'], coords['Y'], coords['X'])
        
        terms = var.name.split('-')
        for i, term in enumerate(terms):
            axis = term[-1]
            terms[i] = np.cos(coords[axis])
        check = terms[0] - terms[1]
        mask = xr.where(np.logical_or(check.isnull(), var.isnull()), 0, 1)
        
        # Assert using numpy
        assert_allclose(var.where(mask).values, check.where(mask).values, 1.E-3)

    
def test_weighted_mean():
    
    for var in sin_od.dataset.data_vars:
        wmean = weighted_mean(sin_od, varNameList=var)
        
        # Test shortcut
        od = sin_od.compute.weighted_mean(varNameList=var)
        assert set(wmean.data_vars).issubset(od.dataset.data_vars)
        
        # Extract mean and weight
        weight = wmean['weight_'+var].values
        wmean  = wmean['w_mean_'+var].values
        check = sin_od.dataset[var].mean().values
        assert np.float32(wmean)==np.float32(check)
        assert np.min(weight)==np.max(weight)

                
            

