import pytest
import xarray as xr
import copy
from . datasets import oceandatasets
from oceanspy.compute import *
from oceanspy import utils
from numpy.random import rand
from numpy.testing import assert_array_equal, assert_allclose
import numpy as np

# Add variables
od_in = copy.copy(oceandatasets['MITgcm_rect_nc'])

# Add Temp and S
dimList = ['time', 'Z', 'Y', 'X']
dimSize = [len(od_in.dataset[dim]) for dim in dimList]
ds_in = xr.Dataset({'Temp': xr.DataArray(rand(*dimSize), dims=dimList),
                    'S'   : xr.DataArray(rand(*dimSize), dims=dimList)})
od_in = od_in.merge_into_oceandataset(ds_in)    

# Add HFac
ds_in = xr.Dataset({'HFacC': xr.DataArray(np.ones([len(od_in.dataset[dim]) for dim in ['Z', 'Y'  , 'X']])  , dims=['Z', 'Y'  , 'X']),
                    'HFacS': xr.DataArray(np.ones([len(od_in.dataset[dim]) for dim in ['Z', 'Yp1', 'X']])  , dims=['Z', 'Yp1', 'X']),
                    'HFacW': xr.DataArray(np.ones([len(od_in.dataset[dim]) for dim in ['Z', 'Y'  , 'Xp1']]), dims=['Z', 'Y'  , 'Xp1'])})
od_in = od_in.merge_into_oceandataset(ds_in)    

def test_potential_density_anomaly():
    
    # Compute Sigma0
    ds_out = potential_density_anomaly(od_in)
    assert ds_out['Sigma0'].attrs['units']     == 'kg/m^3'
    assert ds_out['Sigma0'].attrs['long_name'] == 'potential density anomaly'
    check_params(ds_out, 'Sigma0', ['eq_state'])
    
    # Check values
    Sigma0 = eval("utils.dens{}(od_in.dataset['S'].values, od_in.dataset['Temp'].values, 0)".format(od_in.parameters['eq_state']))
    assert_array_equal(ds_out['Sigma0'].values+1000, Sigma0)
    
    # Test shortcut
    od_out=od_in.compute.potential_density_anomaly()
    ds_out_IN_od_out(ds_out, od_out)
    
def test_Brunt_Vaisala_frequency():
    
    # Compute Sigma0
    ds_out = Brunt_Vaisala_frequency(od_in)
    assert ds_out['N2'].attrs['units']     == 's^-2'
    assert ds_out['N2'].attrs['long_name'] == 'Brunt-Väisälä Frequency'
    check_params(ds_out, 'N2', ['g', 'rho0'])
    
    # Check values
    dSigma0_dZ = gradient(od_in, 'Sigma0', 'Z')
    dSigma0_dZ = dSigma0_dZ['dSigma0_dZ']
    assert_allclose(-dSigma0_dZ.values*od_in.parameters['g']/od_in.parameters['rho0'], ds_out['N2'].values)
    
    # Test shortcut
    od_out=od_in.compute.Brunt_Vaisala_frequency()
    ds_out_IN_od_out(ds_out, od_out)
    

    
    
    
    

def check_params(ds, varName, params):
    for par in params:
        assert par in ds[varName].attrs['OceanSpy_parameters']
    
def ds_out_IN_od_out(ds_out, od_out):
    for var in ds_out.data_vars: 
        assert_array_equal(od_out.dataset[var].values, ds_out[var].values)