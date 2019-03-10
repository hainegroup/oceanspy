import pytest
import xarray as xr
import copy
from oceanspy.compute import *
from oceanspy import utils, open_oceandataset
from numpy.testing import assert_array_equal, assert_allclose
import numpy as np
from . test_compute_static import check_params, ds_out_IN_od_out

# Add variables
od_in = open_oceandataset.from_netcdf('./oceanspy/tests/Data/budgets.nc')

def test_heat_budget():
    
    # Compute heat_budget
    ds_out = heat_budget(od_in)
    for var in ds_out.data_vars:
        assert ds_out[var].attrs['units']     == 'degC/s'
        check_params(ds_out, var, ['rho0', 'c_p'])
        
    tendH = ds_out['tendH']
    check = ds_out['adv_hConvH'] + ds_out['adv_vConvH'] + ds_out['dif_vConvH'] + ds_out['kpp_vConvH'] + ds_out['forcH']
    assert np.fabs(tendH-check).max().values<1.e-17
    
    # Test shortcut
    od_out=od_in.compute.heat_budget()
    ds_out_IN_od_out(ds_out, od_out)
    
def test_salt_budget():
    
    # Compute salt_budget
    ds_out = salt_budget(od_in)
    for var in ds_out.data_vars:
        assert ds_out[var].attrs['units']     == 'psu/s'
        check_params(ds_out, var, ['rho0'])
        
    tendS = ds_out['tendS']
    check = ds_out['adv_hConvS'] + ds_out['adv_vConvS'] + ds_out['dif_vConvS'] + ds_out['kpp_vConvS'] + ds_out['forcS']
    assert np.fabs(tendS-check).max().values<1.e-15
    
    # Test shortcut
    od_out=od_in.compute.salt_budget()
    ds_out_IN_od_out(ds_out, od_out)