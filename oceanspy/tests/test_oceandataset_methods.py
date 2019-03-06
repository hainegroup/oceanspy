import pytest
import xarray as xr
from . datasets import (aliased_ods, oceandatasets)

od_in = oceandatasets['MITgcm_rect_nc']



# Check both rectilinear and curvilinear
@pytest.mark.parametrize("od_name", ['MITgcm_rect_nc', 'MITgcm_curv_nc'])
@pytest.mark.parametrize("grid_pos", ['C', 'G', 'U', 'V', 'wrong'])
def test_tree(od_name, grid_pos):
    # Check regualar and aliases
    for ods in [aliased_ods, oceandatasets]:
        od = ods[od_name]
        # Test points
        if grid_pos=='wrong':
            with pytest.raises(ValueError):
                tree = od.create_tree(grid_pos)
        else:
            tree = od.create_tree(grid_pos)
            
            
def test_merge_into_oceandataset():
    
    # da without name
    da = od_in.dataset['XC']*od_in.dataset['YC']
    with pytest.raises(ValueError) as e:
        od_out = od_in.merge_into_oceandataset(da)
    assert str(e.value) == "xarray.DataArray doesn't have a name. Set it using da.rename()"
    
    # da different name
    da = da.rename('test')
    od_out = od_in.merge_into_oceandataset(da)
    assert od_out.dataset['test'].equals(da)
    
    # ds
    ds = xr.merge([da.rename('test1'), da.rename('test2')])
    od_out = od_in.merge_into_oceandataset(ds)
    assert set(['test1', 'test2']).issubset(od_out.dataset.variables)
    
    # da
    da = xr.zeros_like(od_in.dataset['XC'])
    with pytest.raises(ValueError):
        od_out = od_in.merge_into_oceandataset(da)
        
    od_out = od_in.merge_into_oceandataset(da, overwrite=True)
    
    
    

    
    