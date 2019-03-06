import pytest
import xarray as xr
import numpy  as np
import pandas as pd
import sys

from oceanspy import OceanDataset
from oceanspy._oceandataset import _setter_error_message, _wrong_axes_error_message
from . datasets import datasets
from oceanspy   import DEFAULT_PARAMETERS, AVAILABLE_PARAMETERS, TYPE_PARAMETERS, OCEANSPY_AXES

OVERWRITE_ERROR_MESSAGE = "has been previously set: `overwrite` must be bool"

# Drop attributes
ds_in = datasets['MITgcm_rect_bin']
ds_in.attrs = {}
od_in = OceanDataset(ds_in)

def test_name():
    
    # OceanDataset
    assert od_in.name is None
    
    # Wrong type
    test_str = 1
    with pytest.raises(TypeError) as e:
        od_in.set_name(test_str)
    assert str(e.value) == "`name` must be str"
    
    # Inhibit setter
    test_str = 'test_name'.lower() 
    with pytest.raises(AttributeError) as e:
        od_in.name = test_str
    assert str(e.value) == _setter_error_message('name')
    
    # Check set
    od_out = od_in.set_name(test_str)
    assert od_out.name == test_str
    assert od_in.name != od_out.name
    
    # Check overwrite
    with pytest.raises(ValueError) as e:
        od_out = od_out.set_name(test_str)
    assert str(e.value) == "[name] "+OVERWRITE_ERROR_MESSAGE
    assert od_out.set_name(test_str, overwrite=False).name == od_out.name+'_'+od_out.name
    assert od_out.set_name(test_str.upper(), overwrite=True).name == test_str.upper()
    

def test_description():
    
    # OceanDataset
    assert od_in.description is None
    
    # Wrong type
    test_str = 1
    with pytest.raises(TypeError) as e:
        od_in.set_description(test_str)
    assert str(e.value) == "`description` must be str"
    
    # Inhibit setter
    test_str = 'test_description'.lower() 
    with pytest.raises(AttributeError) as e:
        od_in.description = test_str
    assert str(e.value) == _setter_error_message('description')
    
    # Check set
    od_out = od_in.set_description(test_str)
    assert od_out.description == test_str
    assert od_in.description != od_out.description
    
    # Check overwrite
    with pytest.raises(ValueError) as e:
        od_out = od_out.set_description(test_str)
    assert str(e.value) == "[description] "+OVERWRITE_ERROR_MESSAGE
    assert od_out.set_description(test_str, overwrite=False).description == od_out.description+'_'+od_out.description
    assert od_out.set_description(test_str.upper(), overwrite=True).description == test_str.upper()
    
def test_dataset():
    
    # OceanDataset
    assert isinstance(od_in.dataset, xr.Dataset)
    
    # Inhibit setter
    with pytest.raises(AttributeError):
        od_in.dataset = od_in.dataset
    
def test_aliases():
    
    # OceanDataset
    assert od_in.aliases is None
    
    # Wrong type
    test_dict = 1
    with pytest.raises(TypeError) as e:
        od_in.set_aliases(test_dict)
    assert str(e.value) == "`aliases` must be dict"
    
    # Inhibit setter
    test_dict = {var: var.lower() for var in od_in.dataset.variables}
    with pytest.raises(AttributeError) as e:
        od_in.aliases = test_dict
    assert str(e.value) == _setter_error_message('aliases')
    
    # Check setter
    od_alias = OceanDataset(od_in.dataset.rename(test_dict))
    od_out   = od_in.set_aliases(test_dict)
    assert set(test_dict.keys()).issubset(od_out._ds.variables)
    assert set(test_dict.values()).issubset(od_out.dataset.variables)
    
    # Check reopen
    od_reopen = OceanDataset(od_out.dataset)
    assert od_out.dataset.equals(od_reopen.dataset)
    assert od_out._ds.equals(od_reopen._ds)
    
def test_parameters():
    
    # OceanDataset
    assert od_in.parameters == DEFAULT_PARAMETERS
    
    # Parameters default and checks
    assert DEFAULT_PARAMETERS.keys() == TYPE_PARAMETERS.keys()
    assert set(AVAILABLE_PARAMETERS.keys()).issubset(AVAILABLE_PARAMETERS.keys())
    
    # Wrong type
    test_dict = 1
    with pytest.raises(TypeError) as e:
        od_in.set_parameters(test_dict)
    assert str(e.value) == "`parameters` must be dict"
    
    # Inhibit setter
    test_dict = DEFAULT_PARAMETERS.copy()
    test_dict['c_p'] = 0
    with pytest.raises(AttributeError) as e:
        od_in.parameters = test_dict
    assert str(e.value) == _setter_error_message('parameters')
    
    # Check set
    od_out = od_in.set_parameters(test_dict)
    assert od_out.parameters == test_dict
    assert od_in.parameters != od_out.parameters
    
    # Check wrong parametes
    with pytest.raises(TypeError):
        od_out.set_parameters({'eq_state': 1})
    with pytest.raises(ValueError):
        od_out.set_parameters({'eq_state': 'test'})
        
    # Check new parameters
    with pytest.warns(UserWarning):
        od_out = od_in.set_parameters({'test': 'test'})
    assert 'test' in od_out.parameters
    
def test_grid_coords():
    
    # OceanDataset
    assert od_in.grid_coords is None
    
    # Wrong type
    test_dict = 1
    with pytest.raises(TypeError) as e:
        od_in.set_grid_coords(test_dict)
    assert str(e.value) == "`grid_coords` must be dict"
    
    # Inhibit setter
    test_dict = {'dim0': 1} 
    with pytest.raises(AttributeError) as e:
        od_in.grid_coords = test_dict
    assert str(e.value) == _setter_error_message('grid_coords')
    
    # Wrong axis
    with pytest.raises(ValueError) as e:
        od_out = od_in.set_grid_coords(test_dict)
    assert str(e.value) == (_wrong_axes_error_message(list(test_dict.keys())))
    
    # Wrong type
    test_dict = {'X': 1} 
    good_test_dict = {'Y': {'Y': None, 'Yp1': 0.5}}
    with pytest.raises(TypeError) as e:
        od_out = od_in.set_grid_coords(test_dict)
    assert str(e.value) == "Invalid grid_coords. grid_coords example: {}".format(good_test_dict)
    
    # Wrong shift
    test_dict = {'X': {'X': 1}} 
    list_shift = [0.5, None, -0.5]
    with pytest.raises(ValueError) as e:
        od_out = od_in.set_grid_coords(test_dict)
    assert str(e.value) == ("[{}] not a valid c_grid_axis_shift."
                            " Available options are {}").format(1, list_shift)
    
    # Check set
    od_out = od_in.set_grid_coords(good_test_dict)
    assert od_out.grid_coords == good_test_dict
    assert od_in.grid_coords != od_out.grid_coords
    
    # Check overwrite
    with pytest.raises(ValueError) as e:
        od_out = od_out.set_grid_coords(good_test_dict)
    assert str(e.value) == "[grid_coords] "+OVERWRITE_ERROR_MESSAGE
    
    test_dict = {'Y': {'Y': 0.5, 'Yp1': None}, 
                 'X': {'X': None, 'Xp1': 0.5}}
    assert od_out.set_grid_coords(test_dict, overwrite=False).grid_coords == {**test_dict, **od_out.grid_coords}
    assert od_out.set_grid_coords(test_dict, overwrite=True).grid_coords  == {**od_out.grid_coords, **test_dict}
    
    # Midp
    assert od_in.set_grid_coords({'X': {'X': None}} , add_midp=True).dataset == od_in.dataset
    od_out = od_in.set_grid_coords({'X': {'X': 0.5}} , add_midp=True)
    assert 'X_midp' in od_out.dataset.variables
    assert np.array_equal(od_out.dataset['X_midp'].values, 
                          (od_in.dataset['X'].values[:-1] + od_in.dataset['X'].values[1:])/2)
    
    # Check aliases
    od_alias = OceanDataset(od_in.dataset.rename({'X': 'x'}))
    od_alias = od_alias.set_aliases({'X': 'x'})
    od_out = od_alias.set_grid_coords({'X': {'x': 0.5}} , add_midp=True)
    assert 'X_midp' in od_out._ds.variables
    assert 'x_midp' in od_out.dataset.variables
    
def test_grid_periodic():
    
    # OceanDataset
    assert od_in.grid_periodic==[]
    
    # Wrong type
    test_list = 1
    with pytest.raises(TypeError) as e:
        od_in.set_grid_periodic(test_list)
    assert str(e.value) == "`grid_periodic` must be list"
    
    # Wrong axis
    test_list = ['dim']
    with pytest.raises(ValueError) as e:
        od_out = od_in.set_grid_periodic(test_list)
    assert str(e.value) == (_wrong_axes_error_message(test_list))
    
    
    # Inhibit setter
    with pytest.raises(AttributeError) as e:
        od_in.grid_periodic = test_list
    assert str(e.value) == _setter_error_message('grid_periodic')
    
    
    # Check set
    test_list = ['X']
    od_out = od_in.set_grid_periodic(test_list)
    assert od_out.grid_periodic == test_list
    assert od_in.grid_periodic != od_out.grid_periodic
    
    
    # Check overwrite
    with pytest.raises(ValueError) as e:
        od_out = od_out.set_grid_periodic(test_list)
    assert str(e.value) == "[grid_periodic] "+OVERWRITE_ERROR_MESSAGE
    add_test_list = ['Y']
    assert set(od_out.set_grid_periodic(add_test_list, overwrite=False).grid_periodic) == set(test_list + add_test_list)
    assert od_out.set_grid_periodic(add_test_list, overwrite=True).grid_periodic == add_test_list
    
def test_grid():
    
    # OceanDataset
    assert od_in.grid is None
    
    # Check grid
    od_out = od_in.set_grid_coords({'X': {'X': None, 'Xp1': 0.5}})
    assert od_out.grid is not None
    assert od_out.grid.axes['X']._periodic is False
    assert od_out.grid.axes['X'].coords['center'].name == 'X'
    assert od_out.grid.axes['X'].coords['outer'].name  == 'Xp1'
    od_out = od_out.set_grid_periodic(['X'])
    assert od_out.grid.axes['X']._periodic is True
    
    # Check wrong name
    with pytest.warns(UserWarning):
        od_in.set_grid_coords({'X': {'X': None, 'Xp1': 0.5, 'test': -0.5}}).grid
    
    # Inhibit setter
    with pytest.raises(AttributeError):
        od_in.grid = 1
    with pytest.raises(AttributeError):
        od_in._grid = 1
    
    # Check aliases
    test_dict = {var: var.lower() for var in od_in.dataset.variables}
    od_alias  = OceanDataset(od_in.dataset.rename(test_dict))
    od_out    = od_in.set_aliases(test_dict).set_grid_coords({'X': {'x': None, 'xp1': 0.5}})
    assert od_out.grid.axes['X'].coords['center'].name == 'x'
    assert od_out.grid.axes['X'].coords['outer'].name  == 'xp1'
    assert od_out._grid.axes['X'].coords['center'].name == 'X'
    assert od_out._grid.axes['X'].coords['outer'].name  == 'Xp1'
    
def test_projection():

    # OceanDataset
    assert od_in.projection is None

    # Wrong type
    test_str = 1
    with pytest.raises(TypeError) as e:
        od_in.set_projection(test_str)
    assert str(e.value) == "`projection` must be str or None"
        
    # Inhibit setter
    test_str = 'PlateCarree' 
    with pytest.raises(AttributeError) as e:
        od_in.projection = test_str
    assert str(e.value) == _setter_error_message('projection')
    
    # Check set
    od_out = od_in.set_projection(test_str)
    if 'cartopy' in sys.modules:
        assert od_out.projection is not None 
        od_out = od_out.set_projection(None)
        assert od_out.projection is None
    else:
        with pytest.warns(UserWarning):
            assert od_out.projection is None 

    
    