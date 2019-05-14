# General packages
import pytest
import cartopy
import scipy
import os
import numpy as np
import xarray as xr
import copy as copy
import shutil

# Oceanspy modules
from oceanspy import open_oceandataset, OceanDataset
from oceanspy import (AVAILABLE_PARAMETERS,
                      DEFAULT_PARAMETERS,
                      OCEANSPY_AXES)

# Directory
Datadir = './oceanspy/tests/Data/'
od = open_oceandataset.from_netcdf('{}MITgcm_rect_nc.nc'
                                   ''.format(Datadir))

# Remove global attributes
ds = od.dataset
ds.attrs = {}
clean_od = OceanDataset(ds)

# Aliased od
ds = od.dataset
aliases = {var: var+'_alias' for var in ds.data_vars}
ds = ds.rename(aliases)
alias_od = OceanDataset(ds).set_aliases(aliases)

# Grid only wihtout time_midp
ds = clean_od.dataset
ds = ds.drop(ds.data_vars)
ds = ds.drop(['Y', 'time_midp'])
aliases = {dim: dim+'_alias' for dim in ds.dims}
ds_aliases = ds.rename(aliases)
nomidp_od = OceanDataset(ds)
alias_nomidp_od = OceanDataset(ds_aliases).set_aliases(aliases)

# Od periodic
per_od = od.set_grid_periodic(['X'])

# Od cartesian
ds = od.dataset
for coord in ['XC', 'YC', 'XU', 'YU', 'XV', 'YV', 'XG', 'YG']:
    ds[coord] = np.sin(ds[coord])
cart_od = OceanDataset(ds).set_parameters({'rSphere': None})


# ============
# OceanDataset
# ============
@pytest.mark.parametrize("dataset",
                         [1,
                          od.dataset,
                          clean_od.dataset,
                          alias_od.dataset,
                          per_od.dataset])
def test_OceanDataset(dataset):
    if not isinstance(dataset, xr.Dataset):
        # Raise error if wrong format
        with pytest.raises(TypeError):
            OceanDataset(dataset)
    else:
        # Just test that copy and repr don't raise errors
        new_od = OceanDataset(dataset)
        copy.copy(new_od)
        repr(new_od)


# ===========
# ATTRIBUTES
# ===========
@pytest.mark.parametrize("od", [od, clean_od])
@pytest.mark.parametrize("name", [1, 'test_name'])
@pytest.mark.parametrize("overwrite", [None, True, False])
def test_name(od, name, overwrite):
    if not isinstance(name, str):
        # Raise error if wrong format
        with pytest.raises(TypeError):
            od.set_name(name=name, overwrite=overwrite)
    elif od.name is not None and overwrite is None:
        # Raise error if need to provide overwrite
        with pytest.raises(ValueError):
            od.set_name(name=name, overwrite=overwrite)
    else:
        # Check overwrite
        new_od = od.set_name(name=name, overwrite=overwrite)
        if overwrite is True or overwrite is None or od.name is None:
            assert new_od.name == name
        else:
            assert new_od.name == od.name+'_'+name
    # Inhibit setter
    with pytest.raises(AttributeError):
        od.name = name


@pytest.mark.parametrize("od", [od, clean_od])
@pytest.mark.parametrize("description", [1, 'test_description'])
@pytest.mark.parametrize("overwrite", [None, True, False])
def test_description(od, description, overwrite):
    if not isinstance(description, str):
        # Raise error if wrong format
        with pytest.raises(TypeError, match=r'.*must be.*'):
            od.set_description(description=description, overwrite=overwrite)
    elif od.description is not None and overwrite is None:
        # Raise error if need to provide overwrite
        with pytest.raises(ValueError, match=r'.*has been previously set.*'):
            od.set_description(description=description, overwrite=overwrite)
    else:
        # Check overwrite
        new_od = od.set_description(description=description,
                                    overwrite=overwrite)
        if overwrite is True or overwrite is None or od.description is None:
            assert new_od.description == description
        else:
            assert new_od.description == od.description+'_'+description
    # Inhibit setter (check only once)
    with pytest.raises(AttributeError):
        od.description = description


@pytest.mark.parametrize("od", [od, alias_od])
@pytest.mark.parametrize("aliases", [1,
                                     {var: var+'_alias'
                                      for var in od.dataset.data_vars},
                                     {'Temp':
                                      'S_alias'}])
@pytest.mark.parametrize("overwrite", [None, True, False])
def test_aliases(od, aliases, overwrite):
    if not isinstance(aliases, dict):
        # Raise error if wrong format
        with pytest.raises(TypeError):
            od.set_aliases(aliases=aliases, overwrite=overwrite)
    elif od.aliases is not None and overwrite is None:
        # Raise error if need to provide overwrite
        with pytest.raises(ValueError):
            od.set_aliases(aliases=aliases, overwrite=overwrite)
    else:
        # Check overwrite
        new_od = od.set_aliases(aliases=aliases, overwrite=overwrite)
        if od.aliases is None:
            assert new_od.aliases == aliases
        elif overwrite is True:
            od.aliases == {**od.aliases, **aliases}
        elif overwrite is False:
            assert od.aliases == {**aliases, **od.aliases}
    # Inhibit setter
    with pytest.raises(AttributeError):
        od.aliases = aliases


@pytest.mark.parametrize("od", [od, alias_od])
def test_dataset(od):
    # Compare private and public datasets
    assert od.dataset.equals(od._ds.rename(od.aliases))
    # Inhibit setter
    with pytest.raises(AttributeError):
        od.dataset = od.dataset


@pytest.mark.parametrize("od", [od, clean_od])
@pytest.mark.parametrize("parameters",
                         [1, None,
                          {'mypar': 1},
                          {'eq_state': 1},
                          {'eq_state': '1'},
                          {'eq_state': 'jmd95'}])
def test_parameters(od, parameters):
    if not isinstance(parameters, dict):
        # Raise error if wrong format
        with pytest.raises(TypeError):
            od.set_parameters(parameters=parameters)
    elif not set(parameters.keys()).issubset(AVAILABLE_PARAMETERS.keys()):
        # Warn but add if parameter is useless
        with pytest.warns(UserWarning):
            new_od = od.set_parameters(parameters=parameters)
            assert set(parameters.keys()).issubset(new_od.parameters.keys())
            assert set(od.parameters.keys()).issubset(new_od.parameters.keys())
    elif 'eq_state' in parameters.keys():
        if parameters['eq_state'] not in AVAILABLE_PARAMETERS['eq_state']:
            # Raise error if wrong format or not available choice
            if isinstance(parameters['eq_state'], str):
                with pytest.raises(ValueError):
                    od.set_parameters(parameters=parameters)
            else:
                with pytest.raises(TypeError):
                    od.set_parameters(parameters=parameters)
    else:
        # Check defaults
        new_od = od.set_parameters(parameters=parameters)
        if parameters is None:
            assert new_od.parameters == DEFAULT_PARAMETERS
        else:
            assert set(parameters.keys()).issubset(new_od.parameters.keys())
    # Inhibit setter
    with pytest.raises(AttributeError):
        od.parameters = parameters


@pytest.mark.parametrize("od", [od, nomidp_od])
@pytest.mark.parametrize("grid_coords", [1,
                                         {'wrong_axes':
                                          {'Y': None,
                                           'Yp1': 0.5}},
                                         {'X':
                                          {'X': None,
                                           'Xp1': 0.5,
                                           'wrong_dim': 0.5}},
                                         {'Y':
                                          {'Y': None,
                                           'Yp1': 0.5}},
                                         {'Y':
                                          {'Y': None,
                                           'Yp1': 0.5}},
                                         {'Y':
                                          {'Yp1': -0.5},
                                          'time': {'time': -0.5}}])
@pytest.mark.parametrize("add_midp", [1, True, False])
@pytest.mark.parametrize("overwrite", [None, True, False])
def test_grid_coords(od, grid_coords, add_midp, overwrite):
    if (not isinstance(grid_coords, dict) or not isinstance(add_midp, bool)):
        # Raise error if wrong format
        with pytest.raises(TypeError):
            od.set_grid_coords(grid_coords=grid_coords,
                               add_midp=add_midp,
                               overwrite=overwrite)
    elif not set(grid_coords.keys()).issubset(OCEANSPY_AXES):
        # Raise error if axis not availables
        with pytest.raises(ValueError):
            od.set_grid_coords(grid_coords=grid_coords,
                               add_midp=add_midp,
                               overwrite=overwrite)
    elif od.grid_coords is not None and not isinstance(overwrite, bool):
        # Raise error if overwrite is None
        with pytest.raises(ValueError):
            od.set_grid_coords(grid_coords=grid_coords,
                               add_midp=add_midp,
                               overwrite=overwrite)
    else:
        # Check set and midp
        new_od = od.set_grid_coords(grid_coords=grid_coords,
                                    add_midp=add_midp,
                                    overwrite=overwrite)
        if add_midp is True and 'time' in grid_coords.keys():
            assert 'time_midp' in new_od.dataset.dims
        elif add_midp is False and overwrite is True:
            assert new_od.grid_coords == grid_coords

    # Inhibit setter
    with pytest.raises(AttributeError):
        od.grid_coords = grid_coords


@pytest.mark.parametrize("od, grid_coords",
                         [(alias_nomidp_od, {'X':
                                             {'X_alias': None,
                                              'Xp1_alias': 0.5},
                                             'time':
                                             {'time_alias': 0.5}})])
def test_grid_coords_aliases(od, grid_coords):
    new_od = od.set_grid_coords(grid_coords, add_midp=True)
    assert 'time_alias_midp' in list(new_od.dataset.dims)
    assert 'time_midp' in list(new_od._ds.dims)
    assert list(new_od.grid.axes) == list(new_od._grid.axes)


@pytest.mark.parametrize("od", [od, per_od])
@pytest.mark.parametrize("grid_periodic", [1, ['Y'], []])
def test_grid_periodic(od, grid_periodic):
    if not isinstance(grid_periodic, list):
        # Raise error if wrong format
        with pytest.raises(TypeError):
            od.set_grid_periodic(grid_periodic=grid_periodic)
    else:
        new_od = od.set_grid_periodic(grid_periodic=grid_periodic)
        assert new_od.grid_periodic == grid_periodic
    # Inhibit setter
    with pytest.raises(AttributeError):
        od.grid_periodic = grid_periodic


@pytest.mark.parametrize("od, grid_coords",
                         [(od, {'X': {'X': None, 'Xp1': 0.5}}),
                          (od, {'X': {'X': None, 'Xp1': 0.5,
                                      'wrong_dim': -0.5}})])
def test_grid(od, grid_coords):
    new_od = od.set_grid_coords(grid_coords=grid_coords, overwrite=True)

    if any([True
            for axis in grid_coords
            for dim in grid_coords[axis]
            if dim not in od.dataset.dims]):
        with pytest.warns(UserWarning):
            new_od.grid
            new_od._grid
    else:
        new_od.grid
        new_od._grid

        # Inhibit setter
        with pytest.raises(AttributeError):
            new_od.grid = new_od.grid

        # Inhibit setter
        with pytest.raises(AttributeError):
            new_od._grid = new_od._grid


@pytest.mark.parametrize("projection", [1, None, 'Mercator', 'wrong'])
def test_projection(projection):
    if not isinstance(projection, (str, type(None))):
        # Raise error if wrong format
        with pytest.raises(TypeError):
            od.set_projection(projection=projection)
    elif projection == 'wrong':
        # Raise error for wrong projectons
        with pytest.raises(ValueError):
            od.set_projection(projection=projection)
    else:
        new_od = od.set_projection(projection=projection)
        # Check values
        if projection is None:
            assert new_od.projection == projection
        elif projection == 'Mercator':
            assert isinstance(new_od.projection, cartopy.crs.Mercator)

    # Inhibit setter
    with pytest.raises(AttributeError):
        od.projection = projection


@pytest.mark.parametrize("od", [od, cart_od])
@pytest.mark.parametrize("grid_pos", [1, 'wrong', 'C', 'G', 'U', 'V'])
def test_create_tree(od, grid_pos):
    if not isinstance(grid_pos, str):
        # Raise error if wrong format
        with pytest.raises(TypeError):
            od.create_tree(grid_pos=grid_pos)
    elif grid_pos not in ['C', 'G', 'U', 'V']:
        # Raise error if not a valid grid point
        with pytest.raises(ValueError):
            od.create_tree(grid_pos=grid_pos)
    else:
        tree = od.create_tree(grid_pos=grid_pos)
        assert isinstance(tree, scipy.spatial.ckdtree.cKDTree)


# Take ds out
ds = od.dataset


@pytest.mark.parametrize("od", [od])
@pytest.mark.parametrize("obj", [1,
                                 ds['Temp']*ds['U'],
                                 xr.zeros_like(ds['Temp']).rename('Temp'),
                                 xr.zeros_like(ds['Temp']).rename('newTemp')])
@pytest.mark.parametrize("overwrite", [None, True, False])
def test_merge_into_oceandataset(od, obj, overwrite):
    check1 = (not isinstance(obj, (xr.Dataset, xr.DataArray)))
    check2 = (not isinstance(overwrite, bool))
    if check1 or check2:
        # Raise error if wrong format
        with pytest.raises(TypeError):
            od.merge_into_oceandataset(obj=obj, overwrite=overwrite)
    elif isinstance(obj, (xr.DataArray)) and obj.name is None:
        # Raise error when DataArray doesn't have name
        with pytest.raises(ValueError):
            od.merge_into_oceandataset(obj=obj, overwrite=overwrite)
    elif obj.name in od.dataset.variables:
        # Check warning
        with pytest.warns(UserWarning):
            new_od = od.merge_into_oceandataset(obj=obj, overwrite=overwrite)
        if overwrite is True:
            assert new_od.dataset[obj.name].equals(obj)
        elif overwrite is False:
            assert new_od.dataset[obj.name].equals(od.dataset[obj.name])
    else:
        new_od = od.merge_into_oceandataset(obj=obj, overwrite=overwrite)
        assert new_od.dataset[obj.name].equals(obj)

        # Test dataset
        ds_obj = obj.to_dataset()
        new_od = od.merge_into_oceandataset(obj=ds_obj, overwrite=overwrite)
        assert new_od.dataset[obj.name].equals(obj)


@pytest.mark.parametrize("od", [od])
@pytest.mark.parametrize("compute", [True, False])
def test_save_load(od, compute):
    path = 'test_path'

    od.to_netcdf(path=path+'.nc', compute=compute)
    new_od = open_oceandataset.from_netcdf(path+'.nc')
    os.remove(path+'.nc')

    od.to_zarr(path=path, compute=compute)
    new_od = open_oceandataset.from_zarr(path)
    shutil.rmtree(path, ignore_errors=True)
