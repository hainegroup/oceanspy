# Instructions for developers:
#
# This is the main object of OceanSpy.
# All attributes are stored as global attributes (strings!) of the xr.Dataset.
# When users request an attribute, it is decoded from the global attributes.
# Thus, there are custom attribute setters (class setters are inhibited).
#
# There are private and public objects.
# Private objects use OceanSpy's reference aliases (_ds, _grid),
# while public objects are mirrors of the private objects using custom aliases.
#
# All functions in other modules that operate on od,
# must be added here in the shortcuts section.
#
# Add new attributes/methods in docs/api.rst

##############################################################################
# TODO: create list of OceanSpy name and add link under aliases.
# TODO: create a dictionary with parameters description and add under aliases.
# TODO: add more xgcm options. E.g., default boundary method.
# TODO: implement xgcm autogenerate in _set_coords,
#       set_grid_coords, set_coords when released
# TODO: Use the coords parameter to create xgcm grid instead of
#       _crate_grid.
#       We will pass dictionary in xgcm.Grid,
#       and we can have the option of usining comodo attributes
#       (currently cleaned up so switched off)
##############################################################################

# Required dependencies (private)
import xarray as _xr
import copy as _copy
import numpy as _np
import warnings as _warnings
import sys as _sys
from collections import OrderedDict as _OrderedDict

# From OceanSpy (private)
from . import utils as _utils
from ._ospy_utils import (_check_instance, _check_oceanspy_axes,
                          _setter_error_message, _check_list_of_string,
                          _create_grid, _rename_coord_attrs)
from . subsample import _subsampleMethods
from . compute import _computeMethods
from . plot import _plotMethods
from . animate import _animateMethods

# Recommended dependencies (private)
try:
    import cartopy.crs as _ccrs
except ImportError:  # pragma: no cover
    pass
try:
    from scipy import spatial as _spatial
except ImportError:  # pragma: no cover
    pass
try:
    from dask.diagnostics import ProgressBar as _ProgressBar
except ImportError:  # pragma: no cover
    pass


class OceanDataset:
    """
    OceanDataset combines a :py:obj:`xarray.Dataset`
    with other objects used by OceanSpy (e.g., xgcm.Grid).

    Additional objects are attached to the
    :py:obj:`xarray.Dataset` as global attributes.
    """

    def __init__(self, dataset):
        """
        Parameters
        ----------
        dataset: xarray.Dataset
            The multi-dimensional, in memory, array database.

        References
        ----------
        http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html
        """

        # Check parameters
        _check_instance({'dataset': dataset}, 'xarray.Dataset')

        # Initialize dataset
        self._ds = dataset.copy()

        # Apply aliases
        self = self._apply_aliases()

    def __copy__(self):
        """
        Shallow copy
        """

        return OceanDataset(dataset=self.dataset.copy())

    def __repr__(self):

        main_info = ['<oceanspy.OceanDataset>']
        main_info.append('\nMain attributes:')
        main_info.append("   .dataset: %s" %
                         self.dataset.__repr__()
                         [self.dataset.__repr__().find('<'):
                          self.dataset.__repr__().find('>')+1])
        if self.grid is not None:
            main_info.append("   .grid: %s" %
                             self.grid.__repr__()
                             [self.grid.__repr__().find('<'):
                              self.grid.__repr__().find('>')+1])
        if self.projection is not None:
            main_info.append("   .projection: %s" %
                             self.projection.__repr__()
                             [self.projection.__repr__().find('<'):
                              self.projection.__repr__().find('>')+1])

        more_info = ['\n\nMore attributes:']
        if self.name:
            more_info.append("   .name: %s" %
                             self.name)
        if self.description:
            more_info.append("   .description: %s" %
                             self.description)
        more_info.append("   .parameters: %s" %
                         type(self.parameters))
        if self.aliases:
            more_info.append("   .aliases: %s" %
                             type(self.aliases))
        if self.grid_coords:
            more_info.append("   .grid_coords: %s" %
                             type(self.grid_coords))
        if self.grid_periodic:
            more_info.append("   .grid_periodic: %s" %
                             type(self.grid_periodic))

        info = '\n'.join(main_info)
        info = info + '\n'.join(more_info)
        return info

    # ===========
    # ATTRIBUTES
    # ===========
    # -------------------
    # name
    # -------------------
    @property
    def name(self):
        """
        Name of the OceanDataset.
        """
        name = self._read_from_global_attr('name')

        return name

    @name.setter
    def name(self, name):
        """
        Inhibit setter.
        """
        raise AttributeError(_setter_error_message('name'))

    def set_name(self, name, overwrite=None):
        """
        Set name of the OceanDataset.

        Parameters
        ----------
        name: str
            Name of the OceanDataset.
        overwrite: bool or None
            If None, raises error if name has been previously set.
            If True, overwrite previous name.
            If False, combine with previous name.
        """
        # Check parameters
        _check_instance({'name': name}, 'str')

        # Set name
        self = self._store_as_global_attr(name='name',
                                          attr=name,
                                          overwrite=overwrite)

        return self

    # -------------------
    # description
    # -------------------
    @property
    def description(self):
        """
        Description of the OceanDataset.
        """
        description = self._read_from_global_attr('description')

        return description

    @description.setter
    def description(self, description):
        """
        Inhibit setter.
        """
        raise AttributeError(_setter_error_message('description'))

    def set_description(self, description, overwrite=None):
        """
        Set description of the OceanDataset.

        Parameters
        ----------
        description: str
            Desription of the OceanDataset
        overwrite: bool or None
            If None, raises error if description has been previously set.
            If True, overwrite previous description.
            If False, combine with previous description.
        """
        # Check parameters
        _check_instance({'description': description}, 'str')

        # Set description
        self = self._store_as_global_attr(name='description',
                                          attr=description,
                                          overwrite=overwrite)

        return self

    # -------------------
    # aliases
    # -------------------
    @property
    def aliases(self):
        """
        A dictionary to connect custom variable names
        to OceanSpy reference names.
        Keys are OceanSpy reference names, values are custom names:
        {'ospy_name': 'custom_name'}
        """

        aliases = self._read_from_global_attr('aliases')

        return aliases

    @property
    def _aliases_flipped(self):
        """
        Flip aliases:
        From {'ospy_name': 'custom_name'}
        to {'custom_name': 'ospy_name'}
        """
        if self.aliases:
            aliases_flipped = {custom: ospy
                               for ospy, custom in self.aliases.items()}
        else:
            return self.aliases

        return aliases_flipped

    @aliases.setter
    def aliases(self, aliases):
        """
        Inhibit setter.
        """

        raise AttributeError(_setter_error_message('aliases'))

    def set_aliases(self, aliases, overwrite=None):
        """
        Set aliases to connect custom variables names
        to OceanSpy reference names.

        Parameters
        ----------
        aliases: dict
            Keys are OceanSpy names, values are custom names:
            {'ospy_name': 'custom_name'}
        overwrite: bool or None
            If None, raises error if aliases has been previously set.
            If True, overwrite previous aliases.
            If False, combine with previous aliases.
        """

        # Check parameters
        _check_instance({'aliases': aliases}, 'dict')

        # Set aliases
        self = self._store_as_global_attr(name='aliases',
                                          attr=aliases,
                                          overwrite=overwrite)

        # Apply aliases
        self = self._apply_aliases()

        return self

    def _apply_aliases(self):
        """
        Check if there are variables with custom name in _ds,
        and rename to OceanSpy reference name
        """
        if self._aliases_flipped:
            aliases = {custom: ospy
                       for custom, ospy in self._aliases_flipped.items()
                       if custom in self._ds.variables
                       or custom in self._ds.dims}
            self._ds = self._ds.rename(aliases)

        return self

    # -------------------
    # dataset
    # -------------------
    @property
    def dataset(self):
        """
        xarray.Dataset: A multi-dimensional, in memory, array database.

        References
        ----------
        http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html
        """

        # Show _ds with renamed variables.
        dataset = self._ds.copy()
        if self.aliases:
            aliases = {ospy: custom for ospy, custom in self.aliases.items()
                       if ospy in self._ds
                       or ospy in self._ds.dims}
            dataset = dataset.rename(aliases)

        return dataset

    @dataset.setter
    def dataset(self, dataset):
        """
        Inhibit setter.
        """
        raise AttributeError("Set a new dataset using "
                             "`oceanspy.OceanDataset(dataset)`")

    # -------------------
    # parameters
    # -------------------
    @property
    def parameters(self):
        """
        A dictionary defining model parameters that are used by OceanSpy.
        Default values are used for parameters that have not been set
        (see :py:const:`oceanspy.DEFAULT_PARAMETERS`).
        """
        from oceanspy import DEFAULT_PARAMETERS
        parameters = self._read_from_global_attr('parameters')

        if parameters is None:
            parameters = DEFAULT_PARAMETERS
        else:
            parameters = {**DEFAULT_PARAMETERS, **parameters}

        return parameters

    @parameters.setter
    def parameters(self, parameters):
        """
        Inhibit setter.
        """
        raise AttributeError(_setter_error_message('parameters'))

    def set_parameters(self, parameters):
        """
        Set model parameters used by OceanSpy.
        See :py:const:`oceanspy.DEFAULT_PARAMETERS` for a list of parameters,
        and :py:const:`oceanspy.PARAMETERS_PARAMETERS_DESCRIPTION`
        for their description.
        See :py:const:`oceanspy.AVAILABLE_PARAMETERS` for a list of parameters
        with predefined options.

        Parameters
        ----------
        parameters: dict
            {'name': value}
        """
        from oceanspy import (DEFAULT_PARAMETERS,
                              AVAILABLE_PARAMETERS,
                              TYPE_PARAMETERS)

        # Check parameters
        _check_instance({'parameters': parameters}, 'dict')

        # Check parameters
        warn_params = []
        for key, value in parameters.items():
            if key not in DEFAULT_PARAMETERS.keys():
                warn_params = warn_params + [key]
            else:
                if not isinstance(value, TYPE_PARAMETERS[key]):
                    raise TypeError("Invalid [{}]. "
                                    "Check oceanspy.TYPE_PARAMETERS"
                                    "".format(key))
                check1 = (key in AVAILABLE_PARAMETERS.keys())
                if check1 and (value not in AVAILABLE_PARAMETERS[key]):
                    raise ValueError("Requested [{}] not available. "
                                     "Check oceanspy.AVAILABLE_PARAMETERS"
                                     "".format(key))

        if len(warn_params) != 0:
            _warnings.warn("{} are not OceanSpy parameters"
                           "".format(warn_params), stacklevel=2)

        # Set parameters
        self = self._store_as_global_attr(name='parameters',
                                          attr=parameters,
                                          overwrite=True)

        return self

    # -------------------
    # grid_coords
    # -------------------
    @property
    def grid_coords(self):
        """
        Grid coordinates used by :py:obj:`xgcm.Grid`.

        References
        ----------
        https://xgcm.readthedocs.io/en/stable/grids.html#Grid-Metadata
        """

        grid_coords = self._read_from_global_attr('grid_coords')

        return grid_coords

    @grid_coords.setter
    def grid_coords(self, grid_coords):
        """
        Inhibit setter.
        """

        raise AttributeError(_setter_error_message('grid_coords'))

    def set_grid_coords(self, grid_coords, add_midp=False, overwrite=None):
        """
        Set grid coordinates used by :py:obj:`xgcm.Grid`.

        Parameters
        ----------
        grid_coords: str
            Grid coordinates used by :py:obj:`xgcm.Grid`.
            Keys are axes, and values are dict with
            key=dim and value=c_grid_axis_shift.
            Available c_grid_axis_shift are {0.5, None, -0.5}.
            E.g., {'Y': {'Y': None, 'Yp1': 0.5}}
            See :py:const:`oceanspy.OCEANSPY_AXES` for a list of axes
        add_midp: bool
            If true, add inner dimension (mid points)
            to axes with outer dimension only.
            The new dimension will be named
            as the outer dimension + '_midp'
        overwrite: bool or None
            If None, raises error if grid_coords has been previously set.
            If True, overwrite previous grid_coors.
            If False, combine with previous grid_coors.

        References
        ----------
        https://xgcm.readthedocs.io/en/stable/grids.html#Grid-Metadata
        """

        # Check parameters
        _check_instance({'grid_coords': grid_coords,
                         'add_midp':    add_midp},
                        {'grid_coords': 'dict',
                         'add_midp':    'bool'})

        # Check axes
        _check_oceanspy_axes(list(grid_coords.keys()))

        # Set grid_coords
        self = self._store_as_global_attr(name='grid_coords',
                                          attr=grid_coords,
                                          overwrite=overwrite)

        if add_midp:
            grid_coords = {}
            for axis in self.grid_coords:
                check1 = (len(self.grid_coords[axis]) == 1)
                check2 = (list(self.grid_coords[axis].values())[0] is not None)
                if check1 and check2:

                    # Deal with aliases
                    dim = list(self.grid_coords[axis].keys())[0]
                    if self._aliases_flipped and dim in self._aliases_flipped:
                        _dim = self._aliases_flipped[dim]
                        self = self.set_aliases({_dim+'_midp':
                                                 dim+'_midp'}, overwrite=False)
                    else:
                        _dim = dim

                    # Midpoints are averages of outpoints
                    midp = (self._ds[_dim].values[:-1] +
                            self._ds[_dim].diff(_dim)/2).rename({_dim:
                                                                 _dim+'_midp'})
                    self._ds[_dim+'_midp'] = _xr.DataArray(midp,
                                                           dims=(_dim+'_midp'))
                    if 'units' in self._ds[_dim].attrs:
                        units = self._ds[_dim].attrs['units']
                        self._ds[_dim+'_midp'].attrs['units'] = units
                    if 'long_name' in self._ds[_dim].attrs:
                        long_name = self._ds[_dim].attrs['long_name']
                        long_name = 'Mid-points of {}'.format(long_name)
                        self._ds[_dim+'_midp'].attrs['long_name'] = long_name
                    if 'description' in self._ds[_dim].attrs:
                        desc = self._ds[_dim].attrs['description']
                        desc = 'Mid-points of {}'.format(desc)
                        self._ds[_dim+'_midp'].attrs['description'] = desc

                    grid_coords[axis] = {**self.grid_coords[axis],
                                         dim+'_midp': None}

            self = self._store_as_global_attr(name='grid_coords',
                                              attr=grid_coords,
                                              overwrite=False)
        return self

    # -------------------
    # grid_periodic
    # -------------------
    @property
    def grid_periodic(self):
        """
        List of :py:obj:`xgcm.Grid` axes that are periodic.
        """

        grid_periodic = self._read_from_global_attr('grid_periodic')
        if not grid_periodic:
            grid_periodic = []

        return grid_periodic

    @grid_periodic.setter
    def grid_periodic(self, grid_periodic):
        """
        Inhibit setter.
        """

        raise AttributeError(_setter_error_message('grid_periodic'))

    def set_grid_periodic(self, grid_periodic):
        """
        Set grid axes that will be treated as periodic by :py:obj:`xgcm.Grid`.
        Axes that are not set periodic are non-periodic by default.

        Parameters
        ----------
        grid_periodic: list
            List of periodic axes.
            See :py:const:`oceanspy.OCEANSPY_AXES` for a list of axes
        """

        # Check parameters
        _check_instance({'grid_periodic': grid_periodic}, 'list')

        # Check axes
        _check_oceanspy_axes(grid_periodic)

        # Set grid_periodic
        # Use overwrite True by default because
        # xgcm default is all grid_priodic True.
        self = self._store_as_global_attr(name='grid_periodic',
                                          attr=grid_periodic,
                                          overwrite=True)

        return self

    # -------------------
    # grid
    # -------------------
    @property
    def grid(self):
        """
        :py:obj:`xgcm.Grid`: A collection of axes,
        which is a group of coordinates that all lie
        along the same physical dimension
        but describe different positions relative to a grid cell.

        References
        ----------
        https://xgcm.readthedocs.io/en/stable/api.html#Grid
        """

        dataset = self.dataset.copy()
        coords = self.grid_coords
        periodic = self.grid_periodic
        grid = _create_grid(dataset, coords, periodic)

        return grid

    @property
    def _grid(self):
        """
        :py:obj:`xgcm.Grid` with OceanSpy reference names.
        """

        aliases = self.aliases
        coords = self.grid_coords

        if aliases and coords:
            # Flip aliases
            aliases = {custom: ospy for ospy, custom in aliases.items()}

            # Rename coords
            for axis in coords:
                for dim in coords[axis]:
                    if dim in aliases:
                        coords[axis][aliases[dim]] = coords[axis].pop(dim)

        dataset = self._ds.copy()
        periodic = self.grid_periodic
        grid = _create_grid(dataset, coords, periodic)

        return grid

    @grid.setter
    def grid(self, grid):
        """
        Inhibit setter.
        """
        raise AttributeError("Set a new grid using "
                             ".set_grid_coords and .set_periodic")

    @_grid.setter
    def _grid(self, grid):
        """
        Inhibit setter.
        """
        raise AttributeError("Set a new _grid using "
                             ".set_grid_coords and .set_periodic")

    # -------------------
    # projection
    # -------------------
    @property
    def projection(self):
        """
        Cartopy projection of the OceanDataset.
        """

        projection = self._read_from_global_attr('projection')
        if projection:
            if projection == 'None':
                projection = eval(projection)
            else:
                if 'cartopy' not in _sys.modules:  # pragma: no cover
                    _warnings.warn("cartopy is not available,"
                                   " so projection is None", stacklevel=2)
                    projection = None
                else:
                    projection = eval('_ccrs.{}'.format(projection))

        return projection

    @projection.setter
    def projection(self, projection):
        """
        Inhibit setter.
        """

        raise AttributeError(_setter_error_message('projection'))

    def set_projection(self, projection, **kwargs):
        """
        Set Cartopy projection of the OceanDataset.

        Parameters
        ----------
        projection: str or None
            Cartopy projection of the OceanDataset.
            Use None to remove projection.
        **kwargs:
            Keyword arguments for the projection.
            E.g., central_longitude=0.0 for PlateCarree

        References
        ----------
        https://scitools.org.uk/cartopy/docs/latest/crs/projections.html
        """

        # Check parameters
        if projection is not None:
            # Check
            _check_instance({'projection': projection}, 'str')
            if not hasattr(_ccrs, projection):
                raise ValueError("{} is not a cartopy projection"
                                 "".format(projection))
            projection = '{}(**{})'.format(projection, kwargs)
        else:
            projection = str(projection)

        # Set projection
        self = self._store_as_global_attr(name='projection', attr=projection,
                                          overwrite=True)

        return self

    # ===========
    # METHODS
    # ===========
    def create_tree(self, grid_pos='C'):
        """
        Create a scipy.spatial.cKDTree for quick nearest-neighbor lookup.

        Parameters
        ----------
        grid_pos: str
            Grid position. Options: {'C', 'G', 'U', 'V'}

        Returns
        -------
        tree: scipy.spatial.cKDTree
            Return a xKDTree object that can be used to query a point.

        References
        ----------
        | cKDTree:
          https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
        | Grid:
          https://mitgcm.readthedocs.io/en/latest/algorithm/horiz-grid.html
        """

        # Check parameters
        _check_instance({'grid_pos': grid_pos}, 'str')

        grid_pos_list = ['C', 'G', 'U', 'V']
        if grid_pos not in grid_pos_list:
            raise ValueError("`grid_pos` must be one of {}:"
                             "\nhttps://mitgcm.readthedocs.io"
                             "/en/latest/algorithm/horiz-grid.html"
                             "".format(grid_pos_list))

        # Convert if it's not cartesian
        Y = self._ds['Y'+grid_pos]
        X = self._ds['X'+grid_pos]
        R = self.parameters['rSphere']
        if R:
            x, y, z = _utils.spherical2cartesian(Y=Y, X=X, R=R)
        else:
            x = X
            y = Y
            z = _xr.zeros_like(Y)

        # Stack
        x_stack = x.stack(points=x.dims).values
        y_stack = y.stack(points=y.dims).values
        z_stack = z.stack(points=z.dims).values

        # Construct KD-tree
        tree = _spatial.cKDTree(_np.column_stack((x_stack, y_stack, z_stack)))

        return tree

    def merge_into_oceandataset(self, obj, overwrite=False):
        """
        Merge a Dataset or DataArray into the OceanDataset.

        Parameters
        ----------
        obj: xarray.DataArray or xarray.Dataset
            object to merge.
        overwrite: bool or None
            If True, overwrite existing DataArrays with same name.
            If False, use xarray.merge.
        """

        # Check and make dataset
        if not isinstance(obj, (_xr.DataArray, _xr.Dataset)):
            raise TypeError('`obj` must be xarray.DataArray or xarray.Dataset')
        _check_instance({'overwrite': overwrite}, 'bool')

        # Check name
        obj = obj.drop(obj.coords)
        if isinstance(obj, _xr.DataArray):
            if obj.name is None:
                raise ValueError("xarray.DataArray doesn't have a name."
                                 "Set it using da.rename()")
            else:
                obj = obj.to_dataset()

        # Merge
        dataset = self.dataset
        var2drop = [var for var in obj.variables if var in dataset]
        if overwrite is False:
            obj = obj.drop(var2drop)
            if len(var2drop) != 0:
                _warnings.warn('{} will not be merged.'
                               '\nSet `overwrite=True` if you wish otherwise.'
                               ''.format(var2drop), stacklevel=2)
        else:
            if len(var2drop) != 0:
                _warnings.warn('{} will be overwritten.'
                               ''.format(var2drop), stacklevel=2)
        for var in obj.data_vars:
            # Store dimension attributes that get lost
            attrs = {}
            for dim in obj[var].dims:
                if dim not in dataset.dims:
                    pass
                elif all([i == j
                          for i, j in zip(obj[dim].attrs.items(),
                                          dataset[dim].attrs.items())]):
                    attrs[dim] = dataset[dim].attrs

            # Merge
            dataset[var] = obj[var]

            # Add attributes
            for dim, attr in attrs.items():
                dataset[dim].attrs = attr

        return OceanDataset(dataset)

    def to_netcdf(self, path, **kwargs):
        """
        Write contents to a netCDF file.

        Parameters
        ----------
        path: str
            Path to which to save.
        **kwargs:
            Keyword arguments for :py:func:`xarray.Dataset.to_netcdf()`

        References
        ----------
        http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_netcdf.html
        """

        # Check parameters
        _check_instance({'path': path}, 'str')

        # to_netcdf doesn't like coordinates attribute
        dataset = _rename_coord_attrs(self.dataset)

        # Compute
        compute = kwargs.pop('compute', None)
        print('Writing dataset to [{}].'.format(path))
        if compute is None or compute is False:
            delayed_obj = dataset.to_netcdf(path, compute=False, **kwargs)
            with _ProgressBar():
                delayed_obj.compute()
        else:
            dataset.to_netcdf(path, compute=compute, **kwargs)

    def to_zarr(self, path, **kwargs):
        """
        Write contents to a zarr group.

        Parameters
        ----------
        path: str
            Path to which to save.
        **kwargs:
            Keyword arguments for :py:func:`xarray.Dataset.to_zarr()`

        References
        ----------
        http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html
        """

        # Check parameters
        _check_instance({'path': path}, 'str')

        # to_zarr doesn't like coordinates attribute
        dataset = _rename_coord_attrs(self.dataset)

        # Compute
        compute = kwargs.pop('compute', None)
        print('Writing dataset to [{}].'.format(path))
        if compute is None or compute is False:
            delayed_obj = dataset.to_zarr(path, compute=False, **kwargs)
            with _ProgressBar():
                delayed_obj.compute()
        else:
            dataset.to_zarr(path, compute=compute, **kwargs)

    # ==================================
    # IMPORT (used by open_oceandataset)
    # ==================================
    def shift_averages(self, averageList=None):
        """
        Shift average variables to time_midp.
        Average variables are defined as
        variables with attribute [original_output='average'],
        or variables in averageList.

        Parameters
        ----------
        averageList: 1D array_like, str, or None
            List of variables (strings).
        """

        if averageList is not None:
            averageList = _check_list_of_string(averageList, 'averageList')
        else:
            averageList = []

        for var in self._ds.data_vars:
            original_output = self._ds[var].attrs.pop('original_output', None)
            if original_output == 'average' or var in averageList:
                ds_tmp = self._ds[var].drop('time').isel(time=slice(1, None))
                self._ds[var] = ds_tmp.rename({'time': 'time_midp'})
            if original_output is not None:
                self._ds[var].attrs['original_output'] = original_output
        return self

    def manipulate_coords(self, fillna=False, coords1Dfrom2D=False,
                          coords2Dfrom1D=False, coordsUVfromG=False):
        """
        Manipulate coordinates to make them compatible with OceanSpy.

        Parameters
        ----------
        fillna: bool
            If True, fill NaNs in 2D coordinates
            (e.g., NaNs are created by MITgcm exch2).
        coords1Dfrom2D: bool
            If True, infer 1D coordinates from 2D coordinates (mean of 2D).
            Use with rectilinear grid only.
        coords2Dfrom1D: bool
            If True, infer 2D coordinates from 1D coordinates (brodacast 1D).
        coordsUVfromCG: bool
            If True, compute missing coords (U and V points) from G points.

        References
        ----------
        Grid:
        https://mitgcm.readthedocs.io/en/latest/algorithm/horiz-grid.html
        """

        # Copy because the dataset will change
        self = _copy.copy(self)

        # Coordinates are dimensions only
        self._ds = self._ds.reset_coords()

        # Fill nans (e.g., because of exch2)
        if fillna:
            coords = ['YC', 'XC', 'YG', 'XG', 'YU', 'XU', 'YV', 'XV']
            dims = ['X', 'Y', 'Xp1', 'Yp1', 'Xp1', 'Y', 'X', 'Yp1']

            for i, (coord, dim) in enumerate(zip(coords, dims)):
                if coord in self._ds.variables:
                    ds_tmp = self._ds[coord].ffill(dim).bfill(dim).persist()
                    self._ds[coord] = ds_tmp

        # Get U and V by rolling G
        if coordsUVfromG:

            for i, (point_pos, dim2roll) in enumerate(zip(['U', 'V'],
                                                          ['Yp1', 'Xp1'])):
                for dim in ['Y', 'X']:
                    coord = self._ds[dim+'G'].rolling(**{dim2roll: 2})
                    coord = coord.mean().dropna(dim2roll)
                    coord = coord.drop(coord.coords).rename({dim2roll:
                                                             dim2roll[0]})
                    self._ds[dim+point_pos] = coord
                    if 'units' in self._ds[dim+'G'].attrs:
                        units = self._ds[dim+'G'].attrs['units']
                        self._ds[dim+point_pos].attrs['units'] = units

        # For cartesian grid we can use 1D coordinates
        if coords1Dfrom2D:
            # Take mean
            self._ds['Y'] = self._ds['YC'].mean('X',
                                                keep_attrs=True).persist()
            self._ds['X'] = self._ds['XC'].mean('Y',
                                                keep_attrs=True).persist()
            self._ds['Yp1'] = self._ds['YG'].mean('Xp1',
                                                  keep_attrs=True).persist()
            self._ds['Xp1'] = self._ds['XG'].mean('Yp1',
                                                  keep_attrs=True).persist()

        # Get 2D coordinates broadcasting 1D
        if coords2Dfrom1D:
            # Broadcast
            self._ds['YC'], self._ds['XC'] = _xr.broadcast(self._ds['Y'],
                                                           self._ds['X'])
            self._ds['YG'], self._ds['XG'] = _xr.broadcast(self._ds['Yp1'],
                                                           self._ds['Xp1'])
            self._ds['YU'], self._ds['XU'] = _xr.broadcast(self._ds['Y'],
                                                           self._ds['Xp1'])
            self._ds['YV'], self._ds['XV'] = _xr.broadcast(self._ds['Yp1'],
                                                           self._ds['X'])

            # Add units
            dims2 = ['YC', 'XC', 'YG', 'XG',
                     'YU', 'XU', 'YV', 'XV']
            dims1 = ['Y', 'X', 'Yp1', 'Xp1',
                     'Y', 'Xp1', 'Yp1', 'X']
            for i, (D2, D1) in enumerate(zip(dims2, dims1)):
                if 'units' in self._ds[D1].attrs:
                    self._ds[D2].attrs['units'] = self._ds[D1].attrs['units']

        # Set 2D coordinates
        self._ds = self._ds.set_coords(['YC', 'XC',
                                        'YG', 'XG',
                                        'YU', 'XU',
                                        'YV', 'XV'])

        # Attributes (use xmitgcm)
        try:
            from xmitgcm import variables
            if self.parameters['rSphere'] is None:
                coords = variables.horizontal_coordinates_cartesian
                add_coords = _OrderedDict(
                    XU=dict(attrs=dict(standard_name="longitude_at_u_location",
                                       long_name="longitude",
                                       units="degrees_east",
                                       coordinate="YU XU")),
                    YU=dict(attrs=dict(standard_name="latitude_at_u_location",
                                       long_name="latitude",
                                       units="degrees_north",
                                       coordinate="YU XU")),
                    XV=dict(attrs=dict(standard_name="longitude_at_v_location",
                                       long_name="longitude",
                                       units="degrees_east",
                                       coordinate="YV XV")),
                    YV=dict(attrs=dict(standard_name="latitude_at_v_location",
                                       long_name="latitude",
                                       units="degrees_north",
                                       coordinate="YV XV")))
            else:
                coords = variables.horizontal_coordinates_spherical
                add_coords = _OrderedDict(
                    XU=dict(attrs=dict(standard_name=("plane_x_coordinate"
                                                      "_at_u_location"),
                                       long_name="x coordinate",
                                       units="m",
                                       coordinate="YU XU")),
                    YU=dict(attrs=dict(standard_name=("plane_y_coordinate"
                                                      "_at_u_location"),
                                       long_name="y coordinate",
                                       units="m",
                                       coordinate="YU XU")),
                    XV=dict(attrs=dict(standard_name=("plane_x_coordinate"
                                                      "_at_v_location"),
                                       long_name="x coordinate",
                                       units="m",
                                       coordinate="YV XV")),
                    YV=dict(attrs=dict(standard_name=("plane_y_coordinate"
                                                      "_at_v_location"),
                                       long_name="y coordinate",
                                       units="m",
                                       coordinate="YV XV")))
            coords = _OrderedDict(list(coords.items())
                                  + list(add_coords.items()))
            for var in coords:
                attrs = coords[var]['attrs']
                for attr in attrs:
                    if attr not in self._ds[var].attrs:
                        self._ds[var].attrs[attr] = attrs[attr]
        except ImportErrror:  # pragma: no cover
            pass

        return self

    # =====
    # UTILS
    # =====
    def _store_as_global_attr(self, name, attr, overwrite):
        """
        Store an OceanSpy attribute as dataset global attribute.

        Parameters
        ----------
        name: str
            Name of the attribute. Attribute is stored as OceanSpy_+name.
        attr: str or dict
            Attribute to store
        overwrite: bool or None
            If None, raises error if attr has been previously set.
            If True, overwrite previous attributes.
            If False, combine with previous attributes.
        """

        # Attribute name
        name = 'OceanSpy_'+name

        if overwrite is None and name in self._ds.attrs:
            raise ValueError("[{}] has been previously set: "
                             "`overwrite` must be bool"
                             "".format(name.replace("OceanSpy_", "")))

        # Copy because attributes are added to _ds
        self = _copy.copy(self)

        # Store
        if not overwrite and name in self._ds.attrs:
            prev_attr = self._ds.attrs[name]
            if prev_attr[0] == "{" and prev_attr[-1] == "}":
                attr = {**eval(prev_attr), **attr}
            else:
                attr = prev_attr + '_' + attr

        self._ds.attrs[name] = str(attr)

        return self

    def _read_from_global_attr(self, name):
        """
        Read an OceanSpy attribute stored as dataset global attribute.

        Parameters
        ----------
        name: str
            Name of the attribute.
            Attribute is decoded from 'OceanSpy_'+name.

        Returns
        -------
        attr: str or dict
            Attribute that has been decoded.
        """

        # Attribute name
        name = 'OceanSpy_'+name

        # Check if attributes exists
        if name not in self._ds.attrs:
            return None

        # Read attribute
        attr = self._ds.attrs[name]
        check_dict = (attr[0] == '{' and attr[-1] == '}')
        check_list = (attr[0] == '[' and attr[-1] == ']')
        if check_dict or check_list:
            attr = eval(attr)

        return attr

    # ===========
    # SHORTCUTS
    # ===========
    @property
    def subsample(self):
        """
        Access :py:mod:`oceanspy.subsample` functions.
        """
        return _subsampleMethods(self)

    @property
    def compute(self):
        """
        Access :py:mod:`oceanspy.compute` functions,
        and merge the computed Dataset into the OceanDataset.
        Set overwrite=True
        to overwrite DataArrays already existing in the OceanDataset.
        """
        return _computeMethods(self)

    @property
    def plot(self):
        """
        Access :py:mod:`oceanspy.plot` functions.
        """

        return _plotMethods(self)

    @property
    def animate(self):
        """
        Access :py:mod:`oceanspy.animate` functions.
        """

        return _animateMethods(self)
