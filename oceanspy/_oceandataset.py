import xarray   as _xr
import copy     as _copy
import xgcm     as _xgcm
import numpy    as _np
import warnings as _warnings
from . import subsample as _subsample
from . import compute   as _compute
from . import utils     as _utils

# TODO: add parameters check in set_parameters
# TODO: add more xgcm options. E.g., boundary method.

class OceanDataset:
    """
    OceanDataset combines a xarray.Dataset with other objects used by OceanSpy (e.g., xgcm.Grid).

    Additional objects are attached to the xarray.Dataset as global attributes.
    
    OceanDataset adds, reads, and decodes dataset global attributes.
    """
    
    def __init__(self, 
                 dataset,
                 name = None,
                 description = None):
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
        if not isinstance(dataset, _xr.Dataset):
            raise TypeError("`dataset` must be a xarray.Dataset")
            
        # Initialize dataset
        self._ds = dataset.copy()  
        
        # Apply aliases
        self = self._apply_aliases()
        
            
    def __copy__(self):
        """
        Shallow copy
        """

        return OceanDataset(dataset = self.dataset.copy())
    
    def __deepcopy__(self):
        """
        Deep copy
        """
        
        return OceanDataset(dataset = self.dataset.copy(deep=True))
       
    
    def __repr__(self):
        
        main_info = ['<oceanspy.OceanDataset>']
        main_info.append('\nMain attributes:')
        if self.name:
            main_info.append("   .name: %s" % self.name)
        if self.description:
            main_info.append("   .description: %s" % self.description)
        if self.dataset:
            main_info.append("   .dataset: %s" % self.dataset.__repr__()[self.dataset.__repr__().find('<'):
                                                                     self.dataset.__repr__().find('>')+1]) 
        if self.grid_coords:
            main_info.append("   .grid: %s"    % self.grid.__repr__()[self.grid.__repr__().find('<'):
                                                                    self.grid.__repr__().find('>')+1]) 
        more_info = ['\n\nMore attributes:']
        if self.parameters:
            more_info.append("   .parameters: %s" % type(self.parameters))
        if self.aliases:
            more_info.append("   .aliases: %s" % type(self.aliases))
        if self.grid_coords:
            more_info.append("   .grid_coords: %s" % type(self.grid_coords))
        if self.grid_periodic:
            more_info.append("   .grid_periodic: %s" % type(self.grid_periodic))
        
        info = '\n'.join(main_info)
        if len(more_info)>1:
            info = info+'\n'.join(more_info)
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
        Name of the OceanDataset
        """
        
        name = self._read_from_global_attr('name')
        
        return name

    @name.setter
    def name(self, name):
        """
        Inhibit setter
        """
        
        raise AttributeError("Set new `name` using .set_name")
    
    def set_name(self, name, overwrite=None):
        """
        Set name of the OceanDataset.
        
        Parameters
        ----------
        name: str
            Name of the OceanDataset
        overwrite: bool or None
            If None, raise error if name has been previously set.
            If True, overwrite previous name.  
            If False, combine with previous name.
        """
        
        # Check parameters
        if not isinstance(name, str):
            raise TypeError("`name` must be str")
            
        # Set name
        self = self._store_as_global_attr(name      = 'name', 
                                          attr      = name,
                                          overwrite = overwrite) 
        
        return self
    
    
    
    # -------------------
    # description
    # -------------------
    @property
    def description(self):
        """
        Description of the OceanDataset
        """
        
        description = self._read_from_global_attr('description')
        
        return description

    @description.setter
    def description(self, description):
        """
        Inhibit setter
        """
        
        raise AttributeError("Set new `description` using .set_description")
    
    def set_description(self, description, overwrite=None):
        """
        Set description of the OceanDataset.
        
        Parameters
        ----------
        description: str
            Desription of the OceanDataset
        overwrite: bool or None
            If None, raise error if description has been previously set.
            If True, overwrite previous description.  
            If False, combine with previous description.
        """
        
        # Check parameters
        if not isinstance(description, str):
            raise TypeError("`description` must be str")
            
        # Set description
        self = self._store_as_global_attr(name      = 'description', 
                                          attr      = description,
                                          overwrite = overwrite) 
        
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
        Inhibit setter
        """
        
        raise AttributeError("Do not set a new `dataset`, but initilize a new OceanDataset: oceanspy.OceanDataset(dataset)")
        
    # -------------------
    # grid
    # -------------------
    @property
    def grid(self):
        """
        xgcm.Grid: A collection of axis, which is a group of coordinates that all lie along the same physical dimension but describe different positions relative to a grid cell. 
        
        References
        ----------
        https://xgcm.readthedocs.io/en/stable/api.html#Grid
        """
        
        dataset  = self.dataset.copy()
        coords   = self.grid_coords
        periodic = self.grid_periodic
        grid = _create_grid(dataset, coords, periodic)
        
        return grid
    
    @property
    def _grid(self):
        """
        xgcm.Grid using aliases 
        """
        
        aliases = self.aliases
        coords  = self.grid_coords
        
        if aliases and coords:
            
            # Flip aliases
            aliases = {custom: ospy for ospy, custom in aliases.items()}
            
            # Rename coords
            for axis in coords:
                for dim in coords[axis]:
                    if dim in aliases:
                        coords[axis][aliases[dim]] = coords[axis].pop(dim)
        
        dataset  = self._ds.copy()
        periodic = self.grid_periodic
        grid = _create_grid(dataset, coords, periodic)
        
        return grid
    
    @grid.setter
    def grid(self, grid):
        """
        Inhibit setter
        """
        
        raise AttributeError("You can't change directly `grid`, but you must change OceanSpy's attributes (e.g., using .set_grid_coords and .set_periodic)")
        
        
    # -------------------
    # aliases
    # -------------------
    # TODO: add list of available aliases
    
    @property
    def aliases(self):
        """
        A dictionary to connect custom variable names to OceanSpy reference names.
        Keys are OceanSpy names, values are custom names: {'ospy_name': 'custom_name'}
        """
        
        aliases = self._read_from_global_attr('aliases')
        
        return aliases
    
    @property
    def _aliases_flipped(self):
        """
        Flip aliases: Keys are values names, values are ospy_name names: {'ospy_name': 'custom_name'}
        """
        if self.aliases:
            aliases_flipped = {custom: ospy for ospy, custom in self.aliases.items()}
        else: return self.aliases
        
        return aliases_flipped
    
    @aliases.setter
    def aliases(self, aliases):
        """
        Inhibit setter
        """
        
        raise AttributeError("Set new `aliases` using .set_aliases")
    
    def set_aliases(self, aliases, overwrite=None):
        """
        Set aliases to connect custom variables names to OceanSpy reference names.
        
        Parameters
        ----------
        aliases: dict
            Keys are OceanSpy names, values are custom names: {'ospy_name': 'custom_name'}
        overwrite: bool or None
            If None, raise error if aliases has been previously set.
            If True, overwrite previous aliases.  
            If False, combine with previous aliases.
        """
        
        # Check parameters
        if not isinstance(aliases, dict):
            raise TypeError("`aliases` must be dict")
            
        # Set aliases
        self = self._store_as_global_attr(name      = 'aliases', 
                                          attr      = aliases,
                                          overwrite = overwrite) 
        
        # Apply aliases
        self = self._apply_aliases()
        
        return self
    
    def _apply_aliases(self):
        """
        Check if there are variables with custom name in _ds, and rename to ospy name
        """
        # TODO: maybe add some check? E.g., raise error if alias is already available?
        if self._aliases_flipped:                    
            aliases = {custom: ospy for custom, ospy in self._aliases_flipped.items() 
                       if custom in self._ds 
                       or custom in self._ds.dims}
            self._ds = self._ds.rename(aliases)
            
        return self
    
    # -------------------
    # parameters
    # -------------------
    # TODO: add list of available parameters and documentation
    
    @property
    def parameters(self):
        """
        A dictionary defining model parameters that are used by OceanSpy.
        {'parameter_name': parameter value}
        If a parameter is not available, use default.
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
        Inhibit setter
        """
        
        raise AttributeError("Set new `parameters` using .set_parameters")
    
    def set_parameters(self, parameters):
        """
        Set model parameters used by OceanSpy.
        
        Parameters
        ----------
        parameters: dict
            {'parameter_name': parameter_value}
        """
        
        # Check parameters
        if not isinstance(parameters, dict):
            raise TypeError("`parameters` must be dict")
            
        # Set parameters
        self = self._store_as_global_attr(name      = 'parameters', 
                                          attr      = parameters,
                                          overwrite = False) 
        
        return self
    
    # -------------------
    # grid_coords
    # -------------------
    @property
    def grid_coords(self):
        """
        Grid coordinates used by xgcm.Grid
        
        References
        ----------
        https://xgcm.readthedocs.io/en/stable/grids.html#Grid-Metadata
        """
        
        grid_coords = self._read_from_global_attr('grid_coords')
        
        return grid_coords

    @grid_coords.setter
    def grid_coords(self, grid_coords):
        """
        Inhibit setter
        """
        
        raise AttributeError("Set new `grid_coords` using .set_grid_coords")
    
    def set_grid_coords(self, grid_coords, add_midp=False, overwrite=None):
        """
        Set grid coordinates used by xgcm.Grid.
        
        Parameters
        ----------
        grid_coords: str
            Grid coordinates used by xgcm.Grid.  
            Keys are axis, and values are dict with key=dim and value=c_grid_axis_shift.
            Available axis are {'X', 'Y', 'Z', 'time'}, and available c_grid_axis_shift are {0.5, None, -0.5}
        add_midp: bool
            If true, add inner dimension (mid points) to axis with outer dimension only.
            The new dimension will be called as the outer dimension + '_midp'
        overwrite: bool or None
            If None, raise error if grid_coords has been previously set.
            If True, overwrite previous grid_coors.  
            If False, combine with previous grid_coors.
            
        References
        ----------
        https://xgcm.readthedocs.io/en/stable/grids.html#Grid-Metadata
        """
        
        # Check parameters
        if not isinstance(grid_coords, dict):
            raise TypeError("`grid_coords` must be dict")
            
        list_axis  = ['X', 'Y', 'Z', 'time', 'mooring', 'station']
        list_dims  = [dim for dim in self.dataset.dims]
        list_shift = [0.5, None, -0.5]
        for axis in grid_coords:
            if axis not in list_axis:
                raise ValueError("[{}] not an OceanSpy axis. Available options are {}".format(axis, 
                                                                                       list_axis))
            if grid_coords[axis] is None: continue
            for dim in grid_coords[axis]:
                if dim not in list_dims:
                    raise ValueError("[{}] not a valid dimension. Available options are {}".format(dim,
                                                                                                 list_dims))
                if grid_coords[axis][dim] not in list_shift:
                    raise ValueError("[{}] not a valid c_grid_axis_shift. Available options are {}".format(grid_coords[axis][dim], 
                                                                                                         list_shift))
        if not isinstance(add_midp, (bool, type(None))):
            raise TypeError("`add_midp` must be bool")
            
        # Set grid_coords
        self = self._store_as_global_attr(name      = 'grid_coords', 
                                          attr      =  grid_coords,
                                          overwrite =  overwrite) 
        
        # Add midpoints
        # TODO: this is pretty much what the new xgcm functionality autogenerate does.
        #       We should implement it when there will be a stable release
        # TODO: add attributes
        if add_midp:
            grid_coords = {}
            for axis in self.grid_coords:
                if len(self.grid_coords[axis])==1 and list(self.grid_coords[axis].values())[0]==-0.5:
                    
                    # Deal with aliases
                    dim  = list(self.grid_coords[axis].keys())[0]    
                    if self._aliases_flipped and dim in self._aliases_flipped:
                        _dim = self._aliases_flipped[dim]
                        self = self.set_aliases({_dim+'_midp': dim+'_midp'}, overwrite=False)
                    else: _dim = dim
                    
                    # Midpoints are averages of outpoints
                    midp = (self._ds[_dim].values[:-1]+self._ds[_dim].diff(_dim)/2).rename({_dim: _dim+'_midp'})
                    self._ds[_dim+'_midp'] = _xr.DataArray(midp,
                                                           dims=(_dim+'_midp'))
                    if 'units' in self._ds[_dim].attrs:
                        self._ds[_dim+'_midp'].attrs['units'] = self._ds[_dim].attrs['units']
                        
                    # Update grid_coords
                    grid_coords[axis] = {**self.grid_coords[axis], dim+'_midp': None}
                    
                    
            self = self._store_as_global_attr(name      = 'grid_coords',
                                              attr      =  grid_coords,
                                              overwrite =  False)
        return self
    
    
    
    # -------------------
    # grid_periodic
    # -------------------
    @property
    def grid_periodic(self):
        """
        List of xgcm.Grid axes that are periodic
        """
        
        grid_periodic = self._read_from_global_attr('grid_periodic')
        if not grid_periodic:
            grid_periodic = []
            
        return grid_periodic

    @grid_periodic.setter
    def grid_periodic(self, grid_periodic):
        """
        Inhibit setter
        """
        
        raise AttributeError("Set new `grid_periodic` using .set_grid_periodic")
    
    def set_grid_periodic(self, grid_periodic, overwrite=None):
        """
        Set grid axes that need to be treated as periodic by xgcm.Grid.
        Axis that are not set periodic are non-periodic by default.  
        Note that this is opposite than xgcm, which sets periodic=True by default.
        
        Parameters
        ----------
        grid_periodic: list
            List of periodic axes.
            Available axis are {'X', 'Y', 'Z', 'time'}.
        overwrite: bool or None
            If None, raise error if grid_periodic has been previously set.
            If True, overwrite previous grid_periodic.  
            If False, combine with previous grid_periodic.
        """
        
        # Check parameters
        if not isinstance(grid_periodic, list):
            raise TypeError("`grid_periodic` must be str")
            
        list_axis  = ['X', 'Y', 'Z', 'time']
        for axis in grid_periodic:
            if axis not in list_axis:
                raise ValueError("[{}] not an OceanSpy axis. Available options are [{}]".format(axis, 
                                                                                            list_axis))
            
        # Set grid_periodic
        self = self._store_as_global_attr(name      = 'grid_periodic', 
                                          attr      = grid_periodic,
                                          overwrite = overwrite) 
        
        return self
    
    
    # ===========
    # METHODS
    # ===========
    def add_DataArray(self, da, overwrite=None):
        """
        Add DataArray to the oceandataset
        
        Parameters
        ----------
        da: xarray.DataArray
            xarray.DataArray to be added
        overwrite: bool or None
            If None, raise error if da already exists.
            If True, overwrite existing da.  
            If False, do not add existing da.
        """
        if not isinstance(da, _xr.DataArray):
            raise TypeError('`da` must be xarray.DataArray')
        
        if not isinstance(overwrite, (bool, type(None))):
            raise TypeError("`overwrite` must be bool or None")
        
        # Check if da exists
        if da.name in self.dataset.variables:  
            if overwrite is None:
                raise ValueError("[{}] already exists: `overwrite` must be bool ".format(da.name))
            elif overwrite is False:
                _warnings.warn(("\n[{}] already exists:"
                               "`da` will NOT be added to the oceandataset").format(da.name), stacklevel=2)
                return self
        
        # Only add if overwrite True, or da not in the oceandataset
        attrs   = self.dataset.attrs
        dataset = _xr.merge([self.dataset, da])
        dataset.attrs = attrs
        self = OceanDataset(dataset)
        
        return self
        
    def merge_Dataset(self, ds, overwrite=None):
        """
        Merge Dataset to the oceandataset
        
        Parameters
        ----------
        ds: xarray.Dataset
            xarray.Dataset to be merged
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.
            If True, overwrite existing xarray.DataArray.  
            If False, do not add existing xarray.DataArray.
        """
        if not isinstance(ds, _xr.Dataset):
            raise TypeError('`ds` must be xarray.Dataset')
        
        if not isinstance(overwrite, (bool, type(None))):
            raise TypeError("`overwrite` must be bool or None")
                               
        existing_da = [da for da in ds.variables if da in self.dataset.data_vars]          
        if len(existing_da)!=0:
            if overwrite is None:
                raise ValueError(("{} already exist: "
                                  "`overwrite` must be bool.").format(existing_da))
            elif overwrite is False:
                _warnings.warn(("\n{} already exist "
                                "and will NOT be added to the oceandataset.").format(existing_da), stacklevel=2)
                dataset = self.dataset
                ds      = ds.drop(existing_da)
            else:
                _warnings.warn(("\n{} already exist "
                                "and will be replaced.").format(existing_da), stacklevel=2)
                dataset = self.dataset.drop(existing_da)
        else:   dataset = self.dataset
        
        # Save attributes and merge dataset (raise error if dimension indexes are not matching)
        attrs   = dataset.attrs
        dataset = _xr.merge([dataset, ds]) 
        dataset.attrs = attrs
        self = OceanDataset(dataset)
        
        return self
    
    def set_coords(self, fillna=False, coords1Dfrom2D=False, coords2Dfrom1D=False, coordsUVfromG=False):
        """
        Set dataset coordinates: dimensions + 2D horizontal coordinates.        
        
        Parameters
        ----------
        fillna: bool
            If True, fill NaNs in 2D coordinates propagating backward and forward.  
        coords1Dfrom2D: bool
            If True, compute 1D coordinates from 2D coordinates (means). 
            Use with rectilinear grid only!
        coords2Dfrom1D: bool
            If True, compute 2D coordinates from 1D coordinates (brodacast). 
        coordsUVfromCG: bool
            If True, compute missing coords (U and V points) from G points.
        """
        
        # TODO: Add some check,
        #       like raise error if coords1Dfrom2D is used with curvilinare grids?
        
        # TODO: Add history and attributes to the new variables?
        
        # TODO: Most of this function can be easily replaced by xgcm.autogenerate,
        #       which is way better because it has been designed to work with any grid.
        #       As soon as a new xgcm version will be released, implemente autogenerate.
        
        # Check parameters
        if not isinstance(fillna, bool):
            raise TypeError('`fillna` must be bool')
        if not isinstance(coords1Dfrom2D, bool):
            raise TypeError('`coords1Dfrom2D` must be bool')
        if not isinstance(coordsUVfromG, bool):
            raise TypeError('`coordsUVfromG` must be bool')
        if coords1Dfrom2D and coords2Dfrom1D:
            raise TypeError('`coords1Dfrom2D` and `coords2Dfrom1D` can not be both True')
            
        # Copy because the dataset will change
        self = _copy.copy(self)

        # Coordinates are dimensions only
        self._ds = self._ds.reset_coords()
        
                        
        # Fill nans (e.g., because of exch2)
        if fillna:
            coords = ['YC', 'XC', 'YG', 'XG', 'YU', 'XU', 'YV', 'XV'] 
            dims   = ['X', 'Y', 'Xp1', 'Yp1', 'Xp1', 'Y', 'X', 'Yp1']
            
            for i, (coord, dim) in enumerate(zip(coords, dims)):
                if coord in self._ds.variables:
                    self._ds[coord] = self._ds[coord].ffill(dim).bfill(dim).persist()

        
        # Get U and V by rolling G
        if coordsUVfromG:
            for i, (point_pos, dim2roll) in enumerate(zip(['U', 'V'], ['Yp1', 'Xp1'])):
                for dim in ['Y', 'X']:
                    coord = self._ds[dim+'G'].rolling(**{dim2roll: 2}).mean().dropna(dim2roll)
                    coord = coord.drop(coord.coords).rename({dim2roll: dim2roll[0]})
                    self._ds[dim+point_pos] = coord
                    if 'units' in self._ds[dim+'G'].attrs:
                        self._ds[dim+point_pos].attrs['units'] = self._ds[dim+'G'].attrs['units']
                        
        # For cartesian grid we can use 1D coordinates
        if coords1Dfrom2D:
            # Take mean
            self._ds['Y']   = self._ds['YC'].mean('X', keep_attrs=True).persist()
            self._ds['X']   = self._ds['XC'].mean('Y', keep_attrs=True).persist()
            self._ds['Yp1'] = self._ds['YG'].mean('Xp1', keep_attrs=True).persist()
            self._ds['Xp1'] = self._ds['XG'].mean('Yp1', keep_attrs=True).persist()
        
        # Get 2D coordinates broadcasting 1D
        if coords2Dfrom1D:
            # Broadcast
            self._ds['YC'], self._ds['XC'] = _xr.broadcast(self._ds['Y'],   self._ds['X'])
            self._ds['YG'], self._ds['XG'] = _xr.broadcast(self._ds['Yp1'], self._ds['Xp1'])
            self._ds['YU'], self._ds['XU'] = _xr.broadcast(self._ds['Y'],   self._ds['Xp1'])
            self._ds['YV'], self._ds['XV'] = _xr.broadcast(self._ds['Yp1'], self._ds['X'])
            
            # Add units
            for i, (D2, D1) in enumerate(zip(['YC', 'XC', 'YG',  'XG',  'YU', 'XU',  'YV',  'XV'],
                                             ['Y',  'X',  'Yp1', 'Xp1', 'Y',  'Xp1', 'Yp1', 'X'])):
                if 'units' in self._ds[D1].attrs: self._ds[D2].attrs['units'] = self._ds[D1].attrs['units']
            
        # Set 2D coordinates
        self._ds = self._ds.set_coords(['YC', 'XC',
                                        'YG', 'XG',
                                        'YU', 'XU',
                                        'YV', 'XV'])
        return self
    
    def to_netcdf(self, path):
        """
        Write dataset contents to a netCDF file.
        
        Parameters
        ----------
        path: str
            Path to which to save this dataset.
        
        References
        ----------
        http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_netcdf.html
        """
        # Check parameters
        if not isinstance(path, str):
            raise TypeError('`path` must be str')
        
        from dask.diagnostics import ProgressBar as _ProgressBar
        
        print('Writing dataset to '+path)
        delayed_obj = self.dataset.to_netcdf(path, compute=False)
        with _ProgressBar():
            results = delayed_obj.compute()
        
    def create_tree(self, grid_pos = 'C'):
        """
        Create a scipy.spatial.cKDTree for quick nearest-neighbor lookup.
        
        Parameters
        -----------
        grid_pos: str
            Grid position. Option: {'C', 'G', 'U', 'V'}
            Reference grid: https://mitgcm.readthedocs.io/en/latest/algorithm/horiz-grid.html
        
        Returns
        -------
        tree: scipy.spatial.cKDTree
            Return tree that can be used to query a point.
            
        References
        ----------
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
        """
        
        # Check parameters
        grid_pos_list = ['C', 'G', 'U', 'V']
        if grid_pos not in grid_pos_list:
            raise ValueError("`grid_pos` must be on of {}. \nReference grid: https://mitgcm.readthedocs.io/en/latest/algorithm/horiz-grid.html".format(grid_pos_list))
            
        # Import package
        from scipy import spatial as _spatial
        
        # Convert if is not cartesian
        Y = self._ds['Y'+grid_pos]
        X = self._ds['X'+grid_pos]
        R = self.parameters['rSphere']
        if R:
            x, y, z = _utils.spherical2cartesian(Y = Y, X = X, R = R)
        else:
            x = X; y = Y; z = _xr.zeros_like(Y)

        # Stack
        x_stack = x.stack(points=x.dims).values
        y_stack = y.stack(points=y.dims).values
        z_stack = z.stack(points=z.dims).values
        
        # Construct KD-tree
        tree = _spatial.cKDTree(_np.column_stack((x_stack, y_stack, z_stack)))
    
        return tree
        
    def _store_as_global_attr(self, name, attr, overwrite):
        """
        Store an OceanSpy attribute as dataset global attribute.
        
        Parameters
        ----------
        name: str
            Name of the attribute. Attribute will be stored as 'OceanSpy_'+name.
        attr: str or dict
            Attribute to store
        overwrite: bool or None
            If None, raise error if attr has been previously set.
            If True, overwrite previous attributes.  
            If False, combine with previous attributes.
        """
        
        # Check parameters
        if not isinstance(name, str):
            raise TypeError("`name` must be str")
        if not isinstance(attr, (str, dict, list)):
            raise TypeError("`attr` must be str, dict, or list")
        if not isinstance(overwrite, (bool, type(None))):
            raise TypeError("`overwrite` must be bool or None")
        
        # Attribute name
        name = 'OceanSpy_'+name
        
        if overwrite is None and name in self._ds.attrs:
            raise ValueError("[{}] has been previously set: `overwrite` must be bool ".format(name))
            
        # Copy because attributes are added to _ds
        self = _copy.copy(self)
        
        # Store
        if not overwrite and name in self._ds.attrs:
            prev_attr = self._ds.attrs[name]
            if prev_attr[0] == "{" and prev_attr[-1] == "}":
                attr = {**eval(prev_attr), **attr}
            elif prev_attr[0] == "[" and prev_attr[-1] == "]":
                attr = list(set(eval(prev_attr) + attr))
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
            Name of the attribute. Attribute will be read from 'OceanSpy_'+name.
            
        Returns
        -------
        attr: str or dict
            Attribute that has been read
        """
        
        if not isinstance(name, str):
            raise TypeError("`name` must be str")
        
        # Attribute name
        name = 'OceanSpy_'+name
        
        # Check if attributes exists 
        if name not in self._ds.attrs:
            return None
        
        # Read attribute
        attr = self._ds.attrs[name]
        if (attr[0]=='{' and attr[-1]=='}') or (attr[0]=='[' and attr[-1]==']'):
            attr = eval(attr)
        
        return attr
      
    # ===========
    # SHORTCUTS
    # ===========
    
    # ------------
    # Subsample
    def cutout(self, **kwargs):
        """
        Shortcut for subsample.cutout
        
        Parameters
        ----------
        **kwargs: 
            Arguments for subsample.cutout
            
        See Also
        --------
        subsample.cutout
        """
        
        self = _subsample.cutout(od = self, **kwargs)
        
        return self
    
    def mooring_array(self, **kwargs):
        """
        Shortcut for subsample.mooring_array
        
        Parameters
        ----------
        **kwargs: 
            Arguments for subsample.mooring_array
            
        See Also
        --------
        subsample.mooring_array
        """
        
        self = _subsample.mooring_array(od = self, **kwargs)
        
        return self
    
    def survey_stations(self, **kwargs):
        """
        Shortcut for subsample.survey_stations
        
        Parameters
        ----------
        **kwargs: 
            Arguments for subsample.survey_stations
            
        See Also
        --------
        subsample.survey_stations
        """
        
        self = _subsample.survey_stations(od = self, **kwargs)
        
        return self
    
    def particle_properties(self, **kwargs):
        """
        Shortcut for subsample.particle_properties
        
        Parameters
        ----------
        **kwargs: 
            Arguments for subsample.particle_properties
            
        See Also
        --------
        subsample.particle_properties
        """
        
        self = _subsample.particle_properties(od = self, **kwargs)
        
        return self

    
    # ------------
    # Compute

    def merge_gradient(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.gradient and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.gradient
            
        See Also
        --------
        compute.gradient
        """
        
        self = self.merge_Dataset(_compute.gradient(self, **kwargs), overwrite)
        
        return self
    
    def merge_divergence(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.divergence and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.divergence
            
        See Also
        --------
        compute.divergence
        """
        
        self = self.merge_Dataset(_compute.divergence(self, **kwargs), overwrite)
        
        return self
    
    def merge_curl(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.curl and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.curl
            
        See Also
        --------
        compute.curl
        """
        
        self = self.merge_Dataset(_compute.curl(self, **kwargs), overwrite)
        
        return self
    
    def merge_laplacian(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.laplacian and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.laplacian
            
        See Also
        --------
        compute.laplacian
        """
        
        self = self.merge_Dataset(_compute.laplacian(self, **kwargs), overwrite)
        
        return self
    
    def merge_volume_weighted_mean(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.volume_weighted_mean and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.volume_weighted_mean
            
        See Also
        --------
        compute.volume_weighted_mean
        """
        
        self = self.merge_Dataset(_compute.volume_weighted_mean(self, **kwargs), overwrite)
        
        return self
    
    
    def merge_potential_density_anomaly(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.potential_density_anomaly and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.potential_density_anomaly 
            
        See Also
        --------
        compute.potential_density_anomaly
        """
        
        self = self.merge_Dataset(_compute.potential_density_anomaly(self, **kwargs), overwrite)
        
        return self
    
    def merge_Brunt_Vaisala_frequency(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.Brunt_Vaisala_frequency and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.Brunt_Vaisala_frequency
            
        See Also
        --------
        compute.Brunt_Vaisala_frequency
        """
        
        self = self.merge_Dataset(_compute.Brunt_Vaisala_frequency(self, **kwargs), overwrite)
        
        return self
    
    def merge_vertical_relative_vorticity(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.vertical_relative_vorticity and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.vertical_relative_vorticity 
            
        See Also
        --------
        compute.vertical_relative_vorticity
        """
        
        self = self.merge_Dataset(_compute.vertical_relative_vorticity(self, **kwargs), overwrite)
        
        return self
    
    def merge_relative_vorticity(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.relative_vorticity and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.relative_vorticity
            
        See Also
        --------
        compute.relative_vorticity
        """
        
        self = self.merge_Dataset(_compute.relative_vorticity(self, **kwargs), overwrite)
        
        return self
    
    def merge_kinetic_energy(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.kinetic_energy and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.kinetic_energy
            
        See Also
        --------
        compute.kinetic_energy
        """
        
        self = self.merge_Dataset(_compute.kinetic_energy(self, **kwargs), overwrite)
        
        return self
    
    def merge_eddy_kinetic_energy(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.eddy_kinetic_energy and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.eddy_kinetic_energy 
            
        See Also
        --------
        compute.eddy_kinetic_energy
        """
        
        self = self.merge_Dataset(_compute.eddy_kinetic_energy(self, **kwargs), overwrite)
        
        return self
    
    def merge_horizontal_divergence_velocity(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.horizontal_divergence_velocity and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.  
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.horizontal_divergence_velocity
            
        See Also
        --------
        compute.horizontal_divergence_velocity
        """
        
        self = self.merge_Dataset(_compute.horizontal_divergence_velocity(self, **kwargs), overwrite)
        
        return self
    
    def merge_shear_strain(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.shear_strain and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.   
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.shear_strain
            
        See Also
        --------
        compute.shear_strain
        """
        
        self = self.merge_Dataset(_compute.shear_strain(self, **kwargs), overwrite)
        
        return self
    
    def merge_normal_strain(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.normal_strain and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.normal_strain
            
        See Also
        --------
        compute.normal_strain
        """
        
        self = self.merge_Dataset(_compute.normal_strain(self, **kwargs), overwrite)
        
        return self
    
    def merge_Okubo_Weiss_parameter(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.Okubo_Weiss_parameter and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.Okubo_Weiss_parameter
            
        See Also
        --------
        compute.Okubo_Weiss_parameter
        """
        
        self = self.merge_Dataset(_compute.Okubo_Weiss_parameter(self, **kwargs), overwrite)
        
        return self
    
    def merge_Ertel_potential_vorticity(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.Ertel_potential_vorticity and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.   
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.Ertel_potential_vorticity
            
        See Also
        --------
        compute.Ertel_potential_vorticity
        """
        
        self = self.merge_Dataset(_compute.Ertel_potential_vorticity(self, **kwargs), overwrite)
        
        return self
    
    def merge_mooring_horizontal_volume_transport(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.mooring_horizontal_volume_transport and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.   
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.mooring_horizontal_volume_transport 
            
        See Also
        --------
        compute.mooring_horizontal_volume_transport
        """
        
        self = self.merge_Dataset(_compute.mooring_horizontal_volume_transport(self, **kwargs), overwrite)
        
        return self
    
    def merge_heat_budget(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.heat_budget and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.   
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.heat_budget
            
        See Also
        --------
        compute.heat_budget
        """
        
        self = self.merge_Dataset(_compute.heat_budget(self, **kwargs), overwrite)
        
        return self
    
    def merge_salt_budget(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.salt_budget and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.salt_budget 
            
        See Also
        --------
        compute.salt_budget
        """
        
        self = self.merge_Dataset(_compute.salt_budget(self, **kwargs), overwrite)
        
        return self
    
    def merge_geographical_aligned_velocities(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.geographical_aligned_velocities and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.geographical_aligned_velocities
            
        See Also
        --------
        compute.geographical_aligned_velocities
        """
        
        self = self.merge_Dataset(_compute.geographical_aligned_velocities(self, **kwargs), overwrite)
        
        return self
        
    def merge_survey_aligned_velocities(self, overwrite=None, **kwargs):
        """
        Shortcut for compute.survey_aligned_velocities and OceanDataset.merge_Dataset.
        
        Parameters
        ----------
        overwrite: bool or None
            If None, raise error if any xarray.DataArray already exists.  
            If True, overwrite existing xarray.DataArray.    
            If False, do not add existing xarray.DataArray.  
        **kwargs: 
            Keyword arguments for compute.survey_aligned_velocities
            
        See Also
        --------
        compute.survey_aligned_velocities
        """
        
        self = self.merge_Dataset(_compute.survey_aligned_velocities(self, **kwargs), overwrite)
        
        return self
        

def _create_grid(dataset, coords, periodic):

    # TODO: add option to infer axis using comodo since we're currently cleaning up comodo attributes by default
    
    # Clean up comodo (currently force user to specify axis using set_coords).      
    for dim in dataset.dims:
        dataset[dim].attrs.pop('axis', None)
        dataset[dim].attrs.pop('c_grid_axis_shift', None)

    # Add comodo attributes. 
    # We won't need this step in the future because future versions of xgcm will allow to pass coords in Grid.
    if coords:
        for axis in coords:
            for dim in coords[axis]:
                shift = coords[axis][dim]
                dataset[dim].attrs['axis'] = axis
                if shift:
                    dataset[dim].attrs['c_grid_axis_shift'] = str(shift)
    
    # Create grid
    grid = _xgcm.Grid(dataset, periodic = periodic)

    return grid



