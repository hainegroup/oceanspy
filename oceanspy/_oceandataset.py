import xarray   as _xr
import copy     as _copy
import xgcm     as _xgcm
import numpy    as _np
import warnings as _warnings
import sys      as _sys
from . import compute   as _compute
from . import plot      as _plot
from . import animate   as _animate
from . import utils     as _utils

from . subsample import _subsampleMethdos
from . compute   import _computeMethdos
from . plot      import _plotMethdos
from . animate   import _animateMethdos

try: import cartopy.crs as _ccrs
except ImportError: pass
try: from scipy import spatial as _spatial
except ImportError: pass

# TODO: add more xgcm options. E.g., default boundary method.
# TODO: add attributes to new coordinates (XU, XV, ...)
# TODO: implement xgcm autogenerate in _set_coords, set_grid_coords, set_coords when released
# TODO: _create_grid will be useless with the future release of xgcm. We will pass dictionary in xgcm.Grid,
#       and we can have the option of usining comodo attributes (currently cleaned up so switched off)

class OceanDataset:
    """
    OceanDataset combines a xarray.Dataset with other objects used by OceanSpy (e.g., xgcm.Grid).

    Additional objects are attached to the xarray.Dataset as global attributes.
    
    OceanDataset adds, reads, and decodes dataset global attributes.
    """
    
    def __init__(self, 
                 dataset):
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
        if self.dataset is not None:
            main_info.append("   .dataset: %s"    % self.dataset.__repr__()[self.dataset.__repr__().find('<'):
                                                                            self.dataset.__repr__().find('>')+1]) 
        if self.grid is not None:
            main_info.append("   .grid: %s"       % self.grid.__repr__()[self.grid.__repr__().find('<'):
                                                                         self.grid.__repr__().find('>')+1]) 
        if self.projection is not None:
            main_info.append("   .projection: %s" % self.projection.__repr__()[self.projection.__repr__().find('<'):
                                                                               self.projection.__repr__().find('>')+1]) 
            
        more_info = ['\n\nMore attributes:']
        if self.name:
            more_info.append("   .name: %s" % self.name)
        if self.description:
            more_info.append("   .description: %s" % self.description)
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
            
    # ==================================
    # IMPORT (used by open_oceandataset)
    # ==================================
    
    def _shift_averages(self):
        """
        Shift average variables to time_midp.
        Average variables must have attribute original_output = 'average'.
        """
        for var in self._ds.data_vars:
            original_output = self._ds[var].attrs.pop('original_output', None)
            if original_output == 'average':
                self._ds[var] = self._ds[var].drop('time').isel(time=slice(1, None)).rename({'time': 'time_midp'})
            if original_output is not None:
                self._ds[var].attrs['original_output'] = original_output
        return self
    
    def _set_coords(self, fillna=False, coords1Dfrom2D=False, coords2Dfrom1D=False, coordsUVfromG=False):
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
    
    def import_MITgcm_rect_nc(self, shift_averages = True):
        """
        Set coordinates of a dataset from a MITgcm run with rectilinear grid and data stored in NetCDF format.
        Open and concatentate dataset before running this function.
        
        Parameters
        ----------
        shift_averages: bool
            If True, shift average variable to time_midp. 
            Average variables must have attribute original_output = 'average'
        """
        
        # Check parameters
        if not isinstance(shift_averages, bool):
            raise TypeError('`shift_averages` must be bool')
                
        # Shift averages
        if shift_averages is True:
            self = self._shift_averages()
        
        # Set coordinates
        self = self._set_coords(fillna=True, coords1Dfrom2D=True)
        grid_coords = {'Y'    : {'Y': None, 'Yp1': 0.5},
                       'X'    : {'X': None, 'Xp1': 0.5},
                       'Z'    : {'Z': None, 'Zp1': 0.5, 'Zu': 0.5, 'Zl': -0.5},
                       'time' : {'time': -0.5}}
        self = self.set_grid_coords(grid_coords = grid_coords, add_midp=True)
        
        return self
    
    def import_MITgcm_rect_bin(self, shift_averages = True):
        """
        Set coordinates of a dataset from a MITgcm run with rectilinear grid and data stored in bin format.
        Open and concatentate dataset before running this function.
        
        Parameters
        ----------
        shift_averages: bool
            If True, shift average variable to time_midp. 
            Average variables must have attribute original_output = 'average'
        """
        
        # Check parameters
        if not isinstance(shift_averages, bool):
            raise TypeError('`shift_averages` must be bool')
                
        # Shift averages
        if shift_averages is True:
            self = self._shift_averages()
        # Set coordinates
        self = self._set_coords(coords2Dfrom1D=True)
        grid_coords = {'Y'    : {'Y': None, 'Yp1': 0.5},
                       'X'    : {'X': None, 'Xp1': 0.5},
                       'Z'    : {'Z': None, 'Zp1': 0.5, 'Zu': 0.5, 'Zl': -0.5},
                       'time' : {'time': -0.5}}
        self = self.set_grid_coords(grid_coords = grid_coords, add_midp=True)
        
        return self
    
    def import_MITgcm_curv_nc(self, shift_averages = True):
        """
        Set coordinates of a dataset from a MITgcm run with curvilinear grid and data stored in NetCDF format.
        Open and concatentate dataset before running this function.
        
        Parameters
        ----------
        shift_averages: bool
            If True, shift average variable to time_midp. 
            Average variables must have attribute original_output = 'average'
        """
        
        # Check parameters
        if not isinstance(shift_averages, bool):
            raise TypeError('`shift_averages` must be bool')
                
        # Shift averages
        if shift_averages is True:
            self = self._shift_averages()
        # Set coordinates
        self = self._set_coords(coordsUVfromG=True)
        grid_coords = {'Y'    : {'Y': None, 'Yp1': 0.5},
                       'X'    : {'X': None, 'Xp1': 0.5},
                       'Z'    : {'Z': None, 'Zp1': 0.5, 'Zu': 0.5, 'Zl': -0.5},
                       'time' : {'time': -0.5}}
        self = self.set_grid_coords(grid_coords = grid_coords, add_midp=True)
        
        return self
        
        
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
        
        raise AttributeError(_setter_error_message('name'))
    
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
        
        raise AttributeError(_setter_error_message('description'))
    
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
        
        raise AttributeError("Set a new dataset using `oceanspy.OceanDataset(dataset)`")
    
    # -------------------
    # aliases
    # -------------------
    
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
        
        raise AttributeError(_setter_error_message('aliases'))
    
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
        if self._aliases_flipped:                    
            aliases = {custom: ospy for custom, ospy in self._aliases_flipped.items() 
                       if custom in self._ds.variables 
                       or custom in self._ds.dims}
            self._ds = self._ds.rename(aliases)
            
        return self
    
    # -------------------
    # parameters
    # -------------------
    
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
        
        raise AttributeError(_setter_error_message('parameters'))
    
    def set_parameters(self, parameters):
        """
        Set model parameters used by OceanSpy (see oceanspy.DEFAULT_PARAMETERS)
        
        Parameters
        ----------
        parameters: dict
            {'parameter_name': parameter_value}
        """
        from oceanspy import DEFAULT_PARAMETERS, AVAILABLE_PARAMETERS, TYPE_PARAMETERS
        
        # Check parameters
        if not isinstance(parameters, dict):
            raise TypeError("`parameters` must be dict")
                
        # Check parameters
        warn_params = []
        for key, value in parameters.items():
            if key not in DEFAULT_PARAMETERS.keys(): warn_params = warn_params + [key]
            else:
                if not isinstance(value, TYPE_PARAMETERS[key]):
                    raise TypeError("Invalid [{}]. Check oceanspy.TYPE_PARAMETERS".format(key))
                if key in AVAILABLE_PARAMETERS.keys() and value not in AVAILABLE_PARAMETERS[key]:
                    raise ValueError("Requested [{}] not available. Check oceanspy.AVAILABLE_PARAMETERS".format(key))
                    
        if len(warn_params)!=0:
            _warnings.warn(("{} are not OceanSpy parameters").format(warn_params), stacklevel=2)
        
        # Set parameters
        self = self._store_as_global_attr(name      = 'parameters', 
                                          attr      = parameters,
                                          overwrite = True) 
        
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
        
        raise AttributeError(_setter_error_message('grid_coords'))
    
    
    def set_grid_coords(self, grid_coords, add_midp=False, overwrite=None):
        """
        Set grid coordinates used by xgcm.Grid (see oceanspy.OCEANSPY_AXES).
        
        Parameters
        ----------
        grid_coords: str
            Grid coordinates used by xgcm.Grid.  
            Keys are axis, and values are dict with key=dim and value=c_grid_axis_shift.
            Available c_grid_axis_shift are {0.5, None, -0.5}
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
        
        if not isinstance(add_midp, (bool, type(None))):
            raise TypeError("`add_midp` must be bool")
            
        # Check axes
        _check_oceanspy_axes(list(grid_coords.keys()))
            
        # Check shifts
        list_shift = [0.5, None, -0.5]
        for axis in grid_coords:
            if grid_coords[axis] is None: continue
            elif not isinstance(grid_coords[axis], dict):
                example_grid_coords = {'Y'    : {'Y': None, 'Yp1': 0.5}}
                raise TypeError("Invalid grid_coords. grid_coords example: {}".format(example_grid_coords))
            else:
                for dim in grid_coords[axis]:
                    if grid_coords[axis][dim] not in list_shift:
                        raise ValueError("[{}] not a valid c_grid_axis_shift."
                                         " Available options are {}".format(grid_coords[axis][dim],
                                                                            list_shift))
        
            
        # Set grid_coords
        self = self._store_as_global_attr(name      = 'grid_coords', 
                                          attr      =  grid_coords,
                                          overwrite =  overwrite) 
        
        if add_midp:
            grid_coords = {}
            for axis in self.grid_coords:
                if len(self.grid_coords[axis])==1 and list(self.grid_coords[axis].values())[0] is not None:
                    
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
        
        raise AttributeError(_setter_error_message('grid_periodic'))
    
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
            raise TypeError("`grid_periodic` must be list")
            
        # Check axes
        _check_oceanspy_axes(grid_periodic)
            
        # Set grid_periodic
        self = self._store_as_global_attr(name      = 'grid_periodic', 
                                          attr      = grid_periodic,
                                          overwrite = overwrite) 
        
        return self
    
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

        raise AttributeError("Set a new grid using .set_grid_coords and .set_periodic")
    
    @_grid.setter
    def _grid(self, grid):
        """
        Inhibit setter
        """

        raise AttributeError("Set a new _grid using .set_grid_coords and .set_periodic")
    
    
    # -------------------
    # projection
    # -------------------
    @property
    def projection(self):
        """
        Projection of the OceanDataset.
        """
        
        projection = self._read_from_global_attr('projection')
        if projection:
            if projection=='None':
                projection = eval(projection)
            else:
                if 'cartopy' not in _sys.modules:
                    _warnings.warn(("cartopy is not available, so projection is None").format(da.name), stacklevel=2)
                    projection = None
                else:
                    projection = eval('_ccrs.{}'.format(projection))
            
        return projection

    @projection.setter
    def projection(self, projection):
        """
        Inhibit setter
        """
        
        raise AttributeError(_setter_error_message('projection'))
    
    def set_projection(self, projection, **kwargs):
        """
        Projection of the OceanDataset.
        
        Parameters
        ----------
        projection: str
            cartopy projection of the OceanDataset
        **kwargs:
            Keyword arguments used by cartopy
            E.g., central_longitude=0.0 for PlateCarree
        References
        ----------
        https://scitools.org.uk/cartopy/docs/latest/crs/projections.html
        """
        
        # Check parameters
        if not isinstance(projection, (type(None), str)):
            raise TypeError("`projection` must be str or None")
        
        if projection is not None:
            if not hasattr(_ccrs, projection):
                raise TypeError("{} is not a cartopy projection".format(projection))
            projection = '{}(**{})'.format(projection, kwargs)
        else: 
            projection = str(projection) 

        # Set projection
        self = self._store_as_global_attr(name      = 'projection', 
                                          attr      = projection,
                                          overwrite = True) 

        return self
    
    
    
    # ===========
    # METHODS
    # ===========
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
        
        if 'scipy' not in _sys.modules:
            raise ImportError("cKDTree can not be created because scipy is not installed")
        
        # Check parameters
        if not isinstance(grid_pos, str):
            raise TypeError('`grid_pos` must be str')
        grid_pos_list = ['C', 'G', 'U', 'V']
        if grid_pos not in grid_pos_list:
            raise ValueError(("`grid_pos` must be on of {}:"
                              "\nhttps://mitgcm.readthedocs.io/en/latest/algorithm/horiz-grid.html").format(grid_pos_list))
        
        # Convert if is not cartesian
        Y = self._ds['Y'+grid_pos]
        X = self._ds['X'+grid_pos]
        R = self.parameters['rSphere']
        if R: x, y, z = _utils.spherical2cartesian(Y = Y, X = X, R = R)
        else: x = X; y = Y; z = _xr.zeros_like(Y)

        # Stack
        x_stack = x.stack(points=x.dims).values
        y_stack = y.stack(points=y.dims).values
        z_stack = z.stack(points=z.dims).values
        
        # Construct KD-tree
        tree = _spatial.cKDTree(_np.column_stack((x_stack, y_stack, z_stack)))
    
        return tree
    
    
    
    def merge_into_oceandataset(self, obj, overwrite=False):
        """
        Merge a dataset or DataArray into the oceandataset
        
        Parameters
        ----------
        obj: xarray.DataArray or xarray.Dataset
            xarray object to merge
        overwrite: bool or None
            If True, overwrite existing DataArrays with same name.  
            If False, use xarray.merge
        """
        
        # Check and make dataset
        if not isinstance(obj, (_xr.DataArray, _xr.Dataset)):
            raise TypeError('`obj` must be xarray.DataArray or xarray.Dataset')
        
        obj = obj.drop(obj.coords)
        if isinstance(obj, _xr.DataArray):
            if obj.name is None:
                raise ValueError("xarray.DataArray doesn't have a name. Set it using da.rename()")
            else:
                obj = obj.to_dataset()
                
        if not isinstance(overwrite, bool):
            raise TypeError("`overwrite` must be bool")
        
        # Merge
        dataset  = self.dataset
        var2drop = [var for var in obj.variables if var in dataset]     
        if overwrite is False:
            obj = obj.drop(var2drop)
            if len(var2drop)!=0: _warnings.warn('{} will not be merged.'
                                                '\nSet `overwrite=True` if you wish otherwise.'.format(var2drop), stacklevel=2)
        else:
            if len(var2drop)!=0: _warnings.warn('{} will be overwritten.'.format(var2drop), stacklevel=2)
        for var in obj.data_vars:
            dataset[var] = obj[var]
        
        return OceanDataset(dataset)
    
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
        
        # to_netcdf doesn't like coordinates attribute
        dataset = self.dataset
        for var in dataset.variables:
            attrs = dataset[var].attrs
            coordinates = attrs.pop('coordinates', None)
            dataset[var].attrs = attrs
            if coordinates is not None: dataset[var].attrs['_coordinates'] = coordinates
        
        print('Writing dataset to '+path)
        delayed_obj = dataset.to_netcdf(path, compute=False)
        with _ProgressBar():
            results = delayed_obj.compute()
        
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
            raise ValueError("[{}] has been previously set: "
                             "`overwrite` must be bool".format(name.replace("OceanSpy_", "")))
            
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
    
    @property
    def subsample(self):
        """
        Access subsampling functions.
        
        Examples
        --------
        >>> od = ospy.open_oceandataset.get_started()
        >>> od.subsample.cutout(ZRange=[0, -100], varList=['Temp'])
        """
        
        return _subsampleMethdos(self)

    @property
    def compute(self):
        """
        Access computing functions, and merge the computed dataset into the oceandataset.
        Set overwrite=True to overwrite DataArrays already existing in the oceandataset.
        
        Examples
        --------
        >>> od = ospy.open_oceandataset.get_started()
        >>> od.compute.gradient(varNameList='Temp', overwrite=True)
        """
        
        return _computeMethdos(self)
    
    @property
    def plot(self):
        """
        Access plotting functions.
        
        Examples
        --------
        >>> od = ospy.open_oceandataset.get_started()
        >>> od.plot.TS_diagram(meanAxes=['time', 'Z'], cutout_kwargs={'ZRange': [0, -100]})
        """
        
        return _plotMethdos(self)
    
    @property
    def animate(self):
        """
        Access animating functions.
        
        Examples
        --------
        >>> od = ospy.open_oceandataset.get_started()
        >>> od.animate.TS_diagram(meanAxes=['time', 'Z'], cutout_kwargs={'ZRange': [0, -100]})
        """
        
        return _animateMethdos(self)

        
    

        
        
        
        
        
        
        
        
        
        
        
# ERROR HANDLING
def _check_oceanspy_axes(axes2check):
    """
    Check axes
    """
    from oceanspy import OCEANSPY_AXES
    
    for axis in axes2check:
        if axis not in OCEANSPY_AXES:
            raise ValueError(_wrong_axes_error_message(axes2check))
    
def _wrong_axes_error_message(axes2check):
    from oceanspy import OCEANSPY_AXES
    return ("{} contains non-valid axes."
            " OceanSpy axes are: {}").format(axes2check, OCEANSPY_AXES)

def _setter_error_message(attribute_name):
    """
    Use the same error message for attributes
    """
    return "Set new `{}` using .set_{}".format(attribute_name, attribute_name)    
    
    
# USEFUL FUNCTIONS
def _create_grid(dataset, coords, periodic):
    
    # Clean up comodo (currently force user to specify axis using set_coords).      
    for dim in dataset.dims:
        dataset[dim].attrs.pop('axis', None)
        dataset[dim].attrs.pop('c_grid_axis_shift', None)

    # Add comodo attributes. 
    # We won't need this step in the future because future versions of xgcm will allow to pass coords in Grid.
    warn_dims = []
    if coords:
        for axis in coords:
            for dim in coords[axis]:
                shift = coords[axis][dim]
                if dim in dataset.dims:
                    dataset[dim].attrs['axis'] = axis
                    if shift:
                        dataset[dim].attrs['c_grid_axis_shift'] = str(shift)
                else:
                    warn_dims = warn_dims + [dim]
    if len(warn_dims)!=0: 
        _warnings.warn('{} are not dimensions of the dataset and will be omitted'.format(warn_dims), stacklevel=2)
                    
    # Create grid
    grid = _xgcm.Grid(dataset, periodic = periodic)
    if len(grid.axes)==0: 
        grid = None
        
    return grid


