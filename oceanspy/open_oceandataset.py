"""
Open OceanDataset objects.
"""

# Instructions for developers:
# 1. All functions in this module must return an OceanDataset.
# 2. Add new functions in docs/api.rst

# Import oceanspy dependencies (private)
import xarray as _xr
import warnings as _warnings
import yaml as _yaml

# Import from oceanspy (private)
from ._oceandataset import OceanDataset as _OceanDataset
import oceanspy as _ospy
from ._ospy_utils import _check_instance

# Import extra modules (private)
try:
    import xmitgcm as _xmitgcm
except ImportError:  # pragma: no cover
    pass
try:
    import intake as _intake
except ImportError:  # pragma: no cover
    pass


def from_netcdf(path):
    """
    Load an OceanDataset from a netcdf file.

    Parameters
    ----------
    path: str
        Path from which to read.

    Returns
    -------
    od: OceanDataset
    """

    # Check parameters
    _check_instance({'path': path}, 'str')

    # Open
    print('Opening dataset from [{}].'.format(path))
    ds = _xr.open_dataset(path)

    # Put back coordinates attribute that to_netcdf didn't like
    for var in ds.variables:
        attrs = ds[var].attrs
        coordinates = attrs.pop('_coordinates', None)
        ds[var].attrs = attrs
        if coordinates is not None:
            ds[var].attrs['coordinates'] = coordinates

    # Create and return od
    od = _OceanDataset(ds)
    return od


def from_catalog(name, catalog_url=None):
    """
    Import oceandataset using a yaml catalog.
    Catalogs with suffix xarray.yaml use :py:mod:`intake-xarray`,
    while catalogs with suffix xmitgcm.yaml use
    :py:func:`xmitgcm.open_mdsdataset`.

    Parameters
    ----------
    name: str
        Name of the oceandataset to open.
    catalog_url: str or None
        Path from which to read the catalog.
        If None, use SciServer's catalogs.

    References
    ----------
    | intake-xarray: https://github.com/intake/intake-xarray
    | xmitgcm: https://xmitgcm.readthedocs.io/en/stable/usage.html
    """

    # Check parameters
    if catalog_url is None:  # pragma: no cover
        SciList = ['get_started',
                   'EGshelfIIseas2km_ERAI_6H',
                   'EGshelfIIseas2km_ERAI_1D',
                   'EGshelfIIseas2km_ASR_full',
                   'EGshelfIIseas2km_ASR_crop',
                   'Arctic_Control',
                   'KangerFjord',
                   'EGshelfSJsec500m_3H_hydro',
                   'EGshelfSJsec500m_6H_hydro',
                   'EGshelfSJsec500m_3H_NONhydro',
                   'EGshelfSJsec500m_6H_NONhydro',
                   'xmitgcm_snap']
        if name not in SciList:
            raise ValueError('[{}] is not available on SciServer.'
                             ' Here is a list of available oceandatasets: {}.'
                             ''.format(name, SciList))
    else:
        _check_instance({'catalog_url': catalog_url}, 'str')

    # Message
    print('Opening {}.'.format(name))

    # Assign a xarray url if it's None
    if catalog_url is None:  # pragma: no cover
        url = _ospy.__path__[0] + '/Catalogs/catalog_xarray.yaml'
    else:
        suffixes = ['xarray.yaml', 'xmitgcm.yaml']
        if not any([suf in catalog_url for suf in suffixes]):
            raise ValueError('`catalog_url`'
                             ' must have one of the following suffixes:'
                             ' {}'.format(suffixes))
        else:
            url = catalog_url

    # Xarray catalog
    if 'xarray.yaml' in url:
        cat = _intake.Catalog(url)
        entries = [entry for entry in cat if name in entry]
    else:
        pass

    # Assign a xmitgcm url
    if catalog_url is None and len(entries) == 0:  # pragma: no cover
        url = _ospy.__path__[0] + '/Catalogs/catalog_xmitgcm.yaml'
    else:
        pass

    # Xmitgcm catalog
    if 'xmitgcm.yaml' in url:
        with open(url) as f:
            cat = _yaml.safe_load(f)
        entries = [entry for entry in cat if name in entry]

    # Error if not available
    if len(entries) == 0:
        raise ValueError('[{}] is not i the catalog.'.format(name))

    # Store all dataset
    datasets = []
    chunks = {}
    metadata = {}
    for entry in entries:
        if 'xmitgcm.yaml' in url:
            # Pop args and metadata
            args = cat[entry].pop('args')
            mtdt = cat[entry].pop('metadata', None)

            # If iter is a string, need to be evaluated (likely range)
            iters = args.pop('iters', None)
            if isinstance(iters, str) and 'range' in iters:
                iters = eval(iters)
            if iters is not None:
                args['iters'] = iters

            # Create ds
            with _warnings.catch_warnings():
                # Not sure why, Marcello's print a lot of warnings, Neil no.
                # TODO: this need to be addressed
                _warnings.simplefilter("ignore")
                ds = _xmitgcm.open_mdsdataset(**args)
        else:
            # Use intake-xarray

            # Pop metadata
            mtdt = cat[entry].metadata

            # Create ds
            ds = cat[entry].to_dask()

        # Rename
        rename = mtdt.pop('rename', None)
        ds = ds.rename(rename)

        # Fix Z dimensions (Zmd, ...)
        default_Zs = ['Zp1', 'Zu', 'Zl', 'Z']
        # Make sure they're sorted with decreasing letter number
        default_Zs = sorted(default_Zs, key=len, reverse=True)
        for Zdim in default_Zs:  # pragma: no cover
            for dim, size in ds.sizes.items():
                if dim in default_Zs:
                    continue
                elif Zdim in dim:
                    if size == 1:
                        ds = ds.squeeze(dim)
                    else:
                        if Zdim in ds.dims:
                            ds = ds.rename({Zdim: 'tmp'})
                            ds = ds.rename({'tmp': Zdim, dim: Zdim})
                        else:
                            ds = ds.rename({dim: Zdim})

        # Original output
        or_out = mtdt.pop('original_output', None)
        if or_out is not None:
            for var in ds.data_vars:
                ds[var].attrs['original_output'] = or_out

        # Select
        isel = mtdt.pop('isel', None)
        if isel is not None:
            isel = {key: eval(value) for key, value in isel.items()}
            ds = ds.isel(isel)

        # Append
        datasets.append(ds)

        # Metadata
        metadata = {**metadata, **mtdt}

    # Merge
    ds = _xr.merge(datasets)

    # Consistent chunking
    chunks = {}
    for var in ds.data_vars:
        if ds[var].chunks is not None:
            for i, dim in enumerate(ds[var].dims):
                chunk = ds[var].chunks[i]
                if dim not in chunks or len(chunks[dim]) < len(chunk):
                    chunks[dim] = chunk
    ds = ds.chunk(chunks)

    # Initialize OceanDataset
    od = _OceanDataset(ds)

    # Shift averages
    shift_averages = metadata.pop('shift_averages', None)
    if shift_averages is not None:
        od = od.shift_averages(**shift_averages)

    # Manipulate coordinates
    manipulate_coords = metadata.pop('manipulate_coords', None)
    if manipulate_coords is not None:
        od = od.manipulate_coords(**manipulate_coords)

    # Set grid coordinates
    grid_coords = metadata.pop('grid_coords', None)
    if grid_coords is not None:
        od = od.set_grid_coords(**grid_coords)

    # Set OceanSpy stuff
    for var in ['aliases', 'parameters', 'name', 'description', 'projection']:
        val = metadata.pop(var, None)
        if val is not None:
            od = eval('od.set_{}(val)'.format(var))

    # Print message
    print(od.description)

    return od
