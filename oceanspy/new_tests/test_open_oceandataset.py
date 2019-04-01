# Import modules
import pytest
import subprocess
import numpy as np
from os import listdir
from shutil import copyfile

# Import oceanspy
from oceanspy.open_oceandataset import from_netcdf, from_catalog

# Directory
Datadir = './oceanspy/new_tests/Data/'

# Download xmitgcm test
commands = ['cd {}'.format(Datadir),
            'rm -fr global_oce_latlon*',
            'curl -L -J -O https://ndownloader.figshare.com/files/6494718',
            'tar -xvzf global_oce_latlon.tar.gz',
            'rm -f global_oce_latlon.tar.gz']
subprocess.call('&&'.join(commands), shell=True)

# Download xarray test
commands = ['cd {}'.format(Datadir),
            'rm -f rasm.nc',
            'wget https://github.com/pydata/xarray-data/raw/master/rasm.nc']
subprocess.call('&&'.join(commands), shell=True)

# Create one extra file
for filename in listdir('{}global_oce_latlon/'.format(Datadir)):
    it = '0000039600'
    if it in filename:
        src = '{}global_oce_latlon/{}'.format(Datadir, filename)
        dst = src.replace(it, '0000039601')
        copyfile(src, dst)

# Urls catalogs
xmitgcm_url = "{}catalog_xmitgcm.yaml".format(Datadir)
xarray_url = "{}catalog_xarray.yaml".format(Datadir)


# Test
@pytest.mark.parametrize("name, catalog_url",
                         [("xmitgcm_iters", xmitgcm_url),
                          ("xmitgcm_no_iters", xmitgcm_url),
                          ("xarray", xarray_url),
                          ("error", "error.yaml"),
                          ("error", xarray_url),
                          ("grd_rect", xarray_url),
                          ("grd_curv", xarray_url)])
def test_opening_and_saving(name, catalog_url):
    if name == 'error':
        # Check error
        with pytest.raises(ValueError):
            from_catalog(name, catalog_url)
    else:
        # Open oceandataset
        if name == 'grd_rect':
            # Dask warning (I think because od NaNs)
            with pytest.warns(UserWarning):
                od1 = from_catalog(name, catalog_url)
        else:
            od1 = from_catalog(name, catalog_url)

        # Check dimensions
        if name != 'xarray':
            dimsList = ['X', 'Y', 'Xp1', 'Yp1']
            assert set(dimsList).issubset(set(od1.dataset.dims))

            # Check coordinates
            coordsList = ['XC', 'YC', 'XG', 'YG', 'XU', 'YU', 'XV', 'YV']
            assert set(coordsList).issubset(set(od1.dataset.coords))

            # Check NaNs
            assert all([not np.isnan(od1.dataset[coord].values).any()
                        for coord in coordsList])

        # Check shift
        if name == 'xmitgcm_iters':
            sizes = od1.dataset.sizes
            assert sizes['time'] - sizes['time_midp'] == 1
            assert all(['time_midp' in od1.dataset[var].dims
                        for var in od1.dataset.data_vars if 'ave' in var])

        # Save to netcdf
        filename = 'tmp.nc'
        od1.to_netcdf(filename)

        # Reopen
        od2 = from_netcdf(filename)

        # Check dataset
        if name == 'xarray':
            assert od1.dataset.identical(od2.dataset)

        # Clean up
        subprocess.call('rm -f ' + filename, shell=True)
