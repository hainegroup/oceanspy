# Import modules
import pytest
import subprocess

# Import oceanspy
from oceanspy.open_oceandataset import from_netcdf, from_catalog

# Download xmitgcm test
commands = ['cd ./oceanspy/new_tests/Data/',
            'rm -fr global_oce_latlon*',
            'curl -L -J -O https://ndownloader.figshare.com/files/6494718',
            'tar -xvzf global_oce_latlon.tar.gz',
            'rm -f global_oce_latlon.tar.gz']
subprocess.call('&&'.join(commands), shell=True)

# Download xarray test
commands = ['cd ./oceanspy/new_tests/Data/',
            'rm -f rasm.nc',
            'wget https://github.com/pydata/xarray-data/raw/master/rasm.nc']
subprocess.call('&&'.join(commands), shell=True)


# Test
@pytest.mark.parametrize("name, catalog_url",
                         [("xmitgcm_iters",
                           "./oceanspy/new_tests/Data/catalog_xmitgcm.yaml"),
                          ("xmitgcm_no_iters",
                           "./oceanspy/new_tests/Data/catalog_xmitgcm.yaml"),
                          ("xarray",
                           "./oceanspy/new_tests/Data/catalog_xarray.yaml"),
                          ("error",
                           "error.yaml"),
                          ("error",
                           "./oceanspy/new_tests/Data/catalog_xarray.yaml")])
def test_opening_and_saving(name, catalog_url):
    if name == 'error':
        # Check error
        with pytest.raises(ValueError):
            from_catalog(name, catalog_url)
    else:
        # Open oceandataset
        od1 = from_catalog(name, catalog_url)

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
