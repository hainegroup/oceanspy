# Import modules
import urllib

import numpy as np
import pytest
import yaml

# Import oceanspy
from oceanspy.open_oceandataset import _find_entries, from_catalog, from_netcdf

# SCISERVER DATASETS
url = (
    "https://raw.githubusercontent.com/hainegroup/oceanspy/"
    "main/sciserver_catalogs/datasets_list.yaml"
)
f = urllib.request.urlopen(url)
SCISERVER_DATASETS = yaml.safe_load(f)["datasets"]["sciserver"]

# Directory
Datadir = "./oceanspy/tests/Data/"

# Urls catalogs
xmitgcm_url = "{}catalog_xmitgcm.yaml".format(Datadir)
xarray_url = "{}catalog_xarray.yaml".format(Datadir)
ECCO_url = "{}catalog_ECCO.yaml".format(Datadir)
hycom_url = "{}hycom_test.yaml".format(Datadir)


# Test SciServer
@pytest.mark.parametrize("names", [SCISERVER_DATASETS])
def test_find_entries(names):
    for name in names:
        _find_entries(name, None)


@pytest.mark.parametrize(
    "name, catalog_url",
    [
        ("xmitgcm_iters", xmitgcm_url),
        ("xmitgcm_no_iters", xmitgcm_url),
        ("xarray", xarray_url),
        ("error", xarray_url),
        ("grd_rect", xarray_url),
        ("grd_curv", xarray_url),
        ("LLC", ECCO_url),
        ("HYCOM", hycom_url),
    ],
)
def test_opening_and_saving(name, catalog_url, tmp_path):
    if name == "error":
        # Open oceandataset
        with pytest.raises(ValueError):
            from_catalog(name, catalog_url)
    else:
        # Open oceandataset
        od1 = from_catalog(name, catalog_url)

        # Check dimensions
        if name not in ["xarray", "HYCOM"]:
            dimsList = ["X", "Y", "Xp1", "Yp1"]
            assert set(dimsList).issubset(set(od1.dataset.dims))

            # Check coordinates
            if name == "LLC":
                coordsList = ["XC", "YC", "XG", "YG"]
                assert isinstance(od1.face_connections["face"], dict)
                assert set(["face"]).issubset(set(od1.dataset.dims))

            elif name == "HYCOM":
                coordsList = ["XC", "YC"]
            else:
                coordsList = ["XC", "YC", "XG", "YG", "XU", "YU", "XV", "YV"]
            assert set(coordsList).issubset(set(od1.dataset.coords))

            # Check NaNs
            assert all(
                [not np.isnan(od1.dataset[coord].values).any() for coord in coordsList]
            )

        # Check shift
        if name == "xmitgcm_iters":
            sizes = od1.dataset.sizes
            assert sizes["time"] - sizes["time_midp"] == 1
            assert all(
                [
                    "time_midp" in od1.dataset[var].dims
                    for var in od1.dataset.data_vars
                    if "ave" in var
                ]
            )
            # This is failing on CI for time_midp
            od1._ds["time_midp"].encoding = od1._ds["time"].encoding

        # Save to netcdf
        filename = str(tmp_path / f"tmp_{name}.nc")
        od1.to_netcdf(filename)

        # Reopen
        if name == "LLC":
            args = {"decode_times": False}
        else:
            args = {}
        from_netcdf(filename, **args)
