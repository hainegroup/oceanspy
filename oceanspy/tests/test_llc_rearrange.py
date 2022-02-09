import numpy as _np
import pytest
import xarray as _xr

# From OceanSpy
from oceanspy import open_oceandataset
from oceanspy.llc_rearrange import Dims
from oceanspy.llc_rearrange import LLCtransformation as LLC
from oceanspy.llc_rearrange import (
    arct_connect,
    shift_dataset,
    rotate_dataset,
    mates,
    rotate_vars,
)

Datadir = "./oceanspy/tests/Data/"
ECCO_url = "{}catalog_ECCO.yaml".format(Datadir)
od = open_oceandataset.from_catalog("LLC", ECCO_url)

Nx = od._ds.dims["X"]
Ny = od._ds.dims["Y"]

_datype = _xr.core.dataarray.DataArray
_dstype = _xr.core.dataset.Dataset

@pytest.mark.parametrize(
    "od, var, expected",
    [
        (od, "T", ("X", "Y", "face", "Z", "time")),
        (od, "U", ("Xp1", "Y", "face", "Z", "time")),
        (od, "V", ("X", "Yp1", "face", "Z", "time")),
    ],
)
def test_original_dims(od, var, expected):
    """test original dimensions"""
    dims = Dims([dim for dim in od._ds[var].dims][::-1])
    assert dims == expected


faces = [k for k in range(13)]


expected = [2, 5, 7, 10]  # most faces that connect with arctic cap face=6
acshape = (Nx // 2, Ny)


@pytest.mark.parametrize(
    "od, faces, expected, atype",
    [
        (od, faces, expected, _datype),
        (od, faces[:2], [0, 0, 0, 0], int),
        (od, faces[:6], [0, 0, 0, 0], int),
        (od, [0, 1, 2, 6], [2, 0, 0, 0], _datype),
        (od, faces[:7], [2, 5, 0, 0], _datype),
        (od, faces[6:], [0, 0, 7, 10], int),
    ],
)
def test_arc_connect(od, faces, expected, atype):
    ds = od._ds
    arc_faces, *a, DS = arct_connect(ds, "XG", faces)
    assert arc_faces == expected
    if len(DS) > 0:
        assert type(DS[0]) == atype


transf = "arctic_crown"
cent = ["Atlantic", "Pacific"]
varlist = ["T", "U", "V", "XG", "YG"]

@pytest.mark.parametrize(
    "od, faces, varlist, transf, centered, drop, X0, X1, Y0, Y1",
    [
        (od, 'all', varlist, transf, cent[0], False, 0, 359, 0, 314),
        (od, [2, 5, 6, 7, 10], varlist, transf, cent[0], False, 0, 359, 180, 314),
        (od, [2, 5, 7, 10], varlist, transf, cent[0], False, 0, 359, 180, 269),
        (od, [1, 4, 8, 11], varlist, transf, cent[0], False, 0, 359, 90, 179),
        (od, [0, 3, 9, 12], varlist, transf, cent[0], False, 0, 359, 0, 89),
        (od, [6, 7, 8, 9], varlist, transf, cent[0], False, 0, 89, 314, 0),
        (od, [6, 10, 11, 12], varlist, transf, cent[0], False, 90, 179, 314, 0),
        (od, [0, 1, 2, 6], varlist, transf, cent[0], False, 180, 269, 0, 314),
        (od, [3, 4, 5, 6], varlist, transf, cent[0], False, 270, 359, 0, 314),
        (od, [2, 6, 10], varlist, transf, cent[0], False, 90, 269, 180, 314),
        (od, [6, 7, 10], varlist, transf, cent[0], False, 0, 179, 314, 180),
        (od, [5, 6, 7], varlist, transf, cent[1], False, 90, 269, 180, 314),
        (od, [5, 6, 7, 10], varlist, transf, cent[1], False, 90, 359, 180, 314),
        (od, [6, 7, 8, 10, 11], varlist, transf, cent[0], False, 0, 179, 314, 90),
        (od, [4, 5, 6, 7, 8, 10, 11], varlist, transf, cent[1], False, 90, 359, 90, 314),
        (od, [4, 5, 6, 7, 8, 9, 10, 11, 12], varlist, transf, cent[1], False, 90, 359, 0, 314),
        (od, [4, 5, 6, 7, 8], varlist, transf, cent[1], False, 90, 269, 90, 314),
        (od, [0, 3, 9, 12], varlist, transf, cent[1], False, 0, 359, 0, 89),
        (od, [1, 4, 8, 11], varlist, transf, cent[1], False, 0, 359, 90, 179),
        (od, [7, 8, 10, 11], varlist, transf, cent[0], False, 0, 179, 269, 90),
        (od, [1, 2, 10, 11], varlist, transf, cent[0], False, 90, 269, 90, 269),
        (od, [8, 9, 11, 12], varlist, transf, cent[0], False, 0, 179, 179, 0),
        (od, [0, 1, 11, 12], varlist, transf, cent[0], False, 90, 269, 0, 179),
        (od, [0, 1, 3, 4], varlist, transf, cent[0], False, 180, 359, 0, 179),
        (od, [9, 12], varlist, transf, cent[0], False, 0, 179, 89, 0),
        (od, [0, 12], varlist, transf, cent[0], False, 90, 269, 0, 89),
        (od, [0, 3], varlist, transf, cent[0], False, 180, 359, 0, 89),
        (od, [3, 9], varlist, transf, cent[1], False, 90, 269, 0, 89),
        (od, [0, 9, 12], varlist, transf, cent[0], False, 0, 269, 0, 89),
        (od, [0, 3, 12], varlist, transf, cent[0], False, 90, 359, 0, 89),
        (od, [0, 3, 9], varlist, transf, cent[1], False, 0, 269, 0, 89),        
        (od, [0], varlist[0], transf, cent[0], False, 180, 269, 0, 89),
        (od, [1], varlist[0], transf, cent[0], False, 180, 269, 90, 179),
        (od, [2], varlist[0], transf, cent[0], False, 180, 269, 180, 269),
        (od, [3], varlist[0], transf, cent[0], False, 270, 359, 0, 89),
        (od, [4], varlist[0], transf, cent[0], False, 270, 359, 90, 179),
        (od, [5], varlist[0], transf, cent[0], False, 270, 359, 180, 269),
        (od, [7], varlist[0], transf, cent[0], False, 0, 89, 269, 180),
        (od, [8], varlist[0], transf, cent[0], False, 0, 89, 179, 90),
        (od, [9], varlist[0], transf, cent[0], False, 0, 89, 89, 0),
        (od, [10], varlist[0], transf, cent[0], False, 90, 179, 269, 180),
        (od, [11], varlist[0], transf, cent[0], False, 90, 179, 179, 90),
        (od, [12], varlist[0], transf, cent[0], False, 90, 179, 89, 0),
    ]
)


def test_transformation(od, faces, varlist, transf, centered, drop, X0, X1, Y0, Y1):
    ds = od._ds.reset_coords()
    args = {
        "ds": ds,
        "varlist": varlist,
        "centered": centered,
        "faces": faces,
        "drop": drop,
    }
    if transf == "arctic_crown":
        _transf = LLC.arctic_crown
    elif transf == "arctic_centered":
        _transf = LLC.arctic_centered
    ds = _transf(**args)
    xi, xf = int(ds["X"][0].values), int(ds["X"][-1].values)
    yi, yf = int(ds["Y"][0].values), int(ds["Y"][-1].values)
    assert xi == X0
    assert xf == X1
    assert yi == Y0
    assert yf == Y1



DIMS_c = [dim for dim in od.dataset['XC'].dims if dim not in ["face"]]
DIMS_g = [dim for dim in od.dataset['XG'].dims if dim not in ["face"]]
dims_c = Dims(DIMS_c[::-1])
dims_g = Dims(DIMS_g[::-1])

ds2=[]
ds5=[]
ds7=[]
ds10=[]
ARCT = [ds2, ds5, ds7, ds10]
varlist = ['T', 'U', 'V']
# create dataset
for var_name in varlist:
    *nnn, DS = arct_connect(od.dataset, var_name, faces='all')  # horizontal only
    ARCT[0].append(DS[0])
    ARCT[1].append(DS[1])
    ARCT[2].append(DS[2])
    ARCT[3].append(DS[3])
for i in range(len(ARCT)):  # Not all faces survive the cutout
    if type(ARCT[i][0]) == _datype:
        ARCT[i] = _xr.merge(ARCT[i])

ds2, ds5, ds7, ds10 = ARCT


@pytest.mark.parametrize(
    "ds, dimc, dimg, init_c, final_c, init_g, final_g",
    [
        (ds2, dims_c.X, dims_g.X, [0, 44], [0, 44], [0, 44], [0, 44]),
        (ds7, dims_c.X, dims_g.X, [45, 89], [0, 44], [45, 89], [0, 44]),
        (ds5, dims_c.Y, dims_g.Y, [0, 44], [0, 44], [0, 44], [0, 44]),
        (ds10, dims_c.Y, dims_g.Y, [45, 89], [0, 44], [45, 89], [0, 44]),
    ]
)

def test_shift_dataset(ds, dimc, dimg, init_c, final_c, init_g, final_g):
    nds = shift_dataset(ds, dimc, dimg)
    assert [int(ds[dimc][0].values), int(ds[dimc][-1].values)] == init_c
    assert [int(ds[dimg][0].values), int(ds[dimg][-1].values)] == init_g

    assert [int(nds[dimc][0].values), int(nds[dimc][-1].values)] == final_c
    assert [int(nds[dimg][0].values), int(nds[dimg][-1].values)] == final_g



@pytest.mark.parametrize(
    "ds, var, dimc, dimg, rot_dims",
    [
        (od.dataset.isel(face=6), "T", dims_c, dims_g, ('time', 'Z', 'X', 'Y')),
        (od.dataset.isel(face=6), "U", dims_c, dims_g, ('time', 'Z', 'X', 'Yp1')),
        (od.dataset.isel(face=6), "V", dims_c, dims_g, ('time', 'Z', 'Xp1', 'Y')),
        (0, 'T', dims_c, dims_g, ('time', 'Z', 'X', 'Y')),
        ('string', 'T', dims_c, dims_g, ('time', 'Z', 'X', 'Y')),
    ]
)


def test_rotate_dataset(ds, var, dimc, dimg, rot_dims):
    nds = rotate_dataset(ds, dimc, dimg)
    if type(ds) == _dstype:
        nvar = nds[var]
        assert nvar.dims == rot_dims
    


@pytest.mark.parametrize(
    "ds, var, dims0, rot_dims",
    [
        (od.dataset.isel(face=2), 'T', ('time', 'Z', 'Y', 'X'), ('time', 'Z', 'Y', 'X')),
        (od.dataset.isel(face=2), 'U', ('time', 'Z', 'Y', 'Xp1'), ('time', 'Z', 'Yp1', 'X')),
        (od.dataset.isel(face=2), 'V', ('time', 'Z', 'Yp1', 'X'), ('time', 'Z', 'Y', 'Xp1')),
        (mates(ds2), 'V', ('time', 'Z', 'Yp1', 'X'), ('time', 'Z', 'Y', 'Xp1')),
    ]
)

def test_rotate_vars(ds, var, dims0, rot_dims):
    nds = rotate_vars(ds)
    if type(ds) == _dstype:
        nvar = nds[var]
        assert nvar.dims == rot_dims

