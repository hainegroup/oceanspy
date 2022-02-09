import numpy as _np
import pytest
import xarray as _xr

# From OceanSpy
from oceanspy import open_oceandataset
from oceanspy.llc_rearrange import Dims
from oceanspy.llc_rearrange import LLCtransformation as LLC
from oceanspy.llc_rearrange import (
    arct_connect,
    chunk_sizes,
    face_connect,
    make_chunks,
    pos_chunks,
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
nrot_expected = [0, 1, 2, 3, 4, 5]
rot_expected = [7, 8, 9, 10, 11, 12]


@pytest.mark.parametrize(
    "od, faces, nrot_expected, rot_expected",
    [
        (od, faces, nrot_expected, rot_expected),
        (od, faces[3:6], nrot_expected[3:6], []),
        (od, faces[8:11], [], rot_expected[1:4]),
    ],
)
def test_face_connect(od, faces, nrot_expected, rot_expected):
    ds = od._ds
    nrot_faces, a, b, rot_faces, *nn = face_connect(ds, faces)
    assert nrot_faces == nrot_expected
    assert rot_faces == rot_expected


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


@pytest.mark.parametrize(
    "faces, Nx, Ny, rot, exp_tNX, exp_tNY",
    [
        (faces[:6], Nx, Ny, False, 180, 270),
        (faces[6:], Nx, Ny, True, 180, 270),
        (faces[:3], Nx, Ny, False, 90, 270),
        (faces[3:6], Nx, Ny, False, 90, 270),
        (faces[7:10], Nx, Ny, True, 90, 270),
        (faces[10:], Nx, Ny, True, 90, 270),
        ([0, 2], Nx, Ny, False, None, None),
        ([1, 3], Nx, Ny, False, None, None),
        ([0, 4], Nx, Ny, False, None, None),
        ([0, 5], Nx, Ny, False, None, None),
        ([1, 3], Nx, Ny, False, None, None),
        ([1, 5], Nx, Ny, False, None, None),
        ([2, 3], Nx, Ny, False, None, None),
        ([2, 4], Nx, Ny, False, None, None),
        ([0, 1, 4, 5], Nx, Ny, False, None, None),
        ([1, 2, 3, 4], Nx, Ny, False, None, None),
        ([0, 4, 5], Nx, Ny, False, None, None),
        ([7, 10], Nx, Ny, True, 180, 90),
        ([7, 11], Nx, Ny, True, None, None),
        ([7, 12], Nx, Ny, True, None, None),
        ([8, 10], Nx, Ny, True, None, None),
        ([8, 12], Nx, Ny, True, None, None),
        ([9, 10], Nx, Ny, True, None, None),
        ([9, 11], Nx, Ny, True, None, None),
        ([7, 8, 11, 12], Nx, Ny, True, None, None),
        ([8, 9, 10, 11], Nx, Ny, True, None, None),
    ],
)
def test_chunk_sizes(faces, Nx, Ny, rot, exp_tNX, exp_tNY):
    if _is_connect(faces, rotated=rot):
        tNy, tNx = chunk_sizes(faces, [Nx], [Ny], rotated=rot)
        assert tNy == exp_tNY
        assert tNx == exp_tNX
    else:
        with pytest.raises(ValueError):
            tNy, tNx = chunk_sizes(faces, [Nx], [Ny], rotated=rot)
            assert tNy == exp_tNY
            assert tNx == exp_tNX


@pytest.mark.parametrize(
    "faces, rot, NX, NY, expCX, expCY, epx, epy, epax, epay",
    [
        (
            faces[:7],
            False,
            Nx,
            Ny,
            [[0, 90], [90, 180]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90], [0, 90], [0, 90], [90, 180], [90, 180], [90, 180]],
            [[0, 90], [90, 180], [180, 270], [0, 90], [90, 180], [180, 270]],
            [[0, 90], [90, 180]],
            [[270, 315], [270, 315]],
        ),
        (
            faces[6:],
            True,
            Nx,
            Ny,
            [[0, 90], [90, 180]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90], [0, 90], [0, 90], [90, 180], [90, 180], [90, 180]],
            [[0, 90], [90, 180], [180, 270], [0, 90], [90, 180], [180, 270]],
            [[0, 90], [90, 180]],
            [[270, 315], [270, 315]],
        ),
        (
            faces[:3] + [6],
            False,
            Nx,
            Ny,
            [[0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90], [0, 90], [0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90]],
            [[270, 315]],
        ),
        (
            faces[:3],
            False,
            Nx,
            Ny,
            [[0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90], [0, 90], [0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [],
            [],
        ),
        (
            faces[3:7],
            False,
            Nx,
            Ny,
            [[0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90], [0, 90], [0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90]],
            [[270, 315]],
        ),
        (
            faces[3:6],
            False,
            Nx,
            Ny,
            [[0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90], [0, 90], [0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [],
            [],
        ),
        (
            faces[6:10],
            True,
            Nx,
            Ny,
            [[0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90], [0, 90], [0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90]],
            [[270, 315]],
        ),
        (
            faces[7:10],
            True,
            Nx,
            Ny,
            [[0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90], [0, 90], [0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [],
            [],
        ),
        (
            [6] + faces[10:],
            True,
            Nx,
            Ny,
            [[0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90], [0, 90], [0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90]],
            [[270, 315]],
        ),
        (
            faces[10:],
            True,
            Nx,
            Ny,
            [[0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [[0, 90], [0, 90], [0, 90]],
            [[0, 90], [90, 180], [180, 270]],
            [],
            [],
        ),
        (
            [6, 7, 10],
            True,
            Nx,
            Ny,
            [[0, 90], [90, 180]],
            [[0, 90]],
            [[0, 90], [90, 180]],
            [[0, 90], [0, 90]],
            [[0, 90], [90, 180]],
            [[90, 135], [90, 135]],
        ),
        (
            [2, 5, 6],
            False,
            Nx,
            Ny,
            [[0, 90], [90, 180]],
            [[0, 90]],
            [[0, 90], [90, 180]],
            [[0, 90], [0, 90]],
            [[0, 90], [90, 180]],
            [[90, 135], [90, 135]],
        ),
    ],
)
def test_make_chunks(faces, rot, NX, NY, expCX, expCY, epx, epy, epax, epay):
    if rot:
        fs = [k for k in faces if k in _np.arange(7, 13)]
    else:
        fs = [k for k in faces if k in _np.arange(6)]
    tNy, tNx = chunk_sizes(faces, [Nx], [Ny], rotated=rot)
    delNX = 0
    delNY = 0
    afs = []
    if 6 in faces:
        acnrot_fs = [k for k in faces if k in _np.array([2, 5])]
        acrot_fs = [k for k in faces if k in _np.array([7, 10])]
        if rot:
            delNX = int(Nx / 2)
            afs = acrot_fs
        else:
            delNY = int(Ny / 2)
            afs = acnrot_fs
    tNy = tNy + delNY
    tNx = tNx + delNX
    Nxc = _np.arange(0, tNx + 1, Nx)
    Nyc = _np.arange(0, tNy + 1, Ny)
    xChunk, yChunk = make_chunks(Nxc, Nyc)
    assert xChunk == expCX
    assert yChunk == expCY
    py, px, pyarc, pxarc = pos_chunks(fs, afs, yChunk, xChunk)
    assert epy == py
    assert epx == px
    assert epay == pyarc
    assert epax == pxarc




@pytest.mark.parametrize(
    "od, tNX, tNY, X0, varlist",
    [
        (od, 100, 200, 0, ["T"]),
        (od, 200, 400, 10, ["U"]),
        (od, 200, 400, 0, ["T", "U", "V"]),
    ],
)

def _is_connect(faces, rotated=False):
    """do faces in a facet connect? Not applicable to arc cap, and only
    applicable to either rotated or not rotated facets"""
    if rotated is False:
        A_fac = _np.array([0, 1, 2])
        B_fac = _np.array([3, 4, 5])
    elif rotated is True:
        A_fac = _np.array([7, 8, 9])
        B_fac = _np.array([10, 11, 12])
    A_list = [k for k in faces if k in A_fac]
    B_list = [k for k in faces if k in B_fac]
    cont = 1
    if len(A_list) == 0:
        if len(B_list) > 1:
            if len(B_list) == 2:
                if abs(B_list[1] - B_list[0]) > 1:
                    cont = 0
    else:
        if len(B_list) == 0:
            if len(A_list) > 1:
                if len(A_list) == 2:
                    if abs(A_list[1] - A_list[0]) > 1:
                        cont = 0
        else:
            if len(B_list) == len(A_list):
                if len(A_list) == 1:
                    iA = [
                        _np.where(faces[k] == A_fac)[0][0]
                        for k in range(len(faces))
                        if faces[k] in A_fac
                    ]
                    iB = [
                        _np.where(faces[k] == B_fac)[0][0]
                        for k in range(len(faces))
                        if faces[k] in B_fac
                    ]
                    if iA != iB:
                        cont = 0
                if len(A_list) == 2:
                    if abs(A_list[1] - A_list[0]) > 1:
                        cont = 0
                    else:
                        iA = [
                            _np.where(faces[k] == A_fac)[0][0]
                            for k in range(len(faces))
                            if faces[k] in A_fac
                        ]
                        iB = [
                            _np.where(faces[k] == B_fac)[0][0]
                            for k in range(len(faces))
                            if faces[k] in B_fac
                        ]
                        if iA != iB:
                            cont = 0
            else:
                cont = 0
    return cont
