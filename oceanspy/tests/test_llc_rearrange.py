import pytest
import numpy as _np

# From OceanSpy
from oceanspy import open_oceandataset
from oceanspy.llc_rearrange import (
    Dims,
    make_chunks,
    pos_chunks,
    chunk_sizes,
    face_connect,
    arct_connect,
)
from oceanspy.llc_rearrange import LLCtransformation as LLC


Datadir = './oceanspy/tests/Data/'
ECCO_url = "{}catalog_ECCO.yaml".format(Datadir)
od = open_oceandataset.from_catalog('LLC', ECCO_url)

Nx = od._ds.dims['X']
Ny = od._ds.dims['Y']


@pytest.mark.parametrize(
    "od, var, expected", [
        (od, 'T', ('X', 'Y', 'face', 'Z', 'time')),
        (od, 'U', ('Xp1', 'Y', 'face', 'Z', 'time')),
        (od, 'V', ('X', 'Yp1', 'face', 'Z', 'time')),
    ]
)
def test_original_dims(od, var, expected):
    """ test original dimensions
    """
    dims = Dims([dim for dim in od._ds[var].dims][::-1])
    assert dims == expected


faces = [k for k in range(13)]
nrot_expected = [0, 1, 2, 3, 4, 5]
rot_expected = [7, 8, 9, 10, 11, 12]


@pytest.mark.parametrize(
    "od, faces, nrot_expected, rot_expected", [
        (od, faces, nrot_expected, rot_expected),
        (od, faces[3:6], nrot_expected[3:6], []),
        (od, faces[8:11], [], rot_expected[1:4])
    ]
)
def test_face_connect(od, faces, nrot_expected, rot_expected):
    ds = od._ds
    nrot_faces, a, b, rot_faces, *nn = face_connect(ds, faces)
    assert nrot_faces == nrot_expected
    assert rot_faces == rot_expected


expected = [2, 5, 7, 10]  # faces that connect with arctic cap face=6
acshape = (Nx // 2, Ny)


@pytest.mark.parametrize(
    "od, faces, expected, acshape", [
        (od, faces, expected, acshape),
        (od, faces[:2], [], []),
        (od, faces[:6], [], []),
        (od, [0, 1, 2, 6], expected[:1], acshape),
        (od, faces[:7], expected[:2], acshape),
        (od, faces[6:], expected[2:], acshape)
    ]
)
def test_arc_connect(od, faces, expected, acshape):
    ds = od._ds
    arc_faces, *a, ARCT = arct_connect(ds, 'XG', faces)
    assert arc_faces == expected
    assert len(ARCT) == len(expected)
    if len(ARCT) > 0:
        assert ARCT[0].shape == acshape  # arctic crown


@pytest.mark.parametrize(
    "faces, Nx, Ny, rot, exp_tNX, exp_tNY", [
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
        ([8, 9, 10, 11], Nx, Ny, True, None, None)
    ]
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
    "faces, rot, NX, NY, expCX, expCY, epx, epy, epax, epay", [
        (faces[:7], False, Nx, Ny, [[0, 90], [90, 180]],
                                   [[0, 90], [90, 180], [180, 270]],
                                   [[0, 90], [0, 90], [0, 90],
                                    [90, 180], [90, 180], [90, 180]],
                                   [[0, 90], [90, 180], [180, 270],
                                    [0, 90], [90, 180], [180, 270]],
                                   [[0, 90], [90, 180]],
                                   [[270, 315], [270, 315]]),
        (faces[6:], True, Nx, Ny, [[0, 90], [90, 180]],
                                  [[0, 90], [90, 180], [180, 270]],
                                  [[0, 90], [0, 90], [0, 90],
                                   [90, 180], [90, 180], [90, 180]],
                                  [[0, 90], [90, 180], [180, 270],
                                   [0, 90], [90, 180], [180, 270]],
                                  [[0, 90], [90, 180]],
                                  [[270, 315], [270, 315]]),
        (faces[:3] + [6], False, Nx, Ny, [[0, 90]],
                                         [[0, 90], [90, 180], [180, 270]],
                                         [[0, 90], [0, 90], [0, 90]],
                                         [[0, 90], [90, 180], [180, 270]],
                                         [[0, 90]], [[270, 315]]),
        (faces[:3], False, Nx, Ny, [[0, 90]],
                                   [[0, 90], [90, 180], [180, 270]],
                                   [[0, 90], [0, 90], [0, 90]],
                                   [[0, 90], [90, 180], [180, 270]],
                                   [], []),
        (faces[3:7], False, Nx, Ny, [[0, 90]],
                                    [[0, 90], [90, 180], [180, 270]],
                                    [[0, 90], [0, 90], [0, 90]],
                                    [[0, 90], [90, 180], [180, 270]],
                                    [[0, 90]], [[270, 315]]),
        (faces[3:6], False, Nx, Ny, [[0, 90]],
                                    [[0, 90], [90, 180], [180, 270]],
                                    [[0, 90], [0, 90], [0, 90]],
                                    [[0, 90], [90, 180], [180, 270]],
                                    [], []),
        (faces[6:10], True, Nx, Ny, [[0, 90]],
                                    [[0, 90], [90, 180], [180, 270]],
                                    [[0, 90], [0, 90], [0, 90]],
                                    [[0, 90], [90, 180], [180, 270]],
                                    [[0, 90]], [[270, 315]]),
        (faces[7:10], True, Nx, Ny, [[0, 90]],
                                    [[0, 90], [90, 180], [180, 270]],
                                    [[0, 90], [0, 90], [0, 90]],
                                    [[0, 90], [90, 180], [180, 270]],
                                    [], []),
        ([6] + faces[10:], True, Nx, Ny, [[0, 90]],
                                         [[0, 90], [90, 180], [180, 270]],
                                         [[0, 90], [0, 90], [0, 90]],
                                         [[0, 90], [90, 180], [180, 270]],
                                         [[0, 90]], [[270, 315]],),
        (faces[10:], True, Nx, Ny, [[0, 90]],
                                   [[0, 90], [90, 180], [180, 270]],
                                   [[0, 90], [0, 90], [0, 90]],
                                   [[0, 90], [90, 180], [180, 270]],
                                   [], []),
        ([6, 7, 10], True, Nx, Ny, [[0, 90], [90, 180]], [[0, 90]],
                                   [[0, 90], [90, 180]],
                                   [[0, 90], [0, 90]],
                                   [[0, 90], [90, 180]],
                                   [[90, 135], [90, 135]]),
        ([2, 5, 6], False, Nx, Ny, [[0, 90], [90, 180]],
                                   [[0, 90]],
                                   [[0, 90], [90, 180]],
                                   [[0, 90], [0, 90]],
                                   [[0, 90], [90, 180]],
                                   [[90, 135], [90, 135]]),
    ]
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


transf = ['arctic_crown', 'arctic_centered']
cent = ['Atlantic', 'Pacific']
varlist = ['T', 'U', 'V']


@pytest.mark.parametrize(
    "od, faces, varlist, transf, centered, drop, expNX, expNY", [
        (od, [2, 6, 10], 'all', transf[0], cent[0], True, 179, 134),
        (od, [2, 5, 6, 7, 10], 'T', transf[0], cent[0], False, 360, 135),
        (od, faces[:6], 'T', transf[0], cent[0], False, 180, 270),
        (od, [2, 5, 6, 7, 10], 'T', transf[1], cent[0], True, 269, 269),
        (od, faces, 'T', transf[1], cent[0], True, 269, 269),
    ],
)
def test_transformation(od, faces, varlist, transf, centered, drop, expNX,
                        expNY):
    ds = od._ds.reset_coords()
    args = {
        "ds": ds,
        "varlist": varlist,
        "centered": centered,
        "faces": faces,
        "drop": drop,
    }
    if transf == 'arctic_crown':
        _transf = LLC.arctic_crown
    elif transf == 'arctic_centered':
        _transf = LLC.arctic_centered
    ds = _transf(**args)
    Nx = ds.dims['X']
    Ny = ds.dims['Y']
    assert Nx == expNX
    assert Ny == expNY


@pytest.mark.parametrize(
    "od, tNX, tNY, X0", [
        (od, 100, 200, 0),
        (od, 200, 400, 100),
        (od, None, None, 'Five'),
        (od, 'Four', None, None),
        (od, 0, 0, 0),
    ]
)
def test_make_vars(od, tNX, tNY, X0):
    ds = od._ds.reset_coords()
    if isinstance(tNX, int) and isinstance(tNY, int) and isinstance(X0, int):
        nds = make_array(ds, tNX, tNY, X0)
        assert (set(nds.dims) - set(ds.dims)) == set([])
        assert nds.dims['X'] == tNX
        assert nds.dims['Y'] == tNY
        assert nds.dims['Z'] == ds.dims['Z']
        assert nds.dims['time'] == ds.dims['time']
    else:
        with pytest.raises(TypeError):
            nds = make_array(ds, tNX, tNY, X0)


@pytest.mark.parametrize(
    "od, tNX, tNY, X0, varlist", [
        (od, 100, 200, 0, ['T']),
        (od, 200, 400, 10, ['U']),
        (od, 200, 400, 0, ['T', 'U', 'V'])
    ]
)
def test_init_vars(od, tNX, tNY, X0, varlist):
    ds = od._ds.reset_coords()
    nds = make_array(ds, tNX, tNY, X0)
    nds = init_vars(ds, nds, varlist)
    for var in varlist:
        assert set(ds[var].dims) - set(nds[var].dims) == set(["face"])


def _is_connect(faces, rotated=False):
    """ do faces in a facet connect? Not applicable to arc cap, and only
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
                    iA = [_np.where(faces[k] == A_fac)[0][0]
                          for k in range(len(faces))
                          if faces[k] in A_fac]
                    iB = [_np.where(faces[k] == B_fac)[0][0]
                          for k in range(len(faces))
                          if faces[k] in B_fac]
                    if iA != iB:
                        cont = 0
                if len(A_list) == 2:
                    if abs(A_list[1] - A_list[0]) > 1:
                        cont = 0
                    else:
                        iA = [_np.where(faces[k] == A_fac)[0][0]
                              for k in range(len(faces))
                              if faces[k] in A_fac]
                        iB = [_np.where(faces[k] == B_fac)[0][0]
                              for k in range(len(faces))
                              if faces[k] in B_fac]
                        if iA != iB:
                            cont = 0
            else:
                cont = 0
    return cont
