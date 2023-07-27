# Import modules
import numpy as _np
import pytest

# From OceanSpy
from oceanspy.utils import (
    _reset_range,
    cartesian_path,
    circle_path_array,
    connector,
    edge_completer,
    edge_find,
    edge_slider,
    great_circle_path,
    spherical2cartesian,
    splitter,
    viewer_to_range,
)


def test_RNone():
    spherical2cartesian(1, 1)


def test_error_viewer_to_range():
    with pytest.raises(TypeError):
        viewer_to_range("does not eval to a list")
        viewer_to_range(0)
        viewer_to_range(["not from viewer"])
        viewer_to_range([{"type": "other"}])
        viewer_to_range([{"type": "Polygon", "coordinates": "a"}])


def test_error_path():
    with pytest.raises(ValueError):
        great_circle_path(1, 1, 1, 1)

    with pytest.raises(ValueError):
        cartesian_path(1, 1, 1, 1)

    with pytest.raises(ValueError):
        great_circle_path(1, 2, 3, 4, delta_km=-1)

    with pytest.raises(ValueError):
        cartesian_path(1, 2, 3, 4, delta=-1)


def test_error_circle_array():
    with pytest.raises(ValueError):
        circle_path_array([1, 1], [1, 1], R=6371.0)
    with pytest.raises(ValueError):
        circle_path_array([0], [0, 0, 0], R=6371.0)
    with pytest.raises(TypeError):
        circle_path_array([1, 0], [0, 0], R=None)


@pytest.mark.parametrize(
    "lats, lons, symmetry, resolution",
    [
        ([0, 0.5], [0, 0], "latitude", 25),
        ([0, 0], [0, 0.5], "longitude", 25),
        ([0, 0], [0, 0.125], False, 25),
    ],
)
def test_circle_path_array(lats, lons, symmetry, resolution):
    nY, nX = circle_path_array(lats, lons, R=6371.0, _res=resolution)
    if symmetry == "latitude":
        assert (nX == _np.zeros(len(nX))).all()
    elif symmetry == "longitude":
        assert (nY == _np.zeros(len(nY))).all()
    else:
        assert len(nY) == len(lats)


coords1 = [[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]]
coords2 = [[[5, 0], [4, 1], [3, 2], [2, 3], [1, 4], [0, 5]]]
coords3 = [[[0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11]]]
lons = []
coords4 = '[{"type":"Point","coordinates":[-169.23960833202577,22.865677261831266]}]'
coords5 = '[{"type":"Point","coordinates":[636.7225446274502, -56.11128546740994]}]'
coords6 = '[{"type":"Point","coordinates":[754.2277421326479, -57.34299561290217]}]'
coords7 = '[{"type":"Point","coordinates":[-424.42989807993234, 37.87263032287052]}]'


@pytest.mark.parametrize(
    "coords, types, lon, lat",
    [
        (coords1, "Polygon", [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
        (coords2, "Polygon", [5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
        (coords3, "Polygon", [0, 0, 0, 0, 0, 0], [6, 7, 8, 9, 10, 11]),
        (coords1, "LineString", [[0, 0]], [[1, 1]]),
        (coords2, "LineString", [[5, 0]], [[4, 1]]),
        (coords3, "LineString", [[0, 6]], [[0, 7]]),
        (coords4, "Point", [-169.23960833202577], [22.865677261831266]),
        (coords5, "Point", [-83.27745537254975], [-56.11128546740994]),
        (coords6, "Point", [34.227742132647904], [-57.34299561290217]),
        (coords7, "Point", [-64.42989807993234], [37.87263032287052]),
    ],
)
def test_viewer_to_range(coords, types, lon, lat):
    if type(coords) == list:
        p = [{"type": types, "coordinates": list(coords)}]
    elif type(coords) == str:
        p = coords
    x, y = viewer_to_range(p)
    assert x == lon
    assert y == lat


X0 = _np.array([161, -161])  # track begins west, ends east
X1 = list(_np.arange(161, 180, 2)) + list(_np.arange(-179, -159, 2))  # same as above
X1 = _np.array(X1)
X2 = list(_np.arange(-179, -159, 2))[::-1] + list(_np.arange(161, 180, 2))[::-1]
X2 = _np.array(X2)  # track begins east, ends west.
X3 = _np.array([-20, 20])  # crossing in zero.
X4 = _np.arange(-20, 20, 2)  # no crossing.
XX = list(_np.arange(161, 180, 2)) + list(_np.arange(-179, 20, 2))
X5 = _np.array(XX)
X6 = _np.array(XX[::-1])
X7 = _np.array([20, -20])


@pytest.mark.parametrize(
    "XRange, x0, expected_ref",
    [
        (X0, X0, 53.67),
        (X1, X0, 53.67),
        (X2, X0, 53.67),
        (X3, X3, 180),
        (X4, X3, 180),
        (X5, None, 180),
        (X6, None, 180),
        (X7, X7, 6.67),
    ],
)
def test_reset_range(XRange, x0, expected_ref):
    """test the function rel_lon which redefines the reference long."""
    x_range, ref_lon = _reset_range(XRange)
    if x0 is not None:
        assert len(x_range) == 2
        assert x_range.all() == x0.all()
    else:
        assert x_range is None
    assert _np.round(ref_lon, 2) == expected_ref


@pytest.mark.parametrize(
    "x, y, expected",
    [(1, 2, [0, 2]), (20, 80, [20, 89]), (85, 80, [89, 80]), (85, 1, [85, 0])],
)
def test_edge_find(x, y, expected):
    point = edge_find(x, y, 89)
    assert point == expected


x1 = _np.array([k for k in range(0, 85)])
y1 = 10 * _np.ones(_np.shape(x1))
x2 = _np.array([k for k in range(5, 89)])
y2 = 10 * _np.ones(_np.shape(x2))
x3 = _np.array([k for k in range(5, 85)])
y3 = 10 * _np.ones(_np.shape(x3))


@pytest.mark.parametrize(
    "x, y, exp",
    [(x1, y1, [0, 89]), (x2, y2, [0, 89]), (x3, y3, [0, 89]), (x3[::-1], y3, [89, 0])],
)
def test_edge_completer(x, y, exp):
    xn, yn = edge_completer(x, y, 89)
    diffs = abs(_np.diff(xn)) + abs(_np.diff(yn))
    assert [xn[0], xn[-1]] == exp
    assert _np.max(diffs) == _np.min(diffs) == 1


x1 = _np.array([k for k in range(0, 85, 10)])
y1 = [int(k) for k in _np.linspace(20, 40, len(x1))]


@pytest.mark.parametrize(
    "x, y", [(x1, y1), (x1[::-1], y1), (x1[::-1], y1[::-1]), (x1, y1[::-1])]
)
def test_connector(x, y):
    xn, yn = connector(x, y)
    diffs = abs(_np.diff(xn)) + abs(_np.diff(yn))
    assert len(xn) == len(yn)
    assert _np.max(diffs) == _np.min(diffs) == 1
    assert set(x).issubset(xn)
    assert set(y).issubset(yn)


x1 = [k for k in range(0, 85, 10)]
y1 = [int(k) for k in _np.linspace(20, 40, len(x1))]
fs1 = len(x1) * [5]

x2 = [k for k in range(10, 85, 10)][::-1]
y2 = [int(k) for k in _np.linspace(20, 40, len(x2))][::-1]
fs2 = len(x1) * [2]

y3 = [k for k in range(0, 89, 1)]
x3 = len(y3) * [2]
fs3 = len(y3) * [1]

X1, Y1, Fs1 = [x1, x2], [y1, y2], [fs1, fs2]

X2, Y2, Fs2 = [x1, x3], [y1, y3], [fs1, fs3]


@pytest.mark.parametrize("X, Y, Fs", [(X1, Y1, Fs1), (X2, Y2, Fs2)])
def test_splitter(X, Y, Fs):
    xs = X[0] + X[1]
    ys = Y[0] + Y[1]
    fs = Fs[0] + Fs[1]
    nX, nY = splitter(xs, ys, fs)
    assert len(nX) == len(nY)
    assert nX[0] == X[0]
    assert nX[1] == X[1]
    assert nY[0] == Y[0]
    assert nY[1] == Y[1]


@pytest.mark.parametrize(
    "X, Y, Fs, exp",
    [
        ([45, 46], [89, 0], [1, 2], [46, 89]),
        ([89, 0], [45, 44], [1, 4], [89, 44]),
        ([45, 46], [0, 0], [1, 0], [46, 0]),
        ([89, 0], [45, 40], [8, 9], [89, 40]),
        ([0, 89], [45, 40], [8, 7], [0, 40]),
        ([45, 40], [89, 0], [8, 11], [40, 89]),
        ([89, 89], [89, 89], [0, 1], [None, None]),
        ([0, 44], [45, 89], [1, 11], [0, 45]),
        ([89, 40], [45, 9], [4, 8], [89, 49]),
        ([45, 89], [0, 40], [8, 4], [49, 0]),
    ],
)
def test_edge_slider(X, Y, Fs, exp):
    if set([X[0], Y[0]]) == set([89]):
        with pytest.raises(ValueError):
            edge_slider(X[0], Y[0], Fs[0], X[1], Y[1], Fs[1])
    else:
        newP = edge_slider(X[0], Y[0], Fs[0], X[1], Y[1], Fs[1])
    assert newP == exp
