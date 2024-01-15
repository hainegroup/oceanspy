# Import modules
import numpy as _np
import pytest

# From OceanSpy
from oceanspy.utils import (
    _reset_range,
    cartesian_path,
    circle_path_array,
    connector,
    great_circle_path,
    spherical2cartesian,
    viewer2range,
)


def test_RNone():
    spherical2cartesian(1, 1)


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


tR = ["2012-04-25T00", "2012-04-25T08"]
Point1 = [-37.49995880442971, 56.15599523245322]
Point2 = [-44.90083986169844, 38.27074364198387]
p1 = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"timeFrom": tR[0], "timeTo": tR[1]},
            "geometry": {"type": "Point", "coordinates": Point1},
        }
    ],
}
p2 = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"timeFrom": tR[0], "timeTo": tR[1]},
            "geometry": {"type": "Point", "coordinates": Point1},
        },
        {
            "type": "Feature",
            "properties": {"timeFrom": tR[0], "timeTo": tR[1]},
            "geometry": {"type": "Point", "coordinates": Point2},
        },
    ],
}
p3 = {"A": 1, "B": 0}

Polygon = [
    [-49.39423193218302, 21.652186887750887],
    [-47.54401166786583, -4.241096331293235],
    [-10.010972020288738, -4.241096331293235],
    [-1.5528222405530512, 18.42461597093053],
    [-5.5175799498041425, 29.049519444696543],
    [-22.962513870508992, 27.1846556076123],
    [-27.98454030222706, 22.142679863561654],
    [-19.262073341874636, 18.92541577075147],
    [-10.539606381522212, 21.406311271560995],
    [-11.86119228460591, 9.45629361187676],
    [-21.640927967425295, 4.2093834199523315],
    [-39.87881342998037, 10.237557106013753],
    [-39.61449624936363, 20.418704257499954],
    [-43.314936777997985, 27.1846556076123],
    [-58.64533325376892, 26.94928795043414],
    [-61.552822240553056, 21.652186887750887],
    [-56.26647862821826, 18.675200810535287],
    [-56.002161447601516, 22.387289688558397],
    [-49.39423193218302, 21.652186887750887],
]

LineString = [
    [-29.83476056654425, 35.091987524867804],
    [-27.19158876037684, 44.03609515845176],
    [-27.19158876037684, 48.42109044135867],
    [-28.51317466346055, 54.80814625184328],
]

pnew1 = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"timeFrom": tR[0], "timeTo": tR[1]},
            "geometry": {"type": "Polygon", "coordinates": [Polygon]},
        },
        {
            "type": "Feature",
            "properties": {"timeFrom": tR[0], "timeTo": tR[1]},
            "geometry": {"type": "LineString", "coordinates": LineString},
        },
    ],
}

p4 = pnew = {"type": "FeatureCollection", "features": [pnew1["features"][0]]}
p5 = pnew = {"type": "FeatureCollection", "features": [pnew1["features"][2]]}


@pytest.mark.parametrize(
    "p, timeRange, lats, lons",
    [
        (p1, tR, [Point1[0]], [Point1[1]]),
        (p2, tR, [Point1[0], Point1[0]], [Point1[1]], Point2[1]),
        (p3, None, None, None),
        (
            p4,
            tR,
            [Polygon[k][0] for k in range(len(Polygon))],
            [Polygon[k][1] for k in range(len(Polygon))],
        ),
        (
            p5,
            tR,
            [LineString[k][0] for k in range(len(LineString))],
            [LineString[k][1] for k in range(len(LineString))],
        ),
        (pnew1, None, 1, 1),  # multiple geometry types
    ],
)
def test_viewer2range(p, timeRange, lats, lons):
    if timeRange is not None:
        t, y, x = viewer2range(p)
        assert x == lons
        assert y == lats
        assert t == timeRange
    else:
        if lats is None:
            with pytest.raises(TypeError):
                viewer2range(p)
        else:
            with pytest.raises(ValueError):
                viewer2range(p)


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


x1 = _np.array([k for k in range(0, 85, 10)])
y1 = [int(k) for k in _np.linspace(20, 40, len(x1))]


@pytest.mark.parametrize(
    "x, y",
    [(x1, y1), (x1[::-1], y1), (x1[::-1], y1[::-1]), (x1, y1[::-1]), ([50], [50])],
)
def test_connector(x, y):
    xn, yn = connector(x, y)
    assert len(xn) == len(yn)
    assert set(x).issubset(xn)
    assert set(y).issubset(yn)
    if len(xn) > 1:
        diffs = abs(_np.diff(xn)) + abs(_np.diff(yn))
        assert _np.max(diffs) == _np.min(diffs) == 1
