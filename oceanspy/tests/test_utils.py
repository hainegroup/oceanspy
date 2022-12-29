# Import modules
import numpy as _np
import pytest

# From OceanSpy
from oceanspy.utils import (
    _reset_range,
    cartesian_path,
    great_circle_path,
    spherical2cartesian,
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


coords1 = [[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]]
coords2 = [[[5, 0], [4, 1], [3, 2], [2, 3], [1, 4], [0, 5]]]
coords3 = [[[0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11]]]
coords4 = '[{"type":"Point","coordinates":[-169.23960833202577,22.865677261831266]}]'


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
        (X5, _np.array([161, 19]), 113.67),
        (X6, _np.array([161, 19]), 113.67),
        (X7, X7, 180),
    ],
)
def test_reset_range(XRange, x0, expected_ref):
    """test the function rel_lon which redefines the reference long."""
    x_range, ref_lon = _reset_range(XRange)
    assert len(x_range) == 2
    assert x_range.all() == x0.all()
    assert _np.round(ref_lon, 2) == expected_ref
