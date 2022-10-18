# Import modules
import pytest

# From OceanSpy
from oceanspy.utils import cartesian_path, great_circle_path, spherical2cartesian, _rel_lon


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


X0 = _np.array([161, -161]) # track begins west, ends east
X1 = list(_np.arange(161,180, 2)) + list(_np.arange(-179, -159, 2)) # same as above
X1 = _np.array(X1)
X2 = list(_np.arange(-179, -159, 2))[::-1] + list(_np.arange(161,180, 2))[::-1]
X2 = _np.array(X2)  # track begins east, ends west.
X3 = _np.array([-20, 20])  # crossing in zero.
X4 = _np.array([-20, -1])  # no crossing.

@pytest.mark.parametrize(
    "XRange, expected",
    [
        (X0, 53.67),
        (X1, 53.67), 
        (X2, 53.67),
        (X3, 180),
        (X4, 180),
    ]
)
def test_rel_lon(XRange, expected):
    """ test the function rel_lon which redefines the reference long.
    """
    ref_lon = _rel_lon(XRange)
    assert _np.round(ref_lon, 2) == expected

