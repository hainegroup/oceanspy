# Import modules
import pytest

# From OceanSpy
from oceanspy.utils import (spherical2cartesian,
                            great_circle_path, cartesian_path)


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
