import numpy as np
import pytest
import xarray as xr

from oceanspy import open_oceandataset
from oceanspy._ospy_utils import _check_instance

# -----------------------------------------------------------------------------
# Test data location (symlinked by conftest.py)
# -----------------------------------------------------------------------------
DATADIR = "./oceanspy/tests/Data/"

# -----------------------------------------------------------------------------
# Pure type-checking behavior
# -----------------------------------------------------------------------------


def test_check_instance_accepts_builtins():
    _check_instance(
        {"a": 1, "b": True, "c": "hello"},
        {"a": "int", "b": "bool", "c": "str"},
    )


def test_check_instance_accepts_type_none():
    _check_instance({"x": None}, {"x": "type(None)"})


def test_check_instance_accepts_list_of_alternatives():
    _check_instance({"x": None}, {"x": ["type(None)", "str"]})
    _check_instance({"x": "abc"}, {"x": ["type(None)", "str"]})


def test_check_instance_accepts_tuple_string_syntax():
    _check_instance({"x": 1}, {"x": "(float, int, bool)"})
    _check_instance({"x": 1.5}, {"x": "(float, int, bool)"})
    _check_instance({"x": True}, {"x": "(float, int, bool)"})


def test_check_instance_accepts_dotted_numpy_types():
    _check_instance({"arr": np.zeros((3, 2))}, {"arr": "numpy.ndarray"})


def test_check_instance_accepts_numpy_scalartype():
    _check_instance({"x": np.float64(1.0)}, {"x": "numpy.ScalarType"})
    _check_instance({"x": 3}, {"x": "numpy.ScalarType"})


def test_check_instance_single_spec_applies_to_all_keys():
    _check_instance({"a": 1, "b": 2}, "int")


def test_check_instance_raises_typeerror_with_key_name():
    with pytest.raises(TypeError) as exc:
        _check_instance({"x": "not an int"}, {"x": "int"})

    assert "`x` must be" in str(exc.value)


# -----------------------------------------------------------------------------
# Integration tests: xarray + OceanDataset
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oceandataset():
    """
    Load a real OceanDataset from test data.
    """
    return open_oceandataset.from_netcdf(f"{DATADIR}MITgcm_rect_nc.nc")


def test_check_instance_accepts_xarray_dataset(oceandataset):
    ds = oceandataset.dataset

    # sanity check
    assert isinstance(ds, xr.Dataset)

    _check_instance({"dataset": ds}, "xarray.Dataset")


def test_check_instance_accepts_oceanspy_oceandataset(oceandataset):
    _check_instance({"od": oceandataset}, "oceanspy.OceanDataset")


def test_check_instance_multiple_objects(oceandataset):
    _check_instance(
        {
            "od": oceandataset,
            "dataset": oceandataset.dataset,
        },
        {
            "od": "oceanspy.OceanDataset",
            "dataset": "xarray.Dataset",
        },
    )
