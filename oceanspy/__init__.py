__author__ = """Mattia Almansi"""
__email__ = "mattia.almansi@noc.ac.uk"
__version__ = "0.1.0"

from ._oceandataset import OceanDataset  # noqa: F401
from .llc_rearrange import LLCtransformation  # noqa: F401
from . import (  # noqa: F401
    open_oceandataset,
    subsample,
    compute,
    plot,
    animate,
    utils,
    _ospy_utils,
)
import numpy as _np

DEFAULT_PARAMETERS = {
    "rSphere": 6.371e3,
    "eq_state": "jmd95",
    "rho0": 1027,
    "g": 9.81,
    "eps_nh": 0,
    "omega": 7.292123516990375e-05,
    "c_p": 3.986e3,
    "tempFrz0": 9.01e-02,
    "dTempFrz_dS": -5.75e-02,
}

PARAMETERS_DESCRIPTION = {
    "rSphere": "Radius of sphere for spherical polar"
    "or curvilinear grid (km)."
    "Set it None for cartesian grid.",
    "eq_state": "Equation of state.",
    "rho0": "Reference density (Boussinesq)  ( kg/m^3 )",
    "g": "Gravitational acceleration [m/s^2]",
    "eps_nh": "Non-Hydrostatic coefficient."
    "Set 0 for hydrostatic, 1 for non-hydrostatic.",
    "omega": "Angular velocity ( rad/s )",
    "c_p": "Specific heat capacity ( J/kg/K )",
    "tempFrz0": "Freezing temp. of sea water (intercept)",
    "dTempFrz_dS": "Freezing temp. of sea water (intercept)",
}

OCEANSPY_AXES = ["X", "Y", "Z", "face", "time", "mooring", "station"]


AVAILABLE_PARAMETERS = {"eq_state": ["jmd95", "mdjwf"]}

TYPE_PARAMETERS = {
    "rSphere": (type(None), _np.ScalarType),
    "eq_state": (str),
    "rho0": (_np.ScalarType),
    "g": (_np.ScalarType),
    "eps_nh": (_np.ScalarType),
    "omega": (_np.ScalarType),
    "c_p": (_np.ScalarType),
    "tempFrz0": (_np.ScalarType),
    "dTempFrz_dS": (_np.ScalarType),
}

SCISERVER_DATASETS = [
    "get_started",
    "EGshelfIIseas2km_ASR_full",
    "EGshelfIIseas2km_ASR_crop",
    "EGshelfIIseas2km_ERAI_6H",
    "EGshelfIIseas2km_ERAI_1D",
    "EGshelfSJsec500m_3H_hydro",
    "EGshelfSJsec500m_6H_hydro",
    "EGshelfSJsec500m_3H_NONhydro",
    "EGshelfSJsec500m_6H_NONhydro",
    "Arctic_Control",
    "KangerFjord",
    "LLC4320",
    "ECCO",
]
