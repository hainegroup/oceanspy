__author__ = """Mattia Almansi"""
__email__ = 'mattia.almansi@jhu.edu'
__version__ = '0.0.11'

from ._oceandataset import OceanDataset  # noqa: F401
from . import (open_oceandataset, subsample, compute,  # noqa: F401
               plot, animate, utils, _ospy_utils)  # noqa: F401
import numpy as _np  # noqa: F401

DEFAULT_PARAMETERS = {'rSphere': 6.371E3,  # km or None: cartesian
                      # jmd95, mdjwf
                      'eq_state': 'jmd95',
                      # kg/m^3  TODO: None: compute volume weighted average
                      'rho0': 1027,
                      # m/s^2
                      'g': 9.81,
                      # 0 is hydrostatic, 1 is non-hydrostatic
                      'eps_nh': 0,
                      # rad/s
                      'omega': 7.292123516990375E-05,
                      # specific heat [J/kg/K]
                      'c_p': 3.986E3,
                      # freezing temp. of sea water (intercept)
                      'tempFrz0': 9.01E-02,
                      # freezing temp. of sea water (slope)
                      'dTempFrz_dS': -5.75E-02,
                      }

OCEANSPY_AXES = ['X', 'Y', 'Z', 'time', 'mooring', 'station']

AVAILABLE_PARAMETERS = {'eq_state': ['jmd95', 'mdjwf']}

TYPE_PARAMETERS = {'rSphere': (type(None), _np.ScalarType),
                   'eq_state': (str),
                   'rho0': (_np.ScalarType),
                   'g': (_np.ScalarType),
                   'eps_nh': (_np.ScalarType),
                   'omega': (_np.ScalarType),
                   'c_p': (_np.ScalarType),
                   'tempFrz0': (_np.ScalarType),
                   'dTempFrz_dS': (_np.ScalarType)}

SCISERVER_DATASETS = ['get_started',
                      'EGshelfIIseas2km_ASR_full',
                      'EGshelfIIseas2km_ASR_crop',
                      'EGshelfIIseas2km_ERAI_6H',
                      'EGshelfIIseas2km_ERAI_1D',
                      'EGshelfSJsec500m_3H_hydro',
                      'EGshelfSJsec500m_6H_hydro',
                      'EGshelfSJsec500m_3H_NONhydro',
                      'EGshelfSJsec500m_6H_NONhydro',
                      'Arctic_Control',
                      'KangerFjord']
