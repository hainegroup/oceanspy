__author__ = """Mattia Almansi"""
__email__ = 'mattia.almansi@jhu.edu'
__version__ = '0.0.11'

from ._oceandataset import OceanDataset
from . import open_oceandataset, subsample, compute, plot, animate, utils
import numpy as _np

DEFAULT_PARAMETERS = {'rSphere'    : 6.371E3,                # km or None: cartesian
                      'eq_state'   : 'jmd95',                # jmd95, mdjwf
                      'rho0'       : 1027,                   # kg/m^3  TODO: None: compute volume weighted average
                      'g'          : 9.81,                   # m/s^2
                      'eps_nh'     : 0,                      # 0 is hydrostatic, 1 is non-hydrostatic
                      'omega'      : 7.292123516990375E-05,  # rad/s
                      'c_p'        : 3.986E3,                # specific heat [J/kg/K]
                      'tempFrz0'   : 0,#9.01E-02,               # freezing temp. of sea water (intercept)
                      'dTempFrz_dS': -5.75E-02,              # freezing temp. of sea water (slope)
                      }

OCEANSPY_AXES = ['X', 'Y', 'Z', 'time', 'mooring', 'station']

AVAILABLE_PARAMETERS = {'eq_state': ['jmd95', 'mdjwf']}

TYPE_PARAMETERS = {'rSphere'     : (type(None), _np.ScalarType),
                   'eq_state'    : (str),
                   'rho0'        : (_np.ScalarType),
                   'g'           : (_np.ScalarType),
                   'eps_nh'      : (_np.ScalarType),
                   'omega'       : (_np.ScalarType),
                   'c_p'         : (_np.ScalarType),
                   'tempFrz0'    : (_np.ScalarType),
                   'dTempFrz_dS' : (_np.ScalarType),}

