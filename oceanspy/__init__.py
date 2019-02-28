__author__ = """Mattia Almansi"""
__email__ = 'mattia.almansi@jhu.edu'
__version__ = '0.0.10'

from ._oceandataset import OceanDataset
from . import open_oceandataset, subsample, compute, plot, utils

DEFAULT_PARAMETERS = {'rSphere' : 6.371E3,                # km None: cartesian
                      'eq_state': 'jmd95',                # jmd95, mdjwf
                      'rho0'    : 1027,                   # kg/m^3  TODO: None: compute volume weighted average
                      'g'       : 9.81,                   # m/s^2
                      'eps_nh'  : 0,                      # 0 is hydrostatic
                      'omega'   : 7.292123516990375E-05,  # rad/s
                      'c_p'     : 3.986E3                 # specific heat [J/kg/K]
                      }

AVAILABLE_PARAMETERS = {'eq_state': ['jmd95', 'mdjwf']} 


