__author__ = """Mattia Almansi"""
__email__ = 'mattia.almansi@jhu.edu'
__version__ = '0.0.10'

from ._oceandataset import OceanDataset
from . import open_oceandataset, subsample, utils

ospy_aliases = ['Y', 'X', 
                'Yp1', 'Xp1', 
                'Z', 'Zp1', 'Zu', 'Zl',
                'time', 'time_midp',
                'YC', 'XC', 'YG', 'XG', 'YU', 'XU', 'YV', 'XV']
ospy_parameters = ['rSphere']

"""
Parameters
rSphere: scalar in km (spherical) or None (cartesian)

"""