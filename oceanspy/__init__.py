# -*- coding: utf-8 -*-

"""Top-level package for OceanSpy."""

__author__ = """Mattia Almansi"""
__email__ = 'mattia.almansi@jhu.edu'
__version__ = '0.0.4'

from ._useful_funcs import plot_mercator, disp_variables
from ._autogenerate import generate_ds
from ._cutout import Cutout