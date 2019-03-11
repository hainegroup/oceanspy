import pytest
import xarray as xr
import copy
from .datasets import oceandatasets, MITgcmVarDims
from oceanspy.animate import *
from oceanspy import utils
from numpy.random import rand, uniform
from numpy.testing import assert_array_equal, assert_allclose
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Add variables
od_in = copy.copy(oceandatasets['MITgcm_rect_nc'])
od_in = od_in.subsample.cutout(timeRange=0)

# Add random values
varNeeded = ['Temp', 'S', 
             'HFacC', 'HFacW', 'HFacS',
             'rAz', 'rA', 'rAw', 'rAs',
             'dyC', 'dxC', 'dxF', 'dyF', 'dxG', 'dyG', 'dxV', 'dyU',
             'drF', 'drC',
             'U', 'V', 'W',
             'Depth']
ds_dict = {}
for name, dimList in MITgcmVarDims.items():
    if name not in varNeeded: continue
    dimSize = [len(od_in.dataset[dim]) for dim in dimList]
    ds_dict[name] = xr.DataArray(rand(*dimSize), dims=dimList)
ds_in = xr.Dataset(ds_dict)
od_in = od_in.merge_into_oceandataset(ds_in)

@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize("subs", [None, 'survey', 'mooring'])
@pytest.mark.parametrize("colorName", [None, 'Temp', 'U'])
@pytest.mark.parametrize("meanAxes",  [None, 'Z'])
def test_TS_diagram(subs, meanAxes, colorName):
    
    if subs is None:
        od2plot = od_in
        
    elif subs in ['survey', 'mooring']:
        # Get random coords
        X = od_in.dataset['XC'].stack(XY=('X', 'Y')).values
        Y = od_in.dataset['YC'].stack(XY=('X', 'Y')).values
        X = X[[0, -1]]
        Y = Y[[0, -1]]
        
        if subs=='survey':
            with pytest.warns(UserWarning):
                # Run survey
                od2plot = od_in.subsample.survey_stations(Xsurv=X, Ysurv=Y)
        elif subs=='mooring':
            od2plot = od_in.subsample.mooring_array(Xmoor=X, Ymoor=Y)
    
    # Check shortcut as well
    anim = od2plot.animate.TS_diagram(meanAxes=meanAxes, colorName=colorName, display=False)
    isinstance(anim, animation.FuncAnimation)
    
    anim = TS_diagram(od2plot, meanAxes=meanAxes, colorName=colorName, timeRange=0, display=False)
    isinstance(anim, animation.FuncAnimation)

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_horizontal_section():
    # Clear
    plt.clf()
    plt.cla()
    
    od2plot = od_in
    anim = od2plot.animate.horizontal_section(varName = 'Temp', meanAxes=['Z'], display=False)
    isinstance(anim, animation.FuncAnimation)
    
    anim = horizontal_section(od2plot, varName = 'Temp', meanAxes=['Z'], display=False)
    isinstance(anim, animation.FuncAnimation)

@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize("subs", ['survey', 'mooring'])
def test_vertical_section_faceting(subs):
    # Clear
    plt.clf()
    plt.cla()
    
    if subs in ['survey', 'mooring']:
        # Get random coords
        X = od_in.dataset['XC'].stack(XY=('X', 'Y')).values
        Y = od_in.dataset['YC'].stack(XY=('X', 'Y')).values
        X = X[[0, -1]]
        Y = Y[[0, -1]]
        
        if subs=='survey':
            with pytest.warns(UserWarning):
                # Run survey
                od2plot = od_in.subsample.survey_stations(Xsurv=X, Ysurv=Y)
        elif subs=='mooring':
            od2plot = od_in.subsample.mooring_array(Xmoor=X, Ymoor=Y)
    
    anim = od2plot.animate.vertical_section(varName = 'Temp', display=False)
    isinstance(anim, animation.FuncAnimation)
    
    anim = vertical_section(od2plot, varName = 'Temp', display=False)
    isinstance(anim, animation.FuncAnimation)