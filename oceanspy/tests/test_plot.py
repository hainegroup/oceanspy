import pytest
import xarray as xr
import copy
from .datasets import oceandatasets, MITgcmVarDims
from oceanspy.plot import *
from oceanspy import utils
from numpy.random import rand, uniform
from numpy.testing import assert_array_equal, assert_allclose
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Add variables
od_in = copy.copy(oceandatasets['MITgcm_rect_nc'])

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

od_in = od_in.subsample.cutout(timeRange=od_in.dataset['time'].isel(time=0))

@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize("subs", [None, 'survey', 'mooring'])
@pytest.mark.parametrize("colorName", [None, 'Temp', 'U'])
@pytest.mark.parametrize("meanAxes",  [None, 'time'])
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
    ax = TS_diagram(od2plot, meanAxes=meanAxes, colorName=colorName)
    assert isinstance(ax, plt.Axes)

@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize("varName",  ['Temp', 'U'])    
@pytest.mark.parametrize("subs", [None, 'survey', 'mooring'])
@pytest.mark.parametrize("meanAxes", [False, True])
@pytest.mark.parametrize("intAxes",  [False, True])
def test_time_series(varName, subs, meanAxes, intAxes):
    
    # Clear
    plt.clf()
    plt.cla()
    
    # Different subsampled
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
            
    # Fail when both True or False
    if meanAxes==intAxes:
        with pytest.raises(ValueError):
            ax = time_series(od2plot, varName = varName, meanAxes=meanAxes, intAxes=intAxes)
    else:
        if subs is None:
            ax = time_series(od2plot, varName = varName, meanAxes=meanAxes, intAxes=intAxes)
            assert isinstance(ax, plt.Axes)
            
@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize("meanAxes", [False, True])
@pytest.mark.parametrize("intAxes",  [False, True])
@pytest.mark.parametrize("contourName", [None, 'Depth', 'Temp'])
def test_horizontal_section(meanAxes, intAxes, contourName):
    
    # Clear
    plt.clf()
    plt.cla()
    
    od2plot = od_in
    # Fail when both True or False
    if meanAxes==intAxes:
        with pytest.raises(ValueError):
            ax = horizontal_section(od2plot, varName = 'Temp', meanAxes=meanAxes, intAxes=intAxes, contourName=contourName)
    else:
        ax = horizontal_section(od2plot, varName = 'Temp', meanAxes=meanAxes, intAxes=intAxes, contourName=contourName)
        assert isinstance(ax, plt.Axes)
        
@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize("subs", ['survey', 'mooring'])
@pytest.mark.parametrize("meanAxes", [False, True])
@pytest.mark.parametrize("intAxes",  [False, True])
@pytest.mark.parametrize("varName", ['Temp', 'U'])
@pytest.mark.parametrize("contourName", [None, 'Sigma0'])
def test_vertical_section(subs, varName, meanAxes, intAxes, contourName):
    
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
            
    # Fail when both True or False
    if meanAxes==intAxes==True:
        with pytest.raises(ValueError):
            ax = vertical_section(od2plot, varName = varName, meanAxes=meanAxes, intAxes=intAxes, contourName=contourName)
    else:
        ax = vertical_section(od2plot, varName =varName, meanAxes=meanAxes, intAxes=intAxes, contourName=contourName)
        if meanAxes is not False or intAxes is not False:
            assert isinstance(ax, plt.Axes)
        else:
            # Faceting
            assert isinstance(ax, xr.plot.facetgrid.FacetGrid)
            
@pytest.mark.filterwarnings('ignore::UserWarning')            
def test_horizontal_section_faceting():
    # Clear
    plt.clf()
    plt.cla()
    
    od2plot = od_in
    ax = horizontal_section(od2plot, varName = 'Temp', meanAxes=['Z'])
    assert isinstance(ax, xr.plot.facetgrid.FacetGrid)

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
    
    anim = vertical_section(od2plot, varName = 'Temp')
    assert isinstance(anim, xr.plot.facetgrid.FacetGrid)

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_shortcuts():
    
    # TS diagram
    plt.clf()
    plt.cla()
    ax = od_in.plot.TS_diagram()
    assert isinstance(ax, plt.Axes)
    
    # time_series
    plt.clf()
    plt.cla()
    ax = od_in.plot.time_series(varName = 'Temp', meanAxes=True)
    assert isinstance(ax, plt.Axes)
    
    # horizontal section
    plt.clf()
    plt.cla()
    ax = od_in.plot.horizontal_section(varName = 'Temp', meanAxes=True)
    assert isinstance(ax, plt.Axes)
    
    
    # vertical section
    plt.clf()
    plt.cla()
    
    # Get random coords
    X = od_in.dataset['XC'].stack(XY=('X', 'Y')).values
    Y = od_in.dataset['YC'].stack(XY=('X', 'Y')).values
    X = X[[0, -1]]
    Y = Y[[0, -1]]
    ax = od_in.subsample.survey_stations(Xsurv=X, Ysurv=Y).plot.vertical_section(varName = 'Temp', meanAxes=True)
    assert isinstance(ax, plt.Axes)
    