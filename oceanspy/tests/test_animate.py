# TODO: add tests for aliased datasets.

# Import modules
import pytest
import xarray as xr
import numpy as np

# From OceanSpy
from oceanspy import open_oceandataset
from oceanspy.animate import (TS_diagram,
                              horizontal_section, vertical_section)

# From matplotlib (keep it below oceanspy!)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Directory
Datadir = './oceanspy/tests/Data/'

# Test oceandataset
od = open_oceandataset.from_netcdf('{}MITgcm_rect_nc.nc'
                                   ''.format(Datadir))

# Create mooring, and survey
Xmoor = [od.dataset['XC'].min().values,
         od.dataset['XC'].max().values]
Ymoor = [od.dataset['YC'].min().values,
         od.dataset['YC'].max().values]
od_moor = od.subsample.mooring_array(Xmoor=Xmoor, Ymoor=Ymoor)

Xsurv = [od.dataset['XC'].min().values,
         od.dataset['XC'].mean().values,
         od.dataset['XC'].max().values]
Ysurv = [od.dataset['YC'].min().values,
         od.dataset['YC'].mean().values,
         od.dataset['YC'].max().values]
od_surv = od.subsample.survey_stations(Xsurv=Xsurv, Ysurv=Ysurv)


# ==========
# TS diagram
# ==========
@pytest.mark.parametrize("od_in,"
                         " cutout_kwargs, colorName, Tlim, Slim, cmap_kwargs",
                         [(od,
                           None, None, None, None, None),
                          (od,
                           {'ZRange': 0}, 'Temp', [0, 1], [0, 1],
                           {'robust': True})])
def test_anim_TSdiagram(od_in, cutout_kwargs, colorName,
                        Tlim, Slim, cmap_kwargs):
    plt.close()
    anim = TS_diagram(od_in, cutout_kwargs=cutout_kwargs,
                      colorName=colorName,
                      Tlim=Tlim, Slim=Slim, cmap_kwargs=cmap_kwargs)
    assert isinstance(anim, FuncAnimation)


@pytest.mark.parametrize("od_in", [od])
def test_anim_TSdiagram_warn(od_in):
    plt.close()
    dens = xr.DataArray(np.random.randn(2, 3),
                        coords={'Temp': np.arange(2),
                                'S': np.arange(3)},
                        dims=('Temp', 'S'))
    fig, ax = plt.subplots(1, 1)
    with pytest.warns(UserWarning):
        anim = TS_diagram(od_in, ax=ax, colorName='Temp',
                          FuncAnimation_kwargs={'repeat': False},
                          dens=dens)
    assert isinstance(anim, FuncAnimation)


# ==================
# Horizontal section
# ==================
@pytest.mark.parametrize("od_in, varName, meanAxes,"
                         " FuncAnimation_kwargs, cutout_kwargs",
                         [(od,
                           'Temp', True, None, None),
                          (od,
                           'Temp', True, {'repeat': False}, {'ZRange': 0})])
def test_anim_Hsection(od_in, varName, meanAxes,
                       FuncAnimation_kwargs, cutout_kwargs):
    plt.close()
    anim = horizontal_section(od_in, varName=varName, meanAxes=meanAxes,
                              FuncAnimation_kwargs=FuncAnimation_kwargs,
                              cutout_kwargs=cutout_kwargs)
    assert isinstance(anim, FuncAnimation)


# ================
# Vertical section
# ================
@pytest.mark.parametrize("od_in, varName",
                         [(od_moor, 'Temp'),
                          (od_surv, 'Temp')])
def test_anim_Vsection(od_in, varName):
    plt.close()
    anim = vertical_section(od_in, varName=varName)
    assert isinstance(anim, FuncAnimation)


@pytest.mark.parametrize("od_in", [od])
@pytest.mark.parametrize("subsampMethod", ['mooring_array', 'survey_stations'])
def test_anim_Vsection_subsamp(od_in, subsampMethod):
    plt.close()
    if subsampMethod == 'mooring_array':
        subsamp_kwargs = {'Xmoor': Xmoor, 'Ymoor': Ymoor}
    else:
        subsamp_kwargs = {'Xsurv': Xsurv, 'Ysurv': Ysurv}
    anim = vertical_section(od_in,
                            varName='Temp',
                            subsampMethod=subsampMethod,
                            subsamp_kwargs=subsamp_kwargs,
                            FuncAnimation_kwargs={'repeat': False})
    assert isinstance(anim, FuncAnimation)


def test_shortcuts():
    plt.close()
    anim = od.animate.TS_diagram(display=False)
    assert isinstance(anim, FuncAnimation)

    plt.close()
    anim = od.animate.horizontal_section(varName='Temp',
                                         func_kwargs={'intAxes': 'Z'},
                                         display=False)
    assert isinstance(anim, FuncAnimation)

    plt.close()
    anim = od_moor.animate.vertical_section(varName='Temp', display=False)
    assert isinstance(anim, FuncAnimation)
