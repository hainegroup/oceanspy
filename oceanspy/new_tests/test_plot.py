# Import modules
import pytest
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# From OceanSpy
from oceanspy import open_oceandataset
from oceanspy.plot import TS_diagram, time_series

# Directory
Datadir = './oceanspy/new_tests/Data/'

# Test oceandataset
od = open_oceandataset.from_netcdf('{}MITgcm_rect_nc.nc'
                                   ''.format(Datadir))

# Create mooring, sruvey, and particles
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

times = od.dataset['time']
n_parts = 10
Ypart = np.empty((len(times), n_parts))
Xpart = np.empty((len(times), n_parts))
Zpart = np.zeros((len(times), n_parts))
for p in range(n_parts):
    Ypart[:, p] = np.random.choice(od.dataset['Y'], len(times))
    Xpart[:, p] = np.random.choice(od.dataset['X'], len(times))
# Extract particles
# Warning due to time_midp
with pytest.warns(UserWarning):
    od_part = od.subsample.particle_properties(times=times,
                                               Ypart=Ypart,
                                               Xpart=Xpart,
                                               Zpart=Zpart)


# TS diagram
@pytest.mark.parametrize("od_in", [od])
@pytest.mark.parametrize("Tlim, Slim, dens",
                         [(None, [1], None),
                          ([1], None, None),
                          (None, None, xr.DataArray(np.random.randn(2, 3)))])
def test_TS_error(od_in, Tlim, Slim, dens):
    with pytest.raises(ValueError):
        TS_diagram(od_in, Tlim=Tlim, Slim=Slim, dens=dens)


# Test settings
@pytest.mark.parametrize("od_in", [od])
@pytest.mark.parametrize("Tlim", [[0, 1]])
@pytest.mark.parametrize("Slim", [[0, 1]])
@pytest.mark.parametrize("ax", [True])
@pytest.mark.parametrize("cutout_kwargs", [True])
@pytest.mark.parametrize("cmap_kwargs", [{'robust': True}])
@pytest.mark.parametrize("contour_kwargs", [{'levels': 10}])
@pytest.mark.parametrize("clabel_kwargs", [{'fontsize': 10}])
@pytest.mark.parametrize("dens", [xr.DataArray(np.random.randn(2, 3),
                                               coords={'Temp': np.arange(2),
                                                       'S': np.arange(3)},
                                               dims=('Temp', 'S'))])
@pytest.mark.parametrize("plotFreez", [False])
def test_TS_diagram_set(od_in, Tlim, Slim, ax,
                        cutout_kwargs, cmap_kwargs, contour_kwargs,
                        clabel_kwargs, dens, plotFreez):

    if cutout_kwargs is True:
        cutout_kwargs = {'XRange': [od_in.dataset['XC'].min().values,
                                    od_in.dataset['XC'].max().values]}
    if ax is True:
        _, ax = plt.subplots(1, 1)

    ax = TS_diagram(od_in, Tlim=Tlim, Slim=Slim, ax=ax,
                    cutout_kwargs=cutout_kwargs, cmap_kwargs=cmap_kwargs,
                    contour_kwargs=contour_kwargs, clabel_kwargs=clabel_kwargs,
                    dens=dens, plotFreez=plotFreez)

    if Tlim is not None:
        assert ax.get_ylim() == tuple(Tlim)
    if Slim is not None:
        assert ax.get_xlim() == tuple(Slim)


# Test fields
@pytest.mark.parametrize("od_in", [od, od_moor, od_surv, od_part])
@pytest.mark.parametrize("meanAxes", [None, 'time'])
@pytest.mark.parametrize("colorName", [None, 'Temp', 'Depth', 'Eta', 'U', 'W'])
def test_TS_diagram_field(od_in, meanAxes, colorName):

    ax = TS_diagram(od_in, colorName=colorName, meanAxes=meanAxes)
    assert isinstance(ax, plt.Axes)


# Time series
@pytest.mark.parametrize("od_in", [od])
@pytest.mark.parametrize("varName, meanAxes, intAxes",
                         [('Depth', True, False),
                          ('Temp', None, False),
                          ('Temp', ['X'], False),
                          ('Temp', True, True)])
def test_timeSeries_error(od_in, varName, meanAxes, intAxes):
    if meanAxes is None:
        with pytest.raises(TypeError):
            time_series(od_in, varName=varName,
                        meanAxes=meanAxes, intAxes=intAxes)
    else:
        with pytest.raises(ValueError):
            time_series(od_in, varName=varName,
                        meanAxes=meanAxes, intAxes=intAxes)


@pytest.mark.parametrize("od_in", [od, od_moor, od_surv, od_part])
def test_timeSeries(od_in):
    cutout_kwargs = {'XRange': [od_in.dataset['XC'].min().values,
                                od_in.dataset['XC'].max().values]}
    ax = time_series(od_in, varName='Temp', intAxes=True,
                     cutout_kwargs=cutout_kwargs)
    assert isinstance(ax, plt.Axes)


@pytest.mark.parametrize("od_in", [od])
def test_shortcuts(od_in):

    ax = od_in.plot.TS_diagram()
    assert isinstance(ax, plt.Axes)

    ax = od_in.plot.time_series(varName='Temp', meanAxes=True)
    assert isinstance(ax, plt.Axes)
