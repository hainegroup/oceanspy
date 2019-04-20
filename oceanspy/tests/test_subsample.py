# TODO: cartesian, and Xp1 Yp1 right are not tested.
import pytest
import numpy as np
import xarray as xr
from numpy.testing import assert_array_equal

# From OceanSpy
from oceanspy import open_oceandataset, OceanDataset

# Directory
Datadir = './oceanspy/tests/Data/'

# Test oceandataset
MITgcm_curv_nc = open_oceandataset.from_netcdf('{}MITgcm_curv_nc.nc'
                                               ''.format(Datadir))
MITgcm_rect_bin = open_oceandataset.from_netcdf('{}MITgcm_rect_bin.nc'
                                                ''.format(Datadir))
MITgcm_rect_nc = open_oceandataset.from_netcdf('{}MITgcm_rect_nc.nc'
                                               ''.format(Datadir))

# =======
# CUTOUT
# =======
od = MITgcm_curv_nc
moor_od = OceanDataset(od.dataset.expand_dims('mooring'))
Ywarn = od.dataset['YG'].min()-1
Xwarn = od.dataset['XG'].min()-1
Zwarn = od.dataset['Zp1'].min()-1
Twarn = od.dataset['time'].min()-1
YRange = [od.dataset['YG'].min(), od.dataset['YG'].max()]
XRange = [od.dataset['XG'].min(), od.dataset['XG'].max()]
ZRange = [od.dataset['Zp1'].min(), od.dataset['Zp1'].max()]
timeRange = [od.dataset['time'].values[0], od.dataset['time'].values[-1]]
dropAxes = ['Y', 'X', 'Z', 'time']
dropwarn = 'mooring'


@pytest.mark.parametrize("od, YRange, XRange, ZRange, timeRange, dropAxes",
                         [(od, YRange, XRange, ZRange, timeRange, dropwarn),
                          (od, Ywarn, XRange, ZRange, timeRange, ['time']),
                          (od, YRange, Xwarn, ZRange, timeRange, True),
                          (od, YRange, XRange, Zwarn, timeRange, True),
                          (od, YRange, XRange, ZRange, Twarn, True),
                          (moor_od, YRange, XRange, ZRange, timeRange, True)])
def test_cutout_warnings(od, YRange, XRange, ZRange, timeRange, dropAxes):
    with pytest.warns(UserWarning):
        od.subsample.cutout(XRange=XRange,
                            YRange=YRange,
                            ZRange=ZRange,
                            timeRange=timeRange,
                            dropAxes=dropAxes)


od = MITgcm_curv_nc


@pytest.mark.parametrize("od, varList, XRange, timeFreq",
                         [(od, np.zeros((2, 2)), None, None),
                          (od, None, None, 1),
                          (od, None, np.zeros((2, 2)), None)])
def test_cutout_type_errors(od, varList, XRange, timeFreq):
    with pytest.raises(TypeError):
        od.subsample.cutout(varList=varList,
                            XRange=XRange,
                            timeFreq=timeFreq)


od = MITgcm_curv_nc


@pytest.mark.parametrize("od, varList, sampMethod",
                         [(od, 'wrong', 'snapshot'),
                          (od, None, 'wrong')])
def test_cutout_value_errors(od, varList, sampMethod):
    with pytest.raises(ValueError):
        od.subsample.cutout(varList=varList, sampMethod=sampMethod)


@pytest.mark.parametrize("od", [MITgcm_curv_nc,
                                MITgcm_curv_nc.set_grid_periodic(['X', 'Y']),
                                MITgcm_rect_bin])
@pytest.mark.parametrize("dropAxes", [True, False])
@pytest.mark.parametrize("add_Hbdr", [True, False, 1])
def test_horizontal_cutout(od, dropAxes, add_Hbdr):

    # Cover both case first inde and any index
    for i in range(2):
        if i == 0:
            iX = 0
            iY = 0
        elif i == 1:
            iX = np.random.choice(range(len(od.dataset['Xp1'])))
            iY = np.random.choice(range(len(od.dataset['Yp1'])))
        XRange = od.dataset['XG'].isel(Xp1=iX, Yp1=iY)
        YRange = od.dataset['YG'].isel(Xp1=iX, Yp1=iY)
        new_od = od.subsample.cutout(XRange=XRange,
                                     YRange=YRange,
                                     dropAxes=dropAxes,
                                     add_Hbdr=add_Hbdr)
        if dropAxes is True and add_Hbdr is False:
            assert (len(new_od.dataset['Xp1']) ==
                    len(new_od.dataset['Yp1']) ==
                    len(new_od.dataset['X']) ==
                    len(new_od.dataset['Y']) == 1)
        elif dropAxes is False:
            assert len(new_od.dataset['Xp1'])-len(new_od.dataset['X']) == 1
            assert len(new_od.dataset['Yp1'])-len(new_od.dataset['Y']) == 1


XRange_mask = [MITgcm_curv_nc.dataset['XG'].isel(Xp1=3, Yp1=3).values,
               MITgcm_curv_nc.dataset['XG'].isel(Xp1=-3, Yp1=-3).values]
YRange_mask = [MITgcm_curv_nc.dataset['YG'].isel(Xp1=3, Yp1=3).values,
               MITgcm_curv_nc.dataset['YG'].isel(Xp1=-3, Yp1=-3).values]


@pytest.mark.parametrize("od", [MITgcm_curv_nc])
@pytest.mark.parametrize("mask_outside", [True, False])
@pytest.mark.parametrize("XRange", [XRange_mask, None])
@pytest.mark.parametrize("YRange", [YRange_mask, None])
def test_horizontal_mask(od, mask_outside, XRange, YRange):
    new_od = od.subsample.cutout(XRange=XRange,
                                 YRange=YRange,
                                 mask_outside=mask_outside)

    if XRange is not None and YRange is not None:
        with pytest.raises(ValueError):
            od.subsample.cutout(XRange=np.mean(XRange),
                                YRange=np.mean(YRange),
                                mask_outside=mask_outside)
    if mask_outside is True and (XRange is not None or YRange is not None):
        assert np.isnan(new_od._ds['Temp'].values).any()
    elif mask_outside is False:
        assert not np.isnan(new_od._ds['Temp'].values).any()


@pytest.mark.parametrize("od", [MITgcm_rect_bin])
@pytest.mark.parametrize("dropAxes", [True, False])
@pytest.mark.parametrize("add_Vbdr", [True, False, 1])
def test_vertical_cutout(od, dropAxes, add_Vbdr):
    # Cover both case first index and any index
    for i in range(2):
        if i == 0:
            ZRange = 0
        elif i == 1:
            ZRange = od.dataset['Zp1'].mean()
        new_od = od.subsample.cutout(ZRange=ZRange,
                                     dropAxes=dropAxes,
                                     add_Vbdr=add_Vbdr)
        if dropAxes is True and add_Vbdr is False:
            assert (len(new_od.dataset['Zp1']) ==
                    len(new_od.dataset['Z']) ==
                    len(new_od.dataset['Zu']) ==
                    len(new_od.dataset['Zl']) == 1)
        elif dropAxes is False:
            assert (len(new_od.dataset['Zp1'])-1 ==
                    len(new_od.dataset['Z']) ==
                    len(new_od.dataset['Zu']) ==
                    len(new_od.dataset['Zl']))
            assert (new_od.dataset['Zp1'].isel(Zp1=slice(None, -1)).values ==
                    new_od.dataset['Zl'].values).all()
            assert (new_od.dataset['Zp1'].isel(Zp1=slice(1, None)).values ==
                    new_od.dataset['Zu'].values).all()
    # Test Z, Zp1 only
    new_ds = xr.Dataset({var: od.dataset[var]
                         for var in od.dataset.variables
                         if 'Z' in od.dataset[var].dims
                         or 'Zp1' in od.dataset[var].dims})
    new_ds.attrs = od.dataset.attrs
    new_od = OceanDataset(new_ds)
    new_od = new_od.subsample.cutout(ZRange=ZRange,
                                     dropAxes=dropAxes,
                                     add_Vbdr=add_Vbdr)


with pytest.warns(UserWarning):
    tomerge = MITgcm_rect_bin.dataset['time'].astype(int)
    od_indextime = MITgcm_rect_bin.merge_into_oceandataset(tomerge,
                                                           overwrite=True)


@pytest.mark.parametrize("od", [MITgcm_rect_bin, od_indextime])
@pytest.mark.parametrize("dropAxes", [True, False])
def test_time_cutout(od, dropAxes):
    # Cover both case first and last index and any index
    for i in range(3):
        if i == 0:
            timeRange = od.dataset['time'].isel(time=0)
        elif i == 1:
            timeRange = od.dataset['time'].isel(time=-1)
        elif i == 2:
            tt = int(len(od.dataset['time'])/2)
            timeRange = od.dataset['time'].isel(time=tt)
        new_od = od.subsample.cutout(timeRange=timeRange,
                                     dropAxes=dropAxes)

        if dropAxes is True:
            assert (len(new_od.dataset['time']) ==
                    len(new_od.dataset['time_midp']))
        else:
            assert (len(new_od.dataset['time']) ==
                    len(new_od.dataset['time_midp'])+1)


@pytest.mark.parametrize("od", [MITgcm_rect_nc])
@pytest.mark.parametrize("timeFreq", ['6H', '12H', '8H', '1D'])
@pytest.mark.parametrize("sampMethod", ['snapshot', 'mean'])
def test_time_resampling(od, timeFreq, sampMethod):
    # Warning due to tiset(['1, 20'])me_midp
    with pytest.warns(UserWarning):
        new_od = od.subsample.cutout(timeFreq=timeFreq,
                                     sampMethod=sampMethod)
    if timeFreq == '12H':
        assert (len(new_od.dataset['time']) ==
                np.ceil(len(od.dataset['time'])/2))
    elif timeFreq == '6H':
        assert (len(new_od.dataset['time']) ==
                len(od.dataset['time']))
    elif timeFreq == '1D' and sampMethod == 'mean':
        with pytest.warns(RuntimeWarning):
            # Test time only
            new_ds = new_od.dataset
            new_ds = new_ds.drop('time_midp')
            new_od = OceanDataset(new_ds)
            new_od.subsample.cutout(timeFreq='2D',
                                    sampMethod=sampMethod)


@pytest.mark.parametrize("od", [MITgcm_rect_bin])
@pytest.mark.parametrize("varList", [['X'], ['XC'], ['S'], ['X', 'XC', 'S']])
def test_reduce_variables(od, varList):
    new_od = od.subsample.cutout(varList=varList)
    assert (set(new_od.dataset.dims) - set(od.dataset.dims)) == set([])
    assert (set(new_od.dataset.coords) - set(od.dataset.coords)) == set([])
    assert ((set(varList)-set(od.dataset.dims)-set(od.dataset.coords)) ==
            set(new_od.dataset.data_vars))


# =======
# MOORING
# =======
@pytest.mark.parametrize("od", [MITgcm_rect_nc])
@pytest.mark.parametrize("cartesian", [True, False])
@pytest.mark.parametrize("kwargs", [{}, {'YRange': None,
                                         'XRange': None,
                                         'add_Hbdr': True}])
def test_mooring(od, cartesian, kwargs):

    this_od = od
    if cartesian:
        this_od = this_od.set_parameters({'rSphere': None})

    Xmoor = [this_od.dataset['XC'].min().values,
             this_od.dataset['XC'].max().values]
    Ymoor = [this_od.dataset['YC'].min().values,
             this_od.dataset['YC'].max().values]
    new_od = this_od.subsample.mooring_array(Xmoor=Xmoor,
                                             Ymoor=Ymoor,
                                             **kwargs)

    with pytest.raises(ValueError):
        new_od.subsample.mooring_array(Xmoor=Xmoor, Ymoor=Ymoor)

    for index in [0, -1]:
        assert new_od.dataset['XC'].isel(mooring=index).values == Xmoor[index]
        assert new_od.dataset['YC'].isel(mooring=index).values == Ymoor[index]

    checkX = new_od.grid.diff(new_od.dataset['XC'], 'mooring')
    checkY = new_od.grid.diff(new_od.dataset['YC'], 'mooring')
    assert all((checkX*checkY).values == 0)
    assert all((checkX+checkY).values != 0)
    assert set(['mooring', 'mooring_midp']).issubset(new_od.dataset.dims)
    assert len(new_od.dataset['X']) == len(new_od.dataset['Y']) == 1
    assert len(new_od.dataset['Xp1']) == len(new_od.dataset['Yp1']) == 2
    assert 'mooring_dist' in new_od.dataset.coords


# =======
# SURVEY
# =======
@pytest.mark.parametrize("od", [MITgcm_rect_nc])
@pytest.mark.parametrize("cartesian", [True, False])
@pytest.mark.parametrize("delta", [None, 2])
@pytest.mark.parametrize("kwargs", [{}, {'YRange': None,
                                         'XRange': None,
                                         'add_Hbdr': True}])
def test_survey(od, cartesian, delta, kwargs):

    this_od = od
    if cartesian:
        this_od = this_od.set_parameters({'rSphere': None})

    Xsurv = [this_od.dataset['XC'].min().values,
             this_od.dataset['XC'].mean().values,
             this_od.dataset['XC'].max().values]
    Ysurv = [this_od.dataset['YC'].min().values,
             this_od.dataset['YC'].mean().values,
             this_od.dataset['YC'].max().values]

    if cartesian:
        with pytest.warns(UserWarning):
            if delta is not None:
                with pytest.raises(IndexError):
                    new_od = this_od.subsample.survey_stations(Xsurv=Xsurv,
                                                               Ysurv=Ysurv,
                                                               delta=delta,
                                                               **kwargs)
            else:
                new_od = this_od.subsample.survey_stations(Xsurv=Xsurv,
                                                           Ysurv=Ysurv,
                                                           delta=delta,
                                                           **kwargs)
    else:
        new_od = this_od.subsample.survey_stations(Xsurv=Xsurv,
                                                   Ysurv=Ysurv,
                                                   delta=delta,
                                                   **kwargs)
        with pytest.raises(ValueError):
            new_od.subsample.survey_stations(Xsurv=Xsurv,
                                             Ysurv=Ysurv,
                                             delta=delta,
                                             **kwargs)

        for index in [0, -1]:
            XC = np.float32(new_od.dataset['XC'].isel(station=index).values)
            YC = np.float32(new_od.dataset['YC'].isel(station=index).values)
            assert XC == np.float32(Xsurv[index])
            assert YC == np.float32(Ysurv[index])
        if delta is None:
            assert len(new_od.dataset['station']) == len(Xsurv) == len(Ysurv)
        else:
            assert len(new_od.dataset['station']) >= len(Xsurv) == len(Ysurv)
        assert set(['station', 'station_midp']).issubset(new_od.dataset.dims)
        assert all([dim not in new_od.dataset.dims
                    for dim in ['X', 'Xp1', 'Y', 'Yp1']])
        assert 'station_dist' in new_od.dataset.coords
        assert 'station' in new_od.grid_coords
        new_od.grid


# =========
# PARTICLES
# =========
od = MITgcm_rect_nc
moor_od = OceanDataset(od.dataset.expand_dims('mooring'))
n_parts = 10
times = od.dataset['time']
Ypart = np.empty((len(times), n_parts))
Xpart = np.empty((len(times), n_parts))
Zpart = np.zeros((len(times), n_parts))
for p in range(n_parts):
    Ypart[:, p] = np.random.choice(od.dataset['Y'], len(times))
    Xpart[:, p] = np.random.choice(od.dataset['X'], len(times))
    Zpart[:, p] = np.random.choice(od.dataset['Z'], len(times))


@pytest.mark.parametrize("od, Ypart, Xpart, Zpart, times",
                         [(od, 1, Xpart, Zpart, times),
                          (od, Ypart, Xpart, Zpart, times[0]),
                          (od, np.zeros((2, 2, 2)), Xpart, Zpart, times)])
def test_particle_errors(od, Ypart, Xpart, Zpart, times):
    with pytest.raises(TypeError):
        od.subsample.particle_properties(times=times,
                                         Ypart=Ypart, Xpart=Xpart, Zpart=Zpart)


@pytest.mark.parametrize("od",        [MITgcm_rect_nc])
@pytest.mark.parametrize("cartesian", [True, False])
@pytest.mark.parametrize("varList",   [None, ['Temp']])
@pytest.mark.parametrize("kwargs", [{}, {'timeRange': None,
                                         'YRange': None,
                                         'XRange': None,
                                         'ZRange': None,
                                         'add_Hbdr': True,
                                         'add_Vbdr': True}])
def test_particles(od, cartesian, varList, kwargs):

    this_od = od
    if cartesian:
        this_od = this_od.set_parameters({'rSphere': None})

    # Create 10 random paths
    times = this_od.dataset['time']
    n_parts = 10
    Ypart = np.empty((len(times), n_parts))
    Xpart = np.empty((len(times), n_parts))
    Zpart = np.zeros((len(times), n_parts))
    for p in range(n_parts):
        Ypart[:, p] = np.random.choice(this_od.dataset['Y'], len(times))
        Xpart[:, p] = np.random.choice(this_od.dataset['X'], len(times))

    # Extract particles
    # Warning due to time_midp
    with pytest.warns(UserWarning):
        new_od = this_od.subsample.particle_properties(times=times,
                                                       Ypart=Ypart,
                                                       Xpart=Xpart,
                                                       Zpart=Zpart,
                                                       varList=varList,
                                                       **kwargs)

    assert_array_equal(np.float32(new_od.dataset['XC'].values),
                       np.float32(Xpart))
    assert_array_equal(np.float32(new_od.dataset['YC'].values),
                       np.float32(Ypart))

    # Test without midp
    new_ds = xr.Dataset({var: this_od.dataset[var]
                         for var in this_od.dataset.variables
                         if 'time_midp' not in this_od.dataset[var].dims})
    new_ds.attrs = this_od.dataset.attrs
    new_od = OceanDataset(new_ds).set_grid_coords({'time': {'time': None}},
                                                  overwrite=False)
    new_od.subsample.particle_properties(times=times,
                                         Ypart=Ypart,
                                         Xpart=Xpart,
                                         Zpart=Zpart,
                                         varList=varList)
