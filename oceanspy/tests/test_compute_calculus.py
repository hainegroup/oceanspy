# Import modules
import pytest
import xarray as xr
import numpy as np
from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_array_equal)

# From OceanSpy
from oceanspy import open_oceandataset, OceanDataset
from oceanspy.compute import (gradient, divergence, curl,
                              laplacian, weighted_mean, integral)

# Directory
Datadir = './oceanspy/tests/Data/'

# Test oceandataset
od4calc = open_oceandataset.from_netcdf('{}MITgcm_rect_nc.nc'
                                        ''.format(Datadir))
ds = od4calc.dataset
step = 1.E-2

# Space
for var in ['X', 'XC', 'XV']:
    ds[var] = xr.full_like(ds[var], step).cumsum(dim='X')
for var in ['Y', 'YC', 'YU']:
    ds[var] = xr.full_like(ds[var], step).cumsum(dim='Y')
for var in ['Xp1', 'XG', 'XU']:
    ds[var] = xr.full_like(ds[var], step).cumsum(dim='Xp1')
    ds[var] = ds[var] - step / 2
for var in ['Yp1', 'YG', 'YV']:
    ds[var] = xr.full_like(ds[var], step).cumsum(dim='Yp1')
    ds[var] = ds[var] - step / 2
ds['Z'] = xr.full_like(ds['Z'], - step).cumsum(dim='Z')
ds['Zp1'] = xr.full_like(ds['Zp1'], - step).cumsum(dim='Zp1') + step / 2
ds['Zl'] = xr.full_like(ds['Zl'], - step).cumsum(dim='Zl') + step / 2
ds['Zu'] = xr.full_like(ds['Zu'], - step).cumsum(dim='Zu') - step / 2

# Time
t0 = '1990-09-27T00:00:00'
T = []
for i in range(len(ds['time'])):
    T = T + [np.datetime64(t0) + np.timedelta64(int(i * step * 1.E3), 'ms')]
ds['time'] = np.array(T, dtype='datetime64')
T = []
for i in range(len(ds['time_midp'])):
    T = T + [np.datetime64(t0) + np.timedelta64(int(i * step * 1.E3), 'ms')]
ds['time_midp'] = (np.array(T, dtype='datetime64')
                   + np.timedelta64(int(0.5 * step * 1.E3), 'ms'))

# deltas
for var in ['drF', 'dxC', 'dyC', 'dxF', 'dyF', 'dxG', 'dyG', 'dxV', 'dyU']:
    ds[var] = xr.full_like(ds[var], step)
for var in ['rA', 'rAw', 'rAs', 'rAz']:
    ds[var] = xr.full_like(ds[var], step**2)
for var in ['HFacC', 'HFacW', 'HFacS']:
    ds[var] = xr.ones_like(ds[var])

# Recreate oceandataset
od4calc = OceanDataset(ds)

# Gradient
sinX = xr.zeros_like(od4calc.dataset['Temp']) + np.sin(od4calc.dataset['XC'])
sinY = xr.zeros_like(od4calc.dataset['Temp']) + np.sin(od4calc.dataset['YC'])
sinZ = xr.zeros_like(od4calc.dataset['Temp']) + np.sin(od4calc.dataset['Z'])
sintime = (xr.zeros_like(od4calc.dataset['Temp'])
           + np.sin((od4calc.dataset['time']
                     - od4calc.dataset['time'][0])
                    / np.timedelta64(1, 's')))

sintime.attrs = od4calc.dataset['time'].attrs
cosX = xr.zeros_like(od4calc.dataset['U']) + np.cos(od4calc.dataset['XU'])
cosY = xr.zeros_like(od4calc.dataset['V']) + np.cos(od4calc.dataset['YV'])
cosZ = xr.zeros_like(od4calc.dataset['W']) + np.cos(od4calc.dataset['Zl'])
costime = (xr.zeros_like(od4calc.dataset['oceSPtnd'])
           + np.cos((od4calc.dataset['time_midp']
                     - od4calc.dataset['time_midp'][0])
                    / np.timedelta64(1, 's')))

# Divergence and Curl
X = od4calc.dataset['X']
Y = od4calc.dataset['Y']
Z = od4calc.dataset['Z']
Xp1 = od4calc.dataset['Xp1']
Yp1 = od4calc.dataset['Yp1']
Zl = od4calc.dataset['Zl']
sinUZ, sinUY, sinUX = xr.broadcast(np.sin(Z), np.sin(Y), np.sin(Xp1))
sinVZ, sinVY, sinVX = xr.broadcast(np.sin(Z), np.sin(Yp1), np.sin(X))
sinWZ, sinWY, sinWX = xr.broadcast(np.sin(Zl), np.sin(Y), np.sin(X))
sin_ds = xr.Dataset({'sinX': sinX, 'sinY': sinY, 'sinZ': sinZ,
                     'sintime': sintime,
                     'cosX': cosX, 'cosY': cosY, 'cosZ': cosZ,
                     'costime': costime,
                     'sinUX': sinUX, 'sinUY': sinUY, 'sinUZ': sinUZ,
                     'sinVX': sinVX, 'sinVY': sinVY, 'sinVZ': sinVZ,
                     'sinWX': sinWX, 'sinWY': sinWY, 'sinWZ': sinWZ})
od4calc = od4calc.merge_into_oceandataset(sin_ds)


# GRADIENT
@pytest.mark.parametrize("od", [od4calc])
@pytest.mark.parametrize("axesList", [None, 'X', 'wrong'])
def test_gradient(od, axesList):

    varNameList = ['sinZ', 'sinY', 'sinX', 'sintime']
    if axesList == 'wrong':
        with pytest.raises(ValueError):
            gradient(od, varNameList=varNameList, axesList=axesList)
    else:
        grad_ds = gradient(od, varNameList=varNameList, axesList=axesList)
        if axesList is None:
            axesList = list(od.grid_coords.keys())

        # sin' = cos
        for varName in varNameList:
            for axis in axesList:
                gradName = 'd'+varName+'_'+'d'+axis
                var = grad_ds[gradName]
                if axis not in varName:
                    assert (var.min().values
                            == grad_ds[gradName].max().values
                            == 0)
                else:
                    check = od.dataset['cos'+axis].where(var)
                    mask = xr.where(np.logical_or(check.isnull(),
                                                  var.isnull()), 0, 1)
                    assert_allclose(var.where(mask, drop=True).values,
                                    check.where(mask, drop=True).values, 1.E-3)


@pytest.mark.parametrize("od", [od4calc])
def test_all_gradients(od):
    od_moor = od.subsample.mooring_array(Xmoor=[od.dataset['X'].min().values,
                                                od.dataset['X'].max().values],
                                         Ymoor=[od.dataset['Y'].min().values,
                                                od.dataset['Y'].max().values])
    with pytest.warns(UserWarning):
        X = od_moor.dataset['XC'].squeeze().values
        Y = od_moor.dataset['YC'].squeeze().values
        od_surv = od.subsample.survey_stations(Xsurv=X,
                                               Ysurv=Y)
    # Test all dimension
    DIMS = []
    VARS = []
    for var in od.dataset.data_vars:
        this_dims = list(od.dataset[var].dims)
        append = True
        for dims in DIMS:
            checks = [set(this_dims).issubset(set(dims)),
                      set(dims).issubset(set(this_dims))]
            if all(checks):
                append = False
                continue
        if append:
            VARS = VARS + [var]
            DIMS = DIMS + [list(this_dims)]
    gradient(od, varNameList=VARS)
    gradient(od_moor, varNameList=VARS)
    gradient(od_surv, varNameList=VARS)


# DIVERGENCE
@pytest.mark.parametrize("od, iName, jName, kName",
                         [(od4calc, None, None, 'Temp'),
                          (od4calc, None, 'Temp', None),
                          (od4calc, 'Temp', None, None),
                          (od4calc, None, None, None)])
def test_div_errors(od, iName, jName, kName):
    with pytest.raises(ValueError):
        divergence(od, iName=iName, jName=jName, kName=kName)


@pytest.mark.parametrize("od", [od4calc])
@pytest.mark.parametrize("varNameList", [[None, 'sinVY', 'sinWZ'],
                                         ['sinUX', None, 'sinWZ'],
                                         ['sinUX', 'sinVY', None],
                                         ['sinUX', 'sinVY', 'sinWZ']])
def test_divergence(od, varNameList):

    # Add units
    if None not in varNameList:
        for varName in varNameList:
            od._ds[varName].attrs['units'] = 'm/s'

    # Compute divergence
    dive_ds = divergence(od,
                         iName=varNameList[0],
                         jName=varNameList[1],
                         kName=varNameList[2])

    # sin' = cos
    for varName in varNameList:
        if varName is not None:
            axis = varName[-1]
            diveName = 'd'+varName+'_'+'d'+axis
            var = dive_ds[diveName]

            coords = {coord[0]: var[coord] for coord in var.coords}
            coords['Z'], coords['Y'], coords['X'] = xr.broadcast(coords['Z'],
                                                                 coords['Y'],
                                                                 coords['X'])
            check = np.cos(coords[axis])
            mask = xr.where(np.logical_or(check.isnull(), var.isnull()), 0, 1)

            # Assert using numpy
            var = var.where(mask, drop=True).values
            check = check.where(mask, drop=True).values
            assert_array_almost_equal(var, check, 1.E-3)


# CURL
@pytest.mark.parametrize("od, iName, jName, kName",
                         [(od4calc, None, None, None),
                          (od4calc, 'Temp', 'Temp', None),
                          (od4calc, 'Temp', None, 'Temp'),
                          (od4calc, None, 'Temp', 'Temp')])
def test_curl_errors(od, iName, jName, kName):
    with pytest.raises(ValueError):
        curl(od, iName=iName, jName=jName, kName=kName)


@pytest.mark.parametrize("od", [od4calc])
def test_curl(od):

    velocities = [[None, 'sinVZ', 'sinWY'],
                  ['sinUZ', None, 'sinWX'],
                  ['sinUY', 'sinVX', None],
                  ['sinUY', 'sinVX', 'sinWX']]

    for _, vels in enumerate(velocities):
        # Add units
        if None not in vels:
            for varName in vels:
                od._ds[varName].attrs['units'] = 'm/s'
        curl_ds = curl(od, iName=vels[0], jName=vels[1], kName=vels[2])

        # sin' = cos
        for var in curl_ds.data_vars:
            var = curl_ds[var]

        coords = {coord[0]: var[coord]
                  for coord in var.coords}
        coords['Z'], coords['Y'], coords['X'] = xr.broadcast(coords['Z'],
                                                             coords['Y'],
                                                             coords['X'])

        terms = var.name.split('-')
        for i, term in enumerate(terms):
            axis = term[-1]
            terms[i] = np.cos(coords[axis])
        check = terms[0] - terms[1]
        mask = xr.where(np.logical_or(check.isnull(), var.isnull()), 0, 1)

        # Assert using numpy
        if None in vels:
            assert_array_almost_equal(check.where(mask, drop=True).values,
                                      var.where(mask, drop=True).values, 7)


# LAPLACIAN
@pytest.mark.parametrize("od", [od4calc])
@pytest.mark.parametrize("varNameList", ['Temp', 'U'])
@pytest.mark.parametrize("axesList",    ['wrong', 'X', None])
def test_laplacian(od, varNameList, axesList):
    if varNameList == 'U' or axesList == 'wrong':
        with pytest.raises(ValueError):
            laplacian(od, varNameList=varNameList, axesList=axesList)
    else:
        laplacian(od, varNameList=varNameList, axesList=axesList)


# MEAN
@pytest.mark.parametrize("od", [od4calc])
@pytest.mark.parametrize("varNameList", ['Temp'])
def test_weighted_mean(od, varNameList):
    w_mean_ds = weighted_mean(od, varNameList)
    var = w_mean_ds['w_mean_'+varNameList].values
    check = od.dataset[varNameList].mean().values
    assert var == check


# INTEGRAL
@pytest.mark.parametrize("od", [od4calc])
@pytest.mark.parametrize("varNameList", ['Temp'])
def test_integral(od, varNameList):
    int_ds = integral(od, varNameList)
    w_mean_ds = weighted_mean(od, varNameList)
    int_name = [var for var in int_ds.data_vars][0]
    var = int_ds[int_name]
    check = (w_mean_ds['w_mean_' + varNameList]
             * w_mean_ds['weight_' + varNameList].sum())
    assert var.values == check.values


# INTEGRAL and WEIGHTED MEAN OPTIONS
@pytest.mark.parametrize("od", [od4calc])
def test_all_integrals(od):
    od_moor = od.subsample.mooring_array(Xmoor=[od.dataset['X'].min().values,
                                                od.dataset['X'].max().values],
                                         Ymoor=[od.dataset['Y'].min().values,
                                                od.dataset['Y'].max().values])
    # Test no units
    od_moor.dataset['Temp'].attrs = []
    with pytest.warns(UserWarning):
        X = od_moor.dataset['XC'].squeeze().values
        Y = od_moor.dataset['YC'].squeeze().values
        od_surv = od.subsample.survey_stations(Xsurv=X,
                                               Ysurv=Y)
    # Test all dimension
    DIMS = []
    VARS = []
    for var in od.dataset.data_vars:
        this_dims = list(od.dataset[var].dims)
        append = True
        for dims in DIMS:
            checks = [set(this_dims).issubset(set(dims)),
                      set(dims).issubset(set(this_dims))]
            if all(checks):
                append = False
                continue
        if append:
            VARS = VARS + [var]
            DIMS = DIMS + [list(this_dims)]
    integral(od, varNameList=VARS)
    integral(od_moor, varNameList=VARS)
    integral(od_surv, varNameList=VARS)


@pytest.mark.parametrize("od", [od4calc])
@pytest.mark.parametrize("varNameList", ['Temp', 'U', 'V', 'W', 'momVort3'])
@pytest.mark.parametrize("axesList", ['X', 'Y', 'time', 'Z', 'wrong'])
@pytest.mark.parametrize("storeWeights", [True, False])
def test_int_mean_options(od, varNameList, axesList, storeWeights):
    if axesList == 'wrong':
        with pytest.raises(ValueError):
            weighted_mean(od,
                          varNameList=varNameList,
                          axesList=axesList,
                          storeWeights=storeWeights)
    else:
        weighted_mean(od,
                      varNameList=varNameList,
                      axesList=axesList,
                      storeWeights=storeWeights)


# Test shortcuts
@pytest.mark.parametrize("od_in", [od4calc])
def test_shortcuts(od_in):

    # Only use some variables
    list_calc = ['Temp', 'U', 'V', 'W',
                 'HFacC', 'HFacW', 'HFacS',
                 'drC', 'drF',
                 'dxC', 'dyC',
                 'dxF', 'dyF',
                 'dxG', 'dyG',
                 'dxV', 'dyU',
                 'rA', 'rAw', 'rAs', 'rAz']
    od_in = od_in.subsample.cutout(varList=list_calc)

    # Gradient
    ds_out = gradient(od_in)
    od_out = od_in.compute.gradient()
    ds_out_IN_od_out(ds_out, od_out)

    # Divergence
    ds_out = divergence(od_in, iName='U', jName='V', kName='W')
    od_out = od_in.compute.divergence(iName='U', jName='V', kName='W')
    ds_out_IN_od_out(ds_out, od_out)

    # Curl
    ds_out = curl(od_in, iName='U', jName='V', kName='W')
    od_out = od_in.compute.curl(iName='U', jName='V', kName='W')
    ds_out_IN_od_out(ds_out, od_out)

    # Laplacian
    ds_out = laplacian(od_in, 'Temp')
    od_out = od_in.compute.laplacian(varNameList='Temp')
    ds_out_IN_od_out(ds_out, od_out)

    # Weighted mean
    ds_out = weighted_mean(od_in)
    od_out = od_in.compute.weighted_mean()
    ds_out_IN_od_out(ds_out, od_out)

    # Integral
    ds_out = integral(od_in)
    od_out = od_in.compute.integral()
    ds_out_IN_od_out(ds_out, od_out)


def ds_out_IN_od_out(ds_out, od_out):
    for var in ds_out.data_vars:
        assert_array_equal(od_out.dataset[var].values, ds_out[var].values)
