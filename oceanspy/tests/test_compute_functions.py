# Import modules
import pytest
import numpy as np
import xarray as xr
import oceanspy as ospy

# From OceanSpy
from oceanspy import open_oceandataset
from oceanspy import AVAILABLE_PARAMETERS
from oceanspy.compute import gradient, divergence, curl
from oceanspy.compute import (missing_horizontal_spacing,
                              potential_density_anomaly,
                              Brunt_Vaisala_frequency,
                              velocity_magnitude,
                              horizontal_velocity_magnitude,
                              vertical_relative_vorticity,
                              relative_vorticity,
                              kinetic_energy, eddy_kinetic_energy,
                              horizontal_divergence_velocity,
                              shear_strain, normal_strain,
                              Okubo_Weiss_parameter,
                              Ertel_potential_vorticity,
                              mooring_volume_transport,
                              survey_aligned_velocities,
                              heat_budget, salt_budget)

from numpy.testing import (assert_allclose,
                           assert_array_equal,
                           assert_almost_equal)

# Directory
Datadir = './oceanspy/tests/Data/'

# Create an oceandataset for testing calculus functions
od = open_oceandataset.from_netcdf('{}MITgcm_rect_nc.nc'
                                   ''.format(Datadir))

# Create an oceandataset for testing calculus functions
od_curv = open_oceandataset.from_netcdf('{}MITgcm_curv_nc.nc'
                                        ''.format(Datadir))

# Aliased od
ds = od.dataset
aliases = {var: var+'_alias' for var in ds.data_vars}
ds = ds.rename(aliases)
alias_od = ospy.OceanDataset(ds).set_aliases(aliases)

# Budgets
od_bdg = open_oceandataset.from_netcdf('{}budgets.nc'
                                       ''.format(Datadir))


@pytest.mark.parametrize("od_in", [od])
def test_missing_horizontal_spacing(od_in):

    # Compute
    od_in = od_in.subsample.cutout(varList=['dxC', 'dxG', 'dyC', 'dyG'])
    ds = missing_horizontal_spacing(od_in)
    for varName in ds.variables:
        var = ds[varName]
        check = od.dataset[varName]
        mask = xr.where(np.logical_or(check.isnull(), var.isnull()), 0, 1)
        check = check.where(mask, drop=True).values
        var = var.where(mask, drop=True).values
        assert_allclose(check, var, equal_nan=False)

    # Test shortcut
    od_in.compute.missing_horizontal_spacing()


@pytest.mark.parametrize("od_in", [od, alias_od])
@pytest.mark.parametrize("eq_state", AVAILABLE_PARAMETERS['eq_state'])
def test_potential_density_anomaly(od_in, eq_state):

    od_in = od_in.set_parameters({'eq_state': eq_state})

    # Compute Sigma0
    ds_out = potential_density_anomaly(od_in)
    assert ds_out['Sigma0'].attrs['units'] == 'kg/m^3'
    assert ds_out['Sigma0'].attrs['long_name'] == 'potential density anomaly'
    check_params(ds_out, 'Sigma0', ['eq_state'])

    # Check values
    Sigma0 = eval("ospy.utils.dens{}"
                  "(od_in._ds['S'].values, od_in._ds['Temp'].values, 0)"
                  "".format(od_in.parameters['eq_state']))
    assert_array_equal(ds_out['Sigma0'].values+1000, Sigma0)

    # Test shortcut
    od_out = od_in.compute.potential_density_anomaly()
    ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od, alias_od])
def test_Brunt_Vaisala_frequency(od_in):

    # Compute N2
    ds_out = Brunt_Vaisala_frequency(od_in)
    assert ds_out['N2'].attrs['units'] == 's^-2'
    assert ds_out['N2'].attrs['long_name'] == 'Brunt-Väisälä Frequency'
    check_params(ds_out, 'N2', ['g', 'rho0'])

    # Check values
    dSigma0_dZ = gradient(od_in, 'Sigma0', 'Z')
    dSigma0_dZ = dSigma0_dZ['dSigma0_dZ']
    assert_allclose(-dSigma0_dZ.values
                    * od_in.parameters['g']
                    / od_in.parameters['rho0'], ds_out['N2'].values)

    # Test shortcut
    od_out = od_in.compute.Brunt_Vaisala_frequency()
    ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od, alias_od])
def test_velocity_magnitude(od_in):

    # Compute vel
    ds_out = velocity_magnitude(od_in)
    assert ds_out['vel'].attrs['units'] == 'm/s'
    assert ds_out['vel'].attrs['long_name'] == 'velocity magnitude'

    # Check values
    U = (od_in._ds['U'].values[:, :, :, 1:]
         + od_in._ds['U'].values[:, :, :, :-1]) / 2
    V = (od_in._ds['V'].values[:, :, 1:, :]
         + od_in._ds['V'].values[:, :, :-1, :]) / 2
    W = (od_in._ds['W'].values[:, 1:, :, :]
         + od_in._ds['W'].values[:, :-1, :, :]) / 2
    vel = np.sqrt(U[:, :-1, :, :]**2+V[:, :-1, :, :]**2+W**2)
    assert_allclose(vel, ds_out['vel'].values[:, :-1, :, :])

    # Test shortcut
    od_out = od_in.compute.velocity_magnitude()
    ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od, alias_od])
def test_horizontal_velocity_magnitude(od_in):

    # Compute hor_vel
    ds_out = horizontal_velocity_magnitude(od_in)
    assert ds_out['hor_vel'].attrs['units'] == 'm/s'
    assert (ds_out['hor_vel'].attrs['long_name']
            == 'magnitude of horizontal velocity')

    # Check values
    U = (od_in._ds['U'].values[:, :, :, 1:]
         + od_in._ds['U'].values[:, :, :, :-1]) / 2
    V = (od_in._ds['V'].values[:, :, 1:, :]
         + od_in._ds['V'].values[:, :, :-1, :]) / 2
    hor_vel = np.sqrt(U**2+V**2)
    assert_allclose(hor_vel, ds_out['hor_vel'].values)

    # Test shortcut
    od_out = od_in.compute.horizontal_velocity_magnitude()
    ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od, alias_od])
def test_vertical_relative_vorticity(od_in):

    # Extract and remove zeta
    check = od_in._ds['momVort3']
    aliases = od.aliases
    if aliases is None:
        aliases = {}
    vortName = aliases.pop('momVort3', None)
    ds = od.dataset
    if vortName is None:
        ds = ds.drop('momVort3')
    else:
        ds = ds.drop(vortName)
    od_in = ospy.OceanDataset(ds).set_aliases(aliases)

    # Compute momVort3
    ds_out = vertical_relative_vorticity(od_in)
    var = ds_out['momVort3']

    # Mask and check
    mask = xr.where(np.logical_or(check.isnull(), var.isnull()), 0, 1)
    assert_allclose(check.where(mask, drop=True).values,
                    var.where(mask, drop=True).values, equal_nan=False)

    # Test shortcut
    od_out = od_in.compute.vertical_relative_vorticity()
    ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od, alias_od])
def test_relative_vorticity(od_in):

    # Extract and remove zeta
    check = od_in._ds['momVort3']
    aliases = od.aliases
    if aliases is None:
        aliases = {}
    vortName = aliases.pop('momVort3', None)
    ds = od.dataset
    if vortName is None:
        ds = ds.drop('momVort3')
    else:
        ds = ds.drop(vortName)
    od_in = ospy.OceanDataset(ds).set_aliases(aliases)

    # Compute momVort1, momVort2, momVort3
    ds_out = relative_vorticity(od_in)
    varName = 'momVort'
    for i in range(3):
        assert ds_out[varName+str(i+1)].attrs['units'] == 's^-1'
        long_name = '{}-component of relative vorticity'.format(chr(105+i))
        assert ds_out[varName+str(i+1)].attrs['long_name'] == long_name

    # Check values
    vort = curl(od_in, iName='U',  jName='V', kName='W')
    for i, curlName in enumerate(['dW_dY-dV_dZ',
                                  'dU_dZ-dW_dX',
                                  'dV_dX-dU_dY']):
        assert_allclose(vort[curlName].values, ds_out[varName+str(i+1)].values)
    var = ds_out['momVort3']

    # Mask and check
    mask = xr.where(np.logical_or(check.isnull(), var.isnull()), 0, 1)
    assert_allclose(check.where(mask, drop=True).values,
                    var.where(mask, drop=True).values, equal_nan=False)

    # Test shortcut
    od_out = od_in.compute.relative_vorticity()
    ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od, alias_od])
@pytest.mark.parametrize("eps_nh", [1, 0])
def test_kinetic_energy(od_in, eps_nh):

    od_in = od_in.set_parameters({'eps_nh': eps_nh})

    # Compute KE
    ds_out = kinetic_energy(od_in)
    assert ds_out['KE'].attrs['units'] == 'm^2 s^-2'
    assert ds_out['KE'].attrs['long_name'] == 'kinetic energy'
    check_params(ds_out, 'KE', ['eps_nh'])

    # Check values
    U = (od_in._ds['U'].values[:, :, :, 1:]
         + od_in._ds['U'].values[:, :, :, :-1]) / 2
    V = (od_in._ds['V'].values[:, :, 1:, :]
         + od_in._ds['V'].values[:, :, :-1, :]) / 2
    KE = (U**2+V**2)/2
    if eps_nh == 0:
        assert_allclose(KE, ds_out['KE'].values)
    else:
        assert list(ds_out['KE'].dims) == list(od_in._ds['Temp'].dims)
        with pytest.raises(AssertionError):
            assert_allclose(KE, ds_out['KE'].values)

    # Test shortcut
    od_out = od_in.compute.kinetic_energy()
    ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od, alias_od])
@pytest.mark.parametrize("eps_nh", [1, 0])
def test_eddy_kinetic_energy(od_in, eps_nh):

    od_in = od_in.set_parameters({'eps_nh': eps_nh})

    # Compute KE
    ds_out = eddy_kinetic_energy(od_in)
    assert ds_out['EKE'].attrs['units'] == 'm^2 s^-2'
    assert ds_out['EKE'].attrs['long_name'] == 'eddy kinetic energy'
    check_params(ds_out, 'EKE', ['eps_nh'])

    # Check values
    U = od_in._ds['U'] - od_in._ds['U'].mean('time')
    V = od_in._ds['V'] - od_in._ds['V'].mean('time')
    U = (U.values[:, :, :, 1:] + U.values[:, :, :, :-1]) / 2
    V = (V.values[:, :, 1:, :] + V.values[:, :, :-1, :]) / 2
    EKE = (U**2 + V**2) / 2
    if eps_nh == 0:
        assert_allclose(EKE, ds_out['EKE'].values)
    else:
        assert list(ds_out['EKE'].dims) == list(od_in._ds['Temp'].dims)
        with pytest.raises(AssertionError):
            assert_allclose(EKE, ds_out['EKE'].values)

    # Test shortcut
    od_out = od_in.compute.eddy_kinetic_energy()
    ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od, alias_od])
def test_horizontal_divergence_velocity(od_in):

    # Compute hor_div_vel
    ds_out = horizontal_divergence_velocity(od_in)
    assert ds_out['hor_div_vel'].attrs['units'] == 'm s^-2'
    long_name = 'horizontal divergence of the velocity field'
    assert ds_out['hor_div_vel'].attrs['long_name'] == long_name

    # Check values
    hor_div = divergence(od_in, iName='U',  jName='V')
    hor_div = hor_div['dU_dX'] + hor_div['dV_dY']
    assert_allclose(hor_div.values, ds_out['hor_div_vel'].values)

    # Test shortcut
    od_out = od_in.compute.horizontal_divergence_velocity()
    ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od, alias_od])
def test_shear_strain(od_in):

    # Compute s_strain
    ds_out = shear_strain(od_in)
    assert ds_out['s_strain'].attrs['units'] == 's^-1'
    assert ds_out['s_strain'].attrs['long_name'] == 'shear component of strain'

    # Does it make sense to test this just rewriting
    # the same equation in compute?
    # Here I'm just testing that it works....

    # Test shortcut
    od_out = od_in.compute.shear_strain()
    ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od, alias_od])
def test_normal_strain(od_in):

    # Compute n_strain
    ds_out = normal_strain(od_in)
    assert ds_out['n_strain'].attrs['units'] == 's^-1'
    long_name = 'normal component of strain'
    assert ds_out['n_strain'].attrs['long_name'] == long_name

    # Check values
    divs = divergence(od_in, iName='U', jName='V')
    assert_allclose((divs['dU_dX'] - divs['dV_dY']).values,
                    ds_out['n_strain'].values)

    # Test shortcut
    od_out = od_in.compute.normal_strain()
    ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od, alias_od])
def test_Okubo_Weiss_parameter(od_in):

    # Compute n_strain
    ds_out = Okubo_Weiss_parameter(od_in)

    assert ds_out['Okubo_Weiss'].attrs['units'] == 's^-2'
    assert ds_out['Okubo_Weiss'].attrs['long_name'] == 'Okubo-Weiss parameter'

    # Does it make sense to test this just rewriting
    # the same equation in compute?
    # Here I'm just testing that it works....

    # Test shortcut
    od_out = od_in.compute.Okubo_Weiss_parameter()
    ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od, alias_od])
@pytest.mark.parametrize("full", [False, True])
def test_Ertel_potential_vorticity(od_in, full):

    # Compute Ertel_PV
    ds_out = Ertel_potential_vorticity(od_in, full)
    assert ds_out['Ertel_PV'].attrs['units'] == 'm^-1 s^-1'
    assert ds_out['Ertel_PV'].attrs['long_name'] == 'Ertel potential vorticity'

    # Does it make sense to test this just rewriting
    # the same equation in compute?
    # Here I'm just testing that it works....

    # Test shortcut
    od_out = od_in.compute.Ertel_potential_vorticity(full=full)
    ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od, alias_od])
@pytest.mark.parametrize("mooring", [True, False])
@pytest.mark.parametrize("closed", [True, False])
@pytest.mark.parametrize("flippedX", [True, False])
@pytest.mark.parametrize("flippedY", [True, False])
def test_mooring_volume_transport(od_in, mooring,
                                  closed, flippedX, flippedY):

    if mooring is True:
        if not closed:
            X = [od_in.dataset['X'].min().values,
                 od_in.dataset['X'].max().values]
            Y = [od_in.dataset['Y'].min().values,
                 od_in.dataset['Y'].max().values]
            od_moor = od_in.subsample.mooring_array(Xmoor=X,
                                                    Ymoor=Y)
            if flippedX and not flippedY:
                X = [od_in.dataset['X'].max().values,
                     od_in.dataset['X'].min().values]
                Y = [od_in.dataset['Y'].min().values,
                     od_in.dataset['Y'].max().values]
                od_moor = od_in.subsample.mooring_array(Xmoor=X,
                                                        Ymoor=Y)
            elif flippedX and flippedY:
                X = [od_in.dataset['X'].max().values,
                     od_in.dataset['X'].min().values]
                Y = [od_in.dataset['Y'].max().values,
                     od_in.dataset['Y'].min().values]
                od_moor = od_in.subsample.mooring_array(Xmoor=X,
                                                        Ymoor=Y)
            elif flippedY:
                X = [od_in.dataset['X'].min().values,
                     od_in.dataset['X'].max().values]
                Y = [od_in.dataset['Y'].max().values,
                     od_in.dataset['Y'].min().values]
                od_moor = od_in.subsample.mooring_array(Xmoor=X,
                                                        Ymoor=Y)
        else:
            X = [od_in.dataset['X'].min().values,
                 od_in.dataset['X'].max().values,
                 od_in.dataset['X'].max().values,
                 od_in.dataset['X'].min().values,
                 od_in.dataset['X'].min().values]
            Y = [od_in.dataset['Y'].min().values,
                 od_in.dataset['Y'].min().values,
                 od_in.dataset['Y'].max().values,
                 od_in.dataset['Y'].max().values,
                 od_in.dataset['Y'].min().values]
            od_moor = od_in.subsample.mooring_array(Xmoor=X,
                                                    Ymoor=Y)

        # Compute transport
        ds_out = mooring_volume_transport(od_moor)
        assert 'path' in ds_out.dims

        # Max 2 velocities per grid cell
        sum_dirs = (np.fabs(ds_out['dir_Utransport'].sum('path'))
                    + np.fabs(ds_out['dir_Vtransport'].sum('path'))).values
        assert np.all(np.logical_and(sum_dirs >= 0, sum_dirs <= 2))
        assert_allclose((ds_out['Utransport']+ds_out['Vtransport']).values,
                        ds_out['transport'].values)

        # Test shortcut
        od_out = od_moor.compute.mooring_volume_transport()
        ds_out_IN_od_out(ds_out, od_out)
    else:
        with pytest.raises(ValueError):
            mooring_volume_transport(od_in)


@pytest.mark.parametrize("od_in, gridtype", [(od,       'rect'),
                                             (alias_od, 'rect'),
                                             (od_curv,  'curv')])
@pytest.mark.parametrize("rotate",  [True, False])
@pytest.mark.parametrize("survey",  [True, False])
def test_survey_aligned_velocities(od_in, gridtype, rotate, survey):

    if rotate is True:
        if gridtype == 'curv':
            od_in = od_in.compute.geographical_aligned_velocities()
        else:
            with pytest.raises(ValueError):
                od_in.compute.geographical_aligned_velocities()

    if survey is True:
        X = [od_in.dataset['XC'].min().values,
             od_in.dataset['XC'].max().values]
        Y = [od_in.dataset['YC'].min().values,
             od_in.dataset['YC'].max().values]
        od_surv = od_in.subsample.survey_stations(Xsurv=X,
                                                  Ysurv=Y,
                                                  delta=2)
        # Align
        if gridtype == 'rect' or rotate is False:
            with pytest.warns(UserWarning):
                ds_out = survey_aligned_velocities(od_surv)
        elif rotate is True:
            ds_out = survey_aligned_velocities(od_surv)
            # Test shortcut
            od_surv.compute.survey_aligned_velocities()

        # Chek velocities
        vel_surv = np.sqrt(od_surv._ds['U']**2 + od_surv._ds['V']**2)
        vel_alig = np.sqrt(ds_out['tan_Vel']**2 + ds_out['ort_Vel']**2)
        if rotate is False:
            assert_allclose(vel_surv.values, vel_alig.values)
        else:
            # TODO: not exact match. Check with Renske!
            assert_almost_equal(vel_surv.values, vel_alig.values, 4)

    else:
        with pytest.raises(ValueError):
            survey_aligned_velocities(od_in)


@pytest.mark.parametrize("od_in", [od_bdg])
def test_heat_budget(od_in):

    for i in range(2):
        if i == 1:
            od_in = od_in.subsample.cutout(ZRange=[-5, -20])

        # Compute heat_budget
        ds_out = heat_budget(od_in)
        for var in ds_out.data_vars:
            assert ds_out[var].attrs['units'] == 'degC/s'
            check_params(ds_out, var, ['rho0', 'c_p'])

        var = ds_out['tendH']
        check = (ds_out['adv_hConvH'] + ds_out['adv_vConvH']
                 + ds_out['dif_vConvH'] + ds_out['kpp_vConvH']
                 + ds_out['forcH'])

        # Mask and check
        mask = xr.where(np.logical_or(check.isnull(), var.isnull()), 0, 1)
        assert_allclose(check.where(mask, drop=True).values,
                        var.where(mask, drop=True).values,
                        equal_nan=False)

        # Test shortcut
        od_out = od_in.compute.heat_budget()
        ds_out_IN_od_out(ds_out, od_out)


@pytest.mark.parametrize("od_in", [od_bdg])
def test_salt_budget(od_in):

    # Compute salt_budget
    ds_out = salt_budget(od_in)
    for var in ds_out.data_vars:
        assert ds_out[var].attrs['units'] == 'psu/s'
        check_params(ds_out, var, ['rho0'])

    var = ds_out['tendS']
    check = (ds_out['adv_hConvS'] + ds_out['adv_vConvS']
             + ds_out['dif_vConvS'] + ds_out['kpp_vConvS'] + ds_out['forcS'])

    # Mask and check
    mask = xr.where(np.logical_or(check.isnull(), var.isnull()), 0, 1)
    # Not sure why S it's a little less accurate than heat
    assert_allclose(check.where(mask, drop=True).values,
                    var.where(mask, drop=True).values,
                    1.E-6, equal_nan=False)

    # Test shortcut
    od_out = od_in.compute.salt_budget()
    ds_out_IN_od_out(ds_out, od_out)


def check_params(ds, varName, params):
    for par in params:
        assert par in ds[varName].attrs['OceanSpy_parameters']


def ds_out_IN_od_out(ds_out, od_out):
    for var in ds_out.data_vars:
        assert_array_equal(od_out.dataset[var].values, ds_out[var].values)
