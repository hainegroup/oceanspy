"""
OceanSpy utilities that don't need OceanDataset objects.
"""

import copy as _copy

import numpy as _np

# Required dependencies (private)
import xarray as _xr

# From oceanspy (private)
from ._ospy_utils import _check_instance

# Recommended dependencies (private)
try:
    from geopy.distance import great_circle as _great_circle
except ImportError:  # pragma: no cover
    pass

try:
    import numba

    has_numba = True
except ImportError:
    has_numba = False


def compilable(f):
    if has_numba:
        return numba.njit(f)
    return f


def viewer_to_range(p):
    """
    Takes the output from the poseidon viewer `p` and returns the coordinate
    trajectories in X and Y in a way that is compatible with oceanspy.subsample
    functions.
    Parameters
    ----------
    p: list.
        data exported from the Poseidon viewer
    Returns
    -------
    lon: list.
    lat: list.
    """
    _check_instance({"p": p}, {"p": ["list"]})
    _check_instance({"p[0]": p[0]}, {"p[0]": ["dict"]})
    _check_instance({"type": p[0]["type"]}, {"type": "str"})
    _check_instance({"coord": p[0]["coordinates"]}, {"coord": "list"})

    if p[0]["type"] in ["Polygon", "LineString", "Point"]:
        # print messege
        print("extracting " + p[0]["type"])
    else:
        raise TypeError("not a type extracted by poseidon viewer")

    p_type = p[0]["type"]

    if p_type == "Polygon":
        coords = p[0]["coordinates"][0]
    elif p_type in ["LineString", "Point"]:
        coords = p[0]["coordinates"]

    lon = []
    lat = []

    for i in range(len(coords)):
        lon.append(coords[i][0])
        lat.append(coords[i][1])

    return lon, lat


def _rel_lon(x, ref_lon):
    """
    Change the definition of 0 longitude.
    Return how much east one need to go from ref_lon to x
    This function aims to address the confusion caused by
    the discontinuity in longitude.
    Parameters
    ----------
    x: float, numpy.array, dask.array
        longitude to convert
    ref_lon: float
        a reference longitude. This longitude will become 0deg
    Returns
    -------
    redefined_x: float, numpy.array, dask.array
        converted longitude
    """
    return (x - ref_lon) % 360


def _reset_range(x):
    """Resets the definition of XRange, by default the discontinuity at 180 long.
    Checks that there is no sign change in x and if there is, the only change that
    is allowed is when crossing zero. Otherwise resets ref_lon.

    Parameters
    ----------

    x: numpy.array.
        array with longitude values.

    Returns
    -------

    x0, x1: numpy.array
        endpoints of cutout.
    redefined_x: numpy.array.
        converted longitude.
    """

    ref_lon = 180
    if x is not None:
        if (_np.sign(x) == _np.sign(x[0])).all():  # no sign change
            _ref_lon = ref_lon
            X0, X1 = _np.min(x), _np.max(x)
        else:  # change in sign
            if len(x) == 2:  # list of end points
                X0, X1 = x
                if x[0] > x[1]:  # across discontinuity
                    if abs(x[1] - x[0]) > 300:  # across a discontinuity (Delta X =360)
                        _ref_lon = x[0] - (x[0] - x[1]) / 3
                    else:  # XRange decreases, but not necessarity a dicont.
                        _ref_lon = ref_lon
                else:
                    _ref_lon = ref_lon
            else:  # array of values.
                _del = abs(x[1:] - x[:-1])  # only works with one crossing
                if len(_np.where(abs(_del) > 300)[0]) > 0:  # there's discontinuity
                    ll = _np.where(_del == max(_del))[0][0]
                    if x[ll] > x[ll + 1]:  # track starts west of jump
                        X0 = _np.min(x[: ll + 1])
                        X1 = _np.max(x[ll + 1 :])
                    else:
                        X0 = _np.min(x[ll + 1 :])
                        X1 = _np.max(x[: ll + 1])
                    _ref_lon = X0 - (X0 - X1) / 3
                else:  # no discontinuity
                    X0 = _np.min(x)
                    X1 = _np.max(x)
                    _ref_lon = ref_lon
        x = _np.array([X0, X1])
    else:
        _ref_lon = ref_lon
    return x, _np.round(_ref_lon, 2)


def spherical2cartesian(Y, X, R=None):
    """
    Convert spherical coordinates to cartesian.

    Parameters
    ----------
    Y: _xr.DataArray or array_like
        Spherical Y coordinate (latitude)
    X: _xr.DataArray or array_like
        Spherical X coordinate (longitude)
    R: scalar
        Earth radius in km
        If None, use geopy default

    Returns
    -------
    x: _xr.DataArray or array_like
        Cartesian x coordinate
    y: _xr.DataArray or array_like
        Cartesian y coordinate
    z: scalar
        Cartesian z coordinate
    """
    # Check parameters
    _check_instance({"R": R}, {"R": ["type(None)", "numpy.ScalarType"]})

    # Check parameters
    if R is None:
        from geopy.distance import EARTH_RADIUS

        R = EARTH_RADIUS

    # Convert
    Y_rad = _np.deg2rad(Y)
    X_rad = _np.deg2rad(X)
    x = R * _np.cos(Y_rad) * _np.cos(X_rad)
    y = R * _np.cos(Y_rad) * _np.sin(X_rad)
    z = R * _np.sin(Y_rad)

    return x, y, z


def great_circle_path(lat1, lon1, lat2, lon2, delta_km=None, R=None):
    """
    Generate a great circle trajectory specifying the distance resolution.

    Parameters
    ----------
    lat1: scalar
        Latitude of vertex 1 [degrees N]
    lon1: scalar
        Longitude of vertex 1 [degrees E]
    lat2: scalar
        Latitude of vertex 2 [degrees N]
    lon2: scalar
        Longitude of vertex 2 [degrees E]
    delta_km: scalar, None
        Distance resolution [km]
        If None, only use vertices and return distance
    R: scalar, None
        Earth radius in km
        If None, use geopy default

    Returns
    -------
    lats: 1D numpy.ndarray
        Great circle latitudes [degrees N]
    lons: 1D numpy.ndarray
        Great circle longitudes [degrees E]
    dist: 1D numpy.ndarray
        Distances from vertex 1 [km]

    References
    ----------
    Converted to python and adapted from:
    `<https://ww2.mathworks.cn/matlabcentral/mlc-downloads/downloads/
    submissions/8493/versions/2/previews/generate_great_circle_path.m
    /index.html?access_key=>`_
    """

    # Check parameters
    _check_instance(
        {
            "lat1": lat1,
            "lon1": lon1,
            "lat2": lat2,
            "lon2": lon2,
            "delta_km": delta_km,
            "R": R,
        },
        {
            "lat1": "numpy.ScalarType",
            "lon1": "numpy.ScalarType",
            "lat2": "numpy.ScalarType",
            "lon2": "numpy.ScalarType",
            "delta_km": ["type(None)", "numpy.ScalarType"],
            "R": ["type(None)", "numpy.ScalarType"],
        },
    )

    # Check parameters
    if lat1 == lat2 and lon1 == lon2:
        raise ValueError("Vertexes are overlapping")
    if delta_km is not None and delta_km <= 0:
        raise ValueError("`delta_km` can not be zero or negative")

    # Check parameters
    if R is None:
        from geopy.distance import EARTH_RADIUS

        R = EARTH_RADIUS

    if delta_km is not None:
        # Convert to radians
        lat1 = _np.deg2rad(lat1)
        lon1 = _np.deg2rad(lon1)
        lat2 = _np.deg2rad(lat2)
        lon2 = _np.deg2rad(lon2)

        # Using the right-handed coordinate frame
        # with Z toward (lat,lon)=(0,0) and
        # Y toward (lat,lon)=(90,0),
        # the unit_radial of a (lat,lon) is given by:
        #               [ cos(lat)*sin(lon) ]
        # unit_radial = [     sin(lat)      ]
        #               [ cos(lat)*cos(lon) ]
        unit_radial_1 = _np.array(
            [
                _np.cos(lat1) * _np.sin(lon1),
                _np.sin(lat1),
                _np.cos(lat1) * _np.cos(lon1),
            ]
        )
        unit_radial_2 = _np.array(
            [
                _np.cos(lat2) * _np.sin(lon2),
                _np.sin(lat2),
                _np.cos(lat2) * _np.cos(lon2),
            ]
        )

        # Define the vector that is normal to
        # both unit_radial_1 & unit_radial_2
        normal_vec = _np.cross(unit_radial_1, unit_radial_2)
        unit_normal = normal_vec / _np.sqrt(_np.sum(normal_vec**2))

        # Define the vector that is tangent to the great circle flight path at
        # (lat1,lon1)
        tangent_1_vec = _np.cross(unit_normal, unit_radial_1)
        unit_tangent_1 = tangent_1_vec / _np.sqrt(_np.sum(tangent_1_vec**2))

        # Determine the total arc distance from 1 to 2 (radians)
        total_arc_angle_1_to_2 = _np.arccos(unit_radial_1.dot(unit_radial_2))

        # Determine the set of arc angles to use
        # (approximately spaced by delta_km)
        angs2use = _np.linspace(
            0,
            total_arc_angle_1_to_2,
            int(_np.ceil(total_arc_angle_1_to_2 / (delta_km / R))),
        )  # radians
        M = angs2use.size

        # unit_radials is a 3xM array of M unit radials
        term1 = _np.array([unit_radial_1]).transpose().dot(_np.ones((1, M)))
        term2 = _np.ones((3, 1)).dot(_np.array([_np.cos(angs2use)]))
        term3 = _np.array([unit_tangent_1]).transpose().dot(_np.ones((1, M)))
        term4 = _np.ones((3, 1)).dot(_np.array([_np.sin(angs2use)]))
        unit_radials = term1 * term2 + term3 * term4

        # Convert to latitudes and longitudes
        lats = _np.rad2deg(_np.arcsin(unit_radials[1, :]))
        lons = _np.rad2deg(_np.arctan2(unit_radials[0, :], unit_radials[2, :]))

    else:
        # Use input lon and lat
        lats = _np.concatenate((_np.reshape(lat1, 1), _np.reshape(lat2, 1)))
        lons = _np.concatenate((_np.reshape(lon1, 1), _np.reshape(lon2, 1)))

    # Compute distance
    dists = _np.zeros(lons.shape)
    for i in range(1, len(lons)):
        coord1 = (lats[i - 1], lons[i - 1])
        coord2 = (lats[i], lons[i])
        dists[i] = _great_circle(coord1, coord2, radius=R).km
    dists = _np.cumsum(dists)

    return lats, lons, dists


def cartesian_path(x1, y1, x2, y2, delta=None):
    """
    Generate a trajectory specifying the distance resolution.

    Parameters
    ----------
    x1: scalar
        x coordinate of vertex 1
    y1: scalar
        y coordinate of vertex 1
    x2: scalar
        x coordinate of vertex 2
    y2: scalar
        y coordinate of vertex 2
    delta: scalar
        Distance resolution (same units of x and y)
        If None, only use vertices and return distance
    Returns
    -------
    xs: 1D numpy.ndarray
        x coordinates
    ys: 1D numpy.ndarray
        y coordinates
    dist: 1D numpy.ndarray
        distances
    """

    # Check parameters
    _check_instance(
        {"x1": x1, "x2": x2, "y1": y1, "y2": y2, "delta": delta},
        {
            "x1": "numpy.ScalarType",
            "x2": "numpy.ScalarType",
            "y1": "numpy.ScalarType",
            "y2": "numpy.ScalarType",
            "delta": ["type(None)", "numpy.ScalarType"],
        },
    )

    if x1 == x2 and x1 == x2:
        raise ValueError("Vertexes are overlapping")
    if delta is not None and delta <= 0:
        raise ValueError("`delta` can not be zero or negative")

    # Interpolate
    dist_tot = _np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if delta is None:
        coefs = _np.linspace(0, 1, 2)
    else:
        coefs = _np.linspace(0, 1, int(round(dist_tot / delta)))
    xs = x1 + coefs * (x2 - x1)
    ys = y1 + coefs * (y2 - y1)
    dists = coefs * dist_tot

    return xs, ys, dists


def densjmd95(s, t, p):
    """
    Density of Sea Water using Jackett and McDougall 1995 (JAOT 12)
    polynomial (modified UNESCO polynomial) [JaMc95]_.
    jmd95.py:
    `<http://mitgcm.org/\
    download/daily_snapshot/MITgcm/utils/python/MITgcmutils/MITgcmutils/jmd95.py>`_

    Parameters
    ----------
    s: xarray.DatArray, array-like
        salinity    [psu (PSS-78)]
    t: xarray.DatArray, array-like
        potential temperature [degree C (IPTS-68)]
    p: xarray.DatArray, array-like
        pressure [dbar]
        (p may have dims 1x1, mx1, 1xn or mxn for S(mxn))

    Returns
    -------
    rho: xarray.DatArray, array-like
        density  [kg/m^3]

    References
    ----------
    .. [JaMc95]
        Jackett, D.R. and T.J. Mcdougall, 1995:\
        Minimal Adjustment of Hydrographic Profiles\
        to Achieve Static Stability.\
        J. Atmos. Oceanic Technol., 12, 381–389,\
        https://doi.org/10.1175/1520-0426(1995)012<0381:MAOHPT>2.0.CO;2
    """

    # make sure arguments are floating point
    for var in [s, t, p]:
        if isinstance(var, _xr.DataArray):
            var = var.astype("float")
        else:
            var = _np.asfarray(var)

    # coefficients nonlinear equation of state in pressure coordinates for
    # 1. density of fresh water at p = 0
    eosJMDCFw = [
        999.842594,
        6.793952e-02,
        -9.095290e-03,
        1.001685e-04,
        -1.120083e-06,
        6.536332e-09,
    ]

    # 2. density of sea water at p = 0
    eosJMDCSw = [
        8.244930e-01,
        -4.089900e-03,
        7.643800e-05,
        -8.246700e-07,
        5.387500e-09,
        -5.724660e-03,
        1.022700e-04,
        -1.654600e-06,
        4.831400e-04,
    ]

    # coefficients in pressure coordinates for
    # 3. secant bulk modulus K of fresh water at p = 0
    eosJMDCKFw = [1.965933e04, 1.444304e02, -1.706103e00, 9.648704e-03, -4.190253e-05]

    # 4. secant bulk modulus K of sea water at p = 0
    eosJMDCKSw = [
        5.284855e01,
        -3.101089e-01,
        6.283263e-03,
        -5.084188e-05,
        3.886640e-01,
        9.085835e-03,
        -4.619924e-04,
    ]

    # 5. secant bulk modulus K of sea water at p
    eosJMDCKP = [
        3.186519e00,
        2.212276e-02,
        -2.984642e-04,
        1.956415e-06,
        6.704388e-03,
        -1.847318e-04,
        2.059331e-07,
        1.480266e-04,
        2.102898e-04,
        -1.202016e-05,
        1.394680e-07,
        -2.040237e-06,
        6.128773e-08,
        6.207323e-10,
    ]

    # convert pressure to bar
    p = 0.1 * p
    p2 = p * p
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    s3o2 = s * _np.sqrt(s)

    # density of freshwater at the surface
    rho = (
        eosJMDCFw[0]
        + eosJMDCFw[1] * t
        + eosJMDCFw[2] * t2
        + eosJMDCFw[3] * t3
        + eosJMDCFw[4] * t4
        + eosJMDCFw[5] * t4 * t
    )

    # density of sea water at the surface
    rho = (
        rho
        + s
        * (
            eosJMDCSw[0]
            + eosJMDCSw[1] * t
            + eosJMDCSw[2] * t2
            + eosJMDCSw[3] * t3
            + eosJMDCSw[4] * t4
        )
        + s3o2 * (eosJMDCSw[5] + eosJMDCSw[6] * t + eosJMDCSw[7] * t2)
        + eosJMDCSw[8] * s * s
    )

    # secant bulk modulus of fresh water at the surface
    bulkmod = (
        eosJMDCKFw[0]
        + eosJMDCKFw[1] * t
        + eosJMDCKFw[2] * t2
        + eosJMDCKFw[3] * t3
        + eosJMDCKFw[4] * t4
    )

    # secant bulk modulus of sea water at the surface
    bulkmod = (
        bulkmod
        + s
        * (eosJMDCKSw[0] + eosJMDCKSw[1] * t + eosJMDCKSw[2] * t2 + eosJMDCKSw[3] * t3)
        + s3o2 * (eosJMDCKSw[4] + eosJMDCKSw[5] * t + eosJMDCKSw[6] * t2)
    )

    # secant bulk modulus of sea water at pressure p
    bulkmod = (
        bulkmod
        + p * (eosJMDCKP[0] + eosJMDCKP[1] * t + eosJMDCKP[2] * t2 + eosJMDCKP[3] * t3)
        + p * s * (eosJMDCKP[4] + eosJMDCKP[5] * t + eosJMDCKP[6] * t2)
        + p * s3o2 * eosJMDCKP[7]
        + p2 * (eosJMDCKP[8] + eosJMDCKP[9] * t + eosJMDCKP[10] * t2)
        + p2 * s * (eosJMDCKP[11] + eosJMDCKP[12] * t + eosJMDCKP[13] * t2)
    )
    rho = rho / (1.0 - p / bulkmod)

    return rho


def densmdjwf(s, t, p):
    """
    Density of Sea Water using McDougall et al. 2003 (JAOT 20)
    polynomial (Gibbs Potential) [McJa03]_.
    mdjwf.py:
    `<https://github.com/\
    MITgcm/MITgcm/blob/master/utils/python/MITgcmutils/MITgcmutils/mdjwf.py>`_

    Parameters
    ----------
    s: xarray.DatArray, array-like
        salinity [psu (PSS-78)]
    t: xarray.DatArray, array-like
        potential temperature [degree C (IPTS-68)]
    p: xarray.DatArray, array-like
        pressure [dbar]
        (p may have dims 1x1, mx1, 1xn or mxn for S(mxn))

    Returns
    -------
    rho: xarray.DatArray, array-like
        density  [kg/m^3]

    References
    ----------
    .. [McJa03]
        McDougall, T.J., D.R. Jackett, D.G. Wright, and R. Feistel, 2003:\
        Accurate and Computationally Efficient Algorithms for\
        Potential Temperature and Density of Seawater.\
        J. Atmos. Oceanic Technol., 20, 730–741,\
        https://doi.org/10.1175/1520-0426(2003)20<730:AACEAF>2.0.CO;2
    """

    # make sure arguments are floating point
    for var in [s, t, p]:
        if isinstance(var, _xr.DataArray):
            var = var.astype("float")
        else:
            var = _np.asfarray(var)

    # coefficients nonlinear equation of state in pressure coordinates for
    eosMDJWFnum = [
        7.35212840e00,
        -5.45928211e-02,
        3.98476704e-04,
        2.96938239e00,
        -7.23268813e-03,
        2.12382341e-03,
        1.04004591e-02,
        1.03970529e-07,
        5.18761880e-06,
        -3.24041825e-08,
        -1.23869360e-11,
        9.99843699e02,
    ]

    eosMDJWFden = [
        7.28606739e-03,
        -4.60835542e-05,
        3.68390573e-07,
        1.80809186e-10,
        2.14691708e-03,
        -9.27062484e-06,
        -1.78343643e-10,
        4.76534122e-06,
        1.63410736e-09,
        5.30848875e-06,
        -3.03175128e-16,
        -1.27934137e-17,
        1.00000000e00,
    ]

    p1 = _copy.copy(p)
    t1 = _copy.copy(t)
    t2 = t * t
    s1 = _copy.copy(s)
    sp5 = _np.sqrt(s1)
    p1t1 = p1 * t1

    num = (
        eosMDJWFnum[11]
        + t1 * (eosMDJWFnum[0] + t1 * (eosMDJWFnum[1] + eosMDJWFnum[2] * t1))
        + s1 * (eosMDJWFnum[3] + eosMDJWFnum[4] * t1 + eosMDJWFnum[5] * s1)
        + p1
        * (
            eosMDJWFnum[6]
            + eosMDJWFnum[7] * t2
            + eosMDJWFnum[8] * s1
            + p1 * (eosMDJWFnum[9] + eosMDJWFnum[10] * t2)
        )
    )

    den = (
        eosMDJWFden[12]
        + t1
        * (
            eosMDJWFden[0]
            + t1 * (eosMDJWFden[1] + t1 * (eosMDJWFden[2] + t1 * eosMDJWFden[3]))
        )
        + s1
        * (
            eosMDJWFden[4]
            + t1 * (eosMDJWFden[5] + eosMDJWFden[6] * t2)
            + sp5 * (eosMDJWFden[7] + eosMDJWFden[8] * t2)
        )
        + p1 * (eosMDJWFden[9] + p1t1 * (eosMDJWFden[10] * t2 + eosMDJWFden[11] * p1))
    )

    epsln = 0
    denom = 1.0 / (epsln + den)
    rho = num * denom

    return rho


def static_pressure(Z):
    """
    Returns the static pressure given depth.
    """
    # Coefficients to determine static pressure
    c = [0.059808, -0.025, 0.100766, 2.28405e-7]
    P = c[0] * _np.exp((c[1] * Z) - 1) + c[2] * Z + c[3] * (Z**2)
    P = 10 * P  # decibars
    return P


def Coriolis_parameter(Y, omega=7.2921e-5):
    """
    Compute Coriolis parameter (both vertical and horizontal components).

    .. math::
        (f, e) = (\\hat{\\mathbf{z}}\\cdot
        \\left(2\\mathbf{\\Omega}\\right),
        \\hat{\\mathbf{y}}\\cdot\\left(2\\mathbf{\\Omega}\\right))
        = (2|\\mathbf{\\Omega}|\\sin{\\theta},
        2|\\mathbf{\\Omega}|\\cos{\\theta})

    Parameters
    ----------
    Y: _xr.DataArray or array_like
        Y coordinate (latitude)
    omega: scalar
        Rotation rate of Earth (rad/s)

    Returns
    -------
    f: _xr.DataArray or array_like
        Vertical component of the Earth's rotation vector
    e: _xr.DataArray or array_like
        Horizontal component of the Earth's rotation vector
    """
    # TODO: check Y and omega

    # Convert
    Y_rad = _np.deg2rad(Y)
    f = 2 * omega * _np.sin(Y_rad)
    e = 2 * omega * _np.cos(Y_rad)

    return f, e


def get_maskH(ds, add_Hbdr, XRange, YRange, ref_lon=0):
    """Define this function to avoid repeated code. First time this runs,
    the objective is to figure out which faces survive the cutout. This info
    is then passed, when transforming llc-grids, to llc_rearrange. Second
    time this code runs, it gets applied on a dataset without faces as a
    dimension.
    """
    if "face" in ds.dims and len(ds.X) == 4320:
        args = {"Xp1": slice(0, 4320, 10), "Yp1": slice(0, 4320, 10)}
        ds = _copy.deepcopy(ds.isel(**args))

    maskH = _xr.ones_like(ds["XG"])

    if YRange is not None:
        # Use arrays
        YRange = _np.asarray([_np.min(YRange) - add_Hbdr, _np.max(YRange) + add_Hbdr])
        YRange = YRange.astype(ds["YG"].dtype)

        # Get the closest
        for i, Y in enumerate(YRange):
            diff = _np.fabs(ds["YG"] - Y)
            YRange[i] = ds["YG"].where(diff == diff.min()).min().values
        maskH = maskH.where(
            _np.logical_and(ds["YG"] >= YRange[0], ds["YG"] <= YRange[-1]), 0
        )

    if XRange is not None:
        # Use arrays
        XRange = _np.asarray([XRange[0] - add_Hbdr, XRange[-1] + add_Hbdr])
        XRange = XRange.astype(ds["XG"].dtype)

        # Get the closest
        for i, X in enumerate(XRange):
            diff = _np.fabs(ds["XG"] - X)
            XRange[i] = ds["XG"].where(diff == diff.min()).min().values
        maskH = maskH.where(
            _np.logical_and(
                _rel_lon(ds["XG"], ref_lon) >= _rel_lon(XRange[0], ref_lon),
                _rel_lon(ds["XG"], ref_lon) <= _rel_lon(XRange[-1], ref_lon),
            ),
            0,
        )
    # Can't be all zeros
    if maskH.sum() == 0:
        raise ValueError("Zero grid points in the horizontal range")

    # Find horizontal indexes
    maskH = maskH.assign_coords(
        Yp1=_np.arange(len(maskH["Yp1"])), Xp1=_np.arange(len(maskH["Xp1"]))
    )
    dmaskH = maskH.where(maskH, drop=True)
    return maskH, dmaskH, XRange, YRange


@compilable
def spherical2cartesian_compiled(Y, X, R=6371.0):
    """
    Convert spherical coordinates to cartesian.

    Parameters
    ----------
    Y: np.array
        Spherical Y coordinate (latitude)
    X: np.array
        Spherical X coordinate (longitude)
    R: scalar
        Earth radius in km
        If None, use geopy default
    Returns
    -------
    x: np.array
        Cartesian x coordinate
    y: np.array
        Cartesian y coordinate
    z: np.array
        Cartesian z coordinate
    """

    # Convert
    Y_rad = _np.deg2rad(Y)
    X_rad = _np.deg2rad(X)
    x = R * _np.cos(Y_rad) * _np.cos(X_rad)
    y = R * _np.cos(Y_rad) * _np.sin(X_rad)
    z = R * _np.sin(Y_rad)

    return x, y, z


@compilable
def to_180(x):
    """
    convert any longitude scale to [-180,180)

    Parameters
    ----------
    x: float ,numpy.array
        angles in degree

    Returns
    -------
    redefined_x: float,numpy.ndarray
        the same angle coverted to [-180,180)
    """
    x = x % 360
    return x + (-1) * (x // 180) * 360


def local_to_latlon(u, v, cs, sn):
    """
    convert local vector to north-east

    Parameters
    ----------
    u: float, numpy.array
        the x component of the vector
    v: float, numpy.array
        the y component of the vector
    cs: float, numpy.array
        the cosine of the angle between grid and compass
    sn: float, numpy.array
        the sine of the angle between grid and compass

    Returns
    -------
    uu: float, numpy.array
        the eastward component of the vector
    vv: float, numpy.array
        the northward component of the vector
    """
    uu = u * cs - v * sn
    vv = u * sn + v * cs
    return uu, vv


def get_combination(lst, select):
    """
    Iteratively find all the combination that
    has (select) amount of elements
    and every element belongs to lst

    Parameters
    ----------
    lst: list
        a iterable object to select from
    select: int
        the number of objects to select

    Returns
    -------
    com_list: list
        a list of all the possible combinations
    """
    # TODO: see if the one in itertools can replace this
    if select == 1:
        return [[num] for num in lst]
    else:
        the_lst = []
        for i, num in enumerate(lst):
            sub_lst = get_combination(lst[i + 1 :], select - 1)
            for com in sub_lst:
                com.append(num)
            #             print(sub_lst)
            the_lst += sub_lst
        return the_lst


def find_cs_sn(thetaA, phiA, thetaB, phiB):
    """
    theta is the angle
    between the meridian crossing point A
    and the geodesic connecting A and B

    this function return cos and sin of theta

    Parameters
    ----------
    thetaA: float,numpy.array
        the latitude of the vertex of the angle
    phiA: float,numpy.array
        the longitude of the vertex of the angle
    thetaB: float,numpy.array
        the latitude of the point B
    phiB: float,numpy.array
        the longitude of the point B

    Returns
    -------
    CS: float,numpy.array
        the cosine of the angle
    SN: float, numpy.array
        the sine of the angle
    """
    # O being north pole
    AO = _np.pi / 2 - thetaA
    BO = _np.pi / 2 - thetaB
    dphi = phiB - phiA
    # Spherical law of cosine on AOB
    cos_AB = _np.cos(BO) * _np.cos(AO) + _np.sin(BO) * _np.sin(AO) * _np.cos(dphi)
    sin_AB = _np.sqrt(1 - cos_AB**2)
    # spherical law of sine on triangle AOB
    SN = _np.sin(BO) * _np.sin(dphi) / sin_AB
    CS = _np.sign(thetaB - thetaA) * _np.sqrt(1 - SN**2)
    return CS, SN
