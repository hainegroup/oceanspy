"""
Utils: useful functions used by OceanSpy
"""
import xarray as _xr
import numpy as _np

def spherical2cartesian(Y, X, R = None):
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
    # TODO: check Y and X

    # Check parameters
    if R is None:
        from geopy.distance import EARTH_RADIUS
        R = EARTH_RADIUS
    elif not isinstance(R, _np.ScalarType):
        raise TypeError('R must be None or numpy.scalar')
  
    # Convert
    if isinstance(Y, _xr.DataArray) and isinstance(X, _xr.DataArray):
        Y_rad = _xr.ufuncs.deg2rad(Y)
        X_rad = _xr.ufuncs.deg2rad(X)
        x = R * _xr.ufuncs.cos(Y_rad) * _xr.ufuncs.cos(X_rad)
        y = R * _xr.ufuncs.cos(Y_rad) * _xr.ufuncs.sin(X_rad)
        z = R * _xr.ufuncs.sin(Y_rad)
    else:
        Y_rad = _np.deg2rad(Y)
        X_rad = _np.deg2rad(X)
        x = R * _np.cos(Y_rad) * _np.cos(X_rad)
        y = R * _np.cos(Y_rad) * _np.sin(X_rad)
        z = R * _np.sin(Y_rad)
        
    return x, y, z

    
def great_circle_path(lat1, lon1, lat2, lon2, delta_km, R = None):
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
    delta_km: scalar
        Distance resolution [km]
    R: scalar
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
    Converted to python and adapted from: https://ww2.mathworks.cn/matlabcentral/mlc-downloads/downloads/submissions/8493/versions/2/previews/generate_great_circle_path.m/index.html?access_key=
    """
    
    # Check parameters
    if not isinstance(lat1, _np.ScalarType):    
        raise TypeError('`lat1` must be a scalar')
    if not isinstance(lon1, _np.ScalarType):    
        raise TypeError('`lon1` must be a scalar')
    if not isinstance(lat2, _np.ScalarType):    
        raise TypeError('`lat2` must be a scalar')
    if not isinstance(lon2, _np.ScalarType):    
        raise TypeError('`lon2` must be a scalar')
    if not isinstance(delta_km, _np.ScalarType):    
        raise TypeError('`delta_km` must be a scalar')
    if not isinstance(R, (_np.ScalarType, type(None))):    
        raise TypeError('`R` must be a scalar or None')
    if lat1==lat2 and lon1==lon2:
        raise TypeError('Vertexes are overlapping')
    if delta_km<=0:
        raise TypeError('`delta_km` can not be zero or negative')
                        
    # Check parameters
    if R is None:
        from geopy.distance import EARTH_RADIUS
        R = EARTH_RADIUS
    elif not isinstance(R, _np.ScalarType):
        raise TypeError('R must be None or numpy.scalar')
    
    # Import packages
    from geopy.distance import great_circle as _great_circle
    
    # Convert to radians
    lat1 = _np.deg2rad(lat1); lon1 = _np.deg2rad(lon1)
    lat2 = _np.deg2rad(lat2); lon2 = _np.deg2rad(lon2)

    # Using the right-handed coordinate frame with Z toward (lat,lon)=(0,0) and 
    # Y toward (lat,lon)=(90,0), the unit_radial of a (lat,lon) is given by:
    # 
    #               [ cos(lat)*sin(lon) ]
    # unit_radial = [     sin(lat)      ]
    #               [ cos(lat)*cos(lon) ]
    unit_radial_1 = _np.array([_np.cos(lat1)*_np.sin(lon1), _np.sin(lat1), _np.cos(lat1)*_np.cos(lon1)])
    unit_radial_2 = _np.array([_np.cos(lat2)*_np.sin(lon2), _np.sin(lat2), _np.cos(lat2)*_np.cos(lon2)])

    # Define the vector that is normal to both unit_radial_1 & unit_radial_2
    normal_vec  = _np.cross(unit_radial_1,unit_radial_2); 
    unit_normal = normal_vec / _np.sqrt(_np.sum(normal_vec**2))

    # Define the vector that is tangent to the great circle flight path at
    # (lat1,lon1)
    tangent_1_vec  = _np.cross(unit_normal,unit_radial_1)
    unit_tangent_1 = tangent_1_vec / _np.sqrt(_np.sum(tangent_1_vec**2))

    # Determine the total arc distance from 1 to 2 
    total_arc_angle_1_to_2 = _np.arccos(unit_radial_1.dot(unit_radial_2)) # radians

    # Determine the set of arc angles to use 
    # (approximately spaced by delta_km)
    angs2use = _np.linspace(0,total_arc_angle_1_to_2,_np.ceil(total_arc_angle_1_to_2/(delta_km/R))); # radians
    M = angs2use.size;

    # Now find the unit radials of the entire "trajectory"
    #                                                              [ cos(angs2use(m)) -sin(angs2use(m)) 0 ]   [ 1 ]
    # unit_radial_m = [unit_radial_1 unit_tangent_1 unit_normal] * [ sin(angs2use(m))  cos(angs2use(m)) 0 ] * [ 0 ]
    #                                                              [        0                 0         1 ]   [ 0 ]
    # equals
    #                                                              [ cos(angs2use(m)) ]
    # unit_radial_m = [unit_radial_1 unit_tangent_1 unit_normal] * [ sin(angs2use(m)) ]
    #                                                              [        0         ]
    # equals
    #
    # unit_radial_m = [unit_radial_1*cos(angs2use(m)) + unit_tangent_1*sin(angs2use(m)) + 0]
    #
    # unit_radials is a 3xM array of M unit radials
    unit_radials = _np.array([unit_radial_1]).transpose().dot(_np.ones((1,M))) *\
                   _np.ones((3,1)).dot(_np.array([_np.cos(angs2use)])) +\
                   _np.array([unit_tangent_1]).transpose().dot(_np.ones((1,M))) *\
                   _np.ones((3,1)).dot(_np.array([_np.sin(angs2use)]))

    # Convert to latitudes and longitudes
    lats = _np.rad2deg(_np.arcsin(unit_radials[1,:]))
    lons = _np.rad2deg(_np.arctan2(unit_radials[0,:],unit_radials[2,:]))
    
    # Compute distance
    dists = _np.zeros(lons.shape)
    for i in range(1,len(lons)):
        coord1   = (lats[i-1],lons[i-1])
        coord2   = (lats[i],lons[i])
        dists[i] = _great_circle(coord1,coord2, radius = R).km
    dists = _np.cumsum(dists)
    
    return lats, lons, dists


def cartesian_path(x1, y1, x2, y2, delta):
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
    if not isinstance(x1, _np.ScalarType):    
        raise TypeError('`x1` must be a scalar')
    if not isinstance(y1, _np.ScalarType):    
        raise TypeError('`y1` must be a scalar')
    if not isinstance(x2, _np.ScalarType):    
        raise TypeError('`x2` must be a scalar')
    if not isinstance(y2, _np.ScalarType):    
        raise TypeError('`y2` must be a scalar')
    if not isinstance(delta, _np.ScalarType):    
        raise TypeError('`delta` must be a scalar')
    if x1==x2 and x1==x2:
        raise TypeError('Vertexes are overlapping')
    if delta<=0:
        raise TypeError('`delta` can not be zero or negative')
        
    # Interpolate
    dist_tot = _np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    coefs    = _np.linspace(0, 1, round(dist_tot/delta))
    xs       = x1 + coefs * (x2 - x1)
    ys       = y1 + coefs * (y2 - y1)
    dists    = coefs*dist_tot
    
    return xs, ys, dists