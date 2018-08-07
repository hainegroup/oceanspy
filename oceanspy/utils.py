"""
Utils: functions used by other OceanSpy modules
"""

# Comments for developers:
# 1) Keep imported modules secret using _

import numpy as _np
import pickle as _pickle
import time as _time
import xarray as _xr
from . import compute as _compute

def great_circle_path(lat1,lon1,lat2,lon2,delta_km):
    """
    Generate a great circle trajectory specifying the resolution.
    
    Parameters
    ----------
    lat1: number
        Latitude of vertex 1 [degrees N]
    lon1: number
        Longitude of vertex 1 [degrees E]
    lat2: number
        Latitude of vertex 2 [degrees N]
    lon2: number
        Longitude of vertex 2 [degrees E]
    delta_km: number
        Horizontal resolution [km]
        
    Returns
    -------
    lats: vector
        Great circle latitudes [degrees N]
    lons: vector
        Great circle longitudes [degrees E]
    dist: vector
        Distances from vertex 1 [km]
    """
    
    # Convert to radians
    lat1=_np.deg2rad(lat1)
    lon1=_np.deg2rad(lon1)
    lat2=_np.deg2rad(lat2)
    lon2=_np.deg2rad(lon2)

    # Using the right-handed coordinate frame with Z toward (lat,lon)=(0,0) and 
    # Y toward (lat,lon)=(90,0), the unit_radial of a (lat,lon) is given by:
    # 
    #               [ cos(lat)*sin(lon) ]
    # unit_radial = [     sin(lat)      ]
    #               [ cos(lat)*cos(lon) ]
    unit_radial_1 = _np.array([_np.cos(lat1)*_np.sin(lon1), _np.sin(lat1), _np.cos(lat1)*_np.cos(lon1)])
    unit_radial_2 = _np.array([_np.cos(lat2)*_np.sin(lon2), _np.sin(lat2), _np.cos(lat2)*_np.cos(lon2)])

    # Define the vector that is normal to both unit_radial_1 & unit_radial_2
    normal_vec = _np.cross(unit_radial_1,unit_radial_2); 
    unit_normal = normal_vec / _np.sqrt(_np.sum(normal_vec**2))

    # Define the vector that is tangent to the great circle flight path at
    # (lat1,lon1)
    tangent_1_vec = _np.cross(unit_normal,unit_radial_1)
    unit_tangent_1 = tangent_1_vec / _np.sqrt(_np.sum(tangent_1_vec**2))

    # Determine the total arc distance from 1 to 2 
    total_arc_angle_1_to_2 = _np.arccos(unit_radial_1.dot(unit_radial_2)) # radians

    # Determine the set of arc angles to use 
    # (approximately spaced by delta_km)
    R0 = 6371; # km, radius of a circle having approximately the surface area of the earth
    angs2use = _np.linspace(0,total_arc_angle_1_to_2,_np.ceil(total_arc_angle_1_to_2/(delta_km/R0))); # radians
    M=angs2use.size;

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
    dlon = _np.deg2rad(lons[:-1] - lons[1:])
    dlat = _np.deg2rad(lats[:-1] - lats[1:])
    a = _np.sin(dlat / 2)**2 + _np.cos(lat1) * _np.cos(lat2) * _np.sin(dlon / 2)**2
    c = 2 * _np.arctan2(_np.sqrt(a), _np.sqrt(1 - a))
    dists = _np.append(0,_np.cumsum(R0 * c))
    
    return lats, lons, dists

def rotation_angle(dist, lats, lons):
    """
    Compute the rotation_angle to obtain orthogonal and tangential velocities.
    
    Parameters
    ----------
    dist: vector
        Distances from starting point
    lats: vector
        Longitudes [degrees E]
    lons: vector
        Latitudes [degrees N]
        
    Returns
    -------
    rot_ang: number
        Rotation angle [degrees]
    """
    
    # Find vertices
    imin = dist.argmin(); imax = dist.argmax()
    lat1 = _np.deg2rad(lats[imin]); lon1 = _np.deg2rad(lons[imin])
    lat2 = _np.deg2rad(lats[imax]); lon2 = _np.deg2rad(lons[imax])
    
    # Find rotation angle
    az = _np.arctan2(_np.cos(lat2) * _np.sin(lon2-lon1),
                     _np.cos(lat1) * _np.sin(lat2) - _np.sin(lat1) * _np.cos(lat2) * _np.cos(lon2-lon1)) 
    while _np.rad2deg(az)<0: az = _np.pi*2 + az
    rot_ang = _np.pi/2 - az
    if rot_ang<0: rot_ang = _np.pi*2 + rot_ang
        
    return rot_ang

def compute_missing_variables(ds, info, 
                              varList, 
                              deep_copy = False):
    """
    Try to compute missing variables using oceanspy.compute
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    varList: list
        List of variables to check
    deep_copy: bool
        If True, deep copy ds and info    
        
    Returns
    -------
    ds: xarray.Dataset
    info: open_dataset._info
    """
    
    # Deep copy
    if deep_copy: ds, info = _utils.deep_copy(ds, info)
    
    # Try to compute missing variables
    for var in varList:
        if var in info.var_names: var = info.var_names[var]
        if var not in ds.variables: 
            try: ds, info = eval('_compute.'+var+'(ds   = ds, info = info)')
            except: raise RuntimeError("%s not available and can't be computed" % var)    
                
    return ds, info

def deep_copy(ds, info):
    """
    Deep copy ds and info  
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info   
        
    Returns
    -------
    ds: xarray.Dataset
    info: open_dataset._info
    """
    
    # Message
    print('Copying ds and info')  
    
    # Copy ds
    ds = ds.copy(deep=True)
    
    # Copy info (pickle is faster than copy)
    info = _pickle.loads(_pickle.dumps(info, -1))
    
    return ds, info


def save_ds_info(ds, info, path):
    """
    Save ds and info to path.nc and path.obj, respectively 
    
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info   
    path: str
        Path to which to save ds and info
    """
    
    # Create paths
    ds_path   = path+'.nc'
    info_path = path+'.obj'
    
    # Save ds
    print('Saving ds to', ds_path)
    ds.to_netcdf(ds_path)
    
    # Save info
    info.to_obj(info_path)
    

    
def open_ds_info(path):
    """
    Open ds and info from path.nc and path.obj, respectively 
    
    Parameters
    ---------- 
    path: str
        Path from which to open ds and info.
        Do not provide extensions. 
        ds and info must have the same path (path.nc and path.obj).
        
    Returns
    -------
    ds: xarray.Dataset
    info: open_dataset._info
    """
    
    # Create paths
    ds_path   = path+'.nc'
    info_path = path+'.obj'
    
    # Open ds
    print('Opening ds from', ds_path)
    ds = _xr.open_dataset(ds_path)
    
    # Open info
    print('Opening info from', info_path)
    info = _pickle.load(open(info_path,'rb')) 
    
    return ds, info