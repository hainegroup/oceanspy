import xarray as _xr
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import cartopy.crs as _ccrs

def smart_chunking(ds,
                   order_of_mag = 6,
                   dims2chunk   = ['time', 'Z', 'X', 'Y']):
    """
    Chunk a dataset defining the order of magnitude of elements in each 4D chunk.

    From xarray's documentation:
    A good rule of thumb to create arrays with a minimum chunksize of at least one million elements. 
    With large arrays (10+ GB), the cost of queueing up dask operations can be noticeable, 
    and you may need even larger chunksizes.

    Parameters
    ----------
    ds: xarray.Dataset or None
        Dataset that will be chunked.
    order_of_marg: int
        Order of magnitude of elements in each 4D chunk.
    dims2chunk: list
        Dimensions to chunk. 
        To minimize chunked dimensions, it starts from first, then second if necessary
        Available dimensions are ['time', 'Z', 'X', 'Y']
    
    Returns
    -------
    ds: xarray.Dataset
        Chunked Dataset 
    """
    
    # Check parameters
    if not isinstance(ds, _xr.Dataset):   raise RuntimeError("'ds' must be a xarray.Dataset")
    if not isinstance(order_of_mag, int): raise RuntimeError("'order_of_mag' must be an integer")
    if not isinstance(dims2chunk, list):  raise RuntimeError("'dims2chunk' must be a list")
        
    # Get dimensions' size
    chunks = {}
    for dim in dims2chunk: chunks[dim] = ds[dim].size

    # Loop reemoving 1 every time
    totSize = 1
    for key in chunks: 
        totSize=totSize * chunks[key]
    totOM = int(_np.log10(totSize))
    while totOM>order_of_mag: 
        for dim in dims2chunk:
            if chunks[dim]>1:
                chunks[dim] = chunks[dim]-1
                totSize = 1
                for key in chunks: 
                    totSize=totSize * chunks[key]
                totOM = int(_np.log10(totSize))
                break

    # Chunk dataset
    CHUNKS = {}
    for dim in ds.dims:
        for dim2chunk in dims2chunk:
            if dim==dim2chunk: CHUNKS[dim]=chunks[dim2chunk]
            elif dim[0]==dim2chunk:
                if ds[dim].size==chunks[dim2chunk]+1: CHUNKS[dim]=ds[dim].size
                else: CHUNKS[dim]=chunks[dim2chunk]
            else: 
                CHUNKS[dim]=ds[dim].size
            break
    ds = ds.chunk(chunks=CHUNKS)
    
    return ds



def disp_variables(ds):    
    """
    Print a table with the available variables, and their descriptions and units

    Parameters
    ----------
    ds: xarray.Dataset or None
        Dataset that will be chunked.
    """
    
    # Check parameters
    if not isinstance(ds, _xr.Dataset): raise RuntimeError("'ds' must be a xarray.Dataset")
        
    from IPython.core.display import HTML, display
    name        = ds.variables
    description = []
    units       = []
    for varName in name:
        this_desc  = ds[varName].attrs.get('long_name')
        this_units = ds[varName].attrs.get('units')
        if this_desc is None: 
            this_desc = ds[varName].attrs.get('description')
            if this_desc is None: this_desc = ' '
        if this_units is None: this_units = ' '
        description.append(this_desc)
        units.append(this_units)
    table = {'Name': name,'Description': description, 'Units':units}    
    table = _pd.DataFrame(table)
    display(HTML(table[['Name','Description','Units']].to_html()))
    
    
    
def plot_mercator(da):
    """
    Quick-plot of a 2D DataArray using Mercator projection.

    Parameters
    ----------
    da: xarray.DataArray        
        DataArray to plot.
        
    Returns
    -------
    ax: cartopy.mpl.geoaxes.GeoAxes
        Object used by cartopy, which is a subclass of a normal matplotlib Axes.
    gl: cartopy.mpl.gridliner.Gridliner
        Object used by cartopy to add gridlines and tick labels to a map.
    """
    
    # Check parameters
    if not isinstance(da, _xr.DataArray): raise RuntimeError("'da' must be a xarray.DataArray")
    if len(da.dims)!=2: raise RuntimeError("'da' must have 2 dimensions")    
    if not all(dim[0] in ['X', 'Y'] for dim in da.dims): 
        raise RuntimeError("'da' must have lon/lat dimensions (e.g., X and Y, or Xp1 and Yp1, ...")     
    
    # Import useful stuff
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    
    # Assign lon and lat
    for dim in da.dims:
        if   dim[0] == 'X': lon=da[dim]
        elif dim[0] == 'Y': lat=da[dim]
    
    # Plot
    ax = _plt.axes(projection=_ccrs.Mercator(lon.values.mean(), lat.values.min(), lat.values.max()))
    gl = ax.gridlines(crs=_ccrs.PlateCarree(), draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    da.plot.pcolormesh(ax=ax, 
                       transform=_ccrs.PlateCarree());
        
    return ax, gl    
    
