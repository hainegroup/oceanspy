import xarray as xr
import numpy as np

def smart_chunking(ds,
                   limOM      = 6,
                   dims2chunk = ['time', 'X', 'Y', 'Z']):
    """
    Chunk a dataset defining the order of magnitude of the elements in each chunk.

    From xarray's documentation:
    A good rule of thumb to create arrays with a minimum chunksize of at least one million elements. 
    With large arrays (10+ GB), the cost of queueing up dask operations can be noticeable, 
    and you may need even larger chunksizes.

    Parameters
    ----------
    ds: xarray.Dataset or None
       Dataset that will be chunked.
    limOM: int
          Order of magnitude of elements in each chunk.
    dims2chunk: list
               Dimensions to chunk. 
               To minimize chunked dimensions, it starts from first, then second if necessary
               Available dimensions are ['time', 'X', 'Y', 'Z']
    
    Returns
    -------
    ds: xarray.Dataset
       Chunked Dataset 
    """
    
    # Check parameters
    if not isinstance(ds, xr.Dataset):   raise RuntimeError("'ds' needs to be a xarray.Dataset")
    if not isinstance(limOM, int):       raise RuntimeError("'limOM' needs to be an integer")
    if not isinstance(dims2chunk, list): raise RuntimeError("'dims2chunk' needs to be a list")
        
    # Get dimensions' size
    OM=6
    dims2chunk=['time', 'X', 'Y', 'Z']
    chunks = {}
    for dim in dims2chunk: chunks[dim] = ds[dim].size

    # Loop reemoving 1 every time
    totSize = 1
    for key in chunks: 
        totSize=totSize * chunks[key]
    totOM = int(np.log10(totSize))
    while totOM>OM: 
        for dim in dims2chunk:
            if chunks[dim]>1:
                chunks[dim] = chunks[dim]-1
                totSize = 1
                for key in chunks: 
                    totSize=totSize * chunks[key]
                totOM = int(np.log10(totSize))
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

