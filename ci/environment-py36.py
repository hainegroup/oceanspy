name: test_env
channels:
  - conda-forge  
dependencies:
  - python=3.6
  - xarray
  - dask
  - bottleneck
  - netCDF4
  - cartopy
  - pytest
  - pip:
    - xgcm
    - codecov
    - pytest-cov
