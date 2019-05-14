.. _installation:

============
Installation
============

First download and install Miniconda_ or Anaconda_.

The easiest way to install OceanSpy's dependencies is to use the conda-forge channel.
Open a terminal, then run the following commands:

.. code-block:: bash
    
    $ conda config --set channel_priority strict
    $ conda config --prepend channels conda-forge
    $ conda install -y dask distributed bottleneck netCDF4 xarray cartopy esmpy ffmpeg intake-xarray tqdm geopy xgcm xesmf xmitgcm Ipython tqdm oceanspy 
    $ pip install --no-deps --force-reinstall git+https://github.com/xgcm/xmitgcm.git

The commands above install the latest stable release of OceanSpy.
Add the following command to install the latest development version of OceanSpy:

.. code-block:: bash

    $ pip install --no-deps --force-reinstall git+https://github.com/malmans2/oceanspy.git

.. _Anaconda: https://www.anaconda.com/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
