.. _installation:

============
Installation
============

In case OceanSpy  has to be run on a local machine or server, it needs to be downladed and installed first. Use the following 3 steps to install OceanSpy.

1. First, download and install Miniconda_ or Anaconda_.

2. Secondly, OceanSpy and its dependencies need to be installed. The easiest way to install OceanSpy's dependencies is to use the conda-forge channel. 

  Open a terminal, then run the following commands:

  .. code-block:: bash
    
    $ conda update conda
    $ conda config --set channel_priority strict
    $ conda config --prepend channels conda-forge
    $ conda install -y dask distributed bottleneck netCDF4 xarray cartopy esmpy ffmpeg intake-xarray tqdm geopy xgcm xesmf xmitgcm Ipython tqdm oceanspy 
    $ pip install --upgrade xmitgcm

  The commands above install the latest stable release of OceanSpy along with its dependencies.

3. Finally, in case you would like to use the latest version of OceanSpy under development rather than the stable release, add the following command:

  .. code-block:: bash

    $ pip install --upgrade git+https://github.com/malmans2/oceanspy.git

.. _Anaconda: https://www.anaconda.com/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
