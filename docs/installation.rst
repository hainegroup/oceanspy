.. _installation:

============
Installation
============

If you would like to use OceanSpy for your own datasets and run it on a local machine or server, you will need to download and install it first. Use the following 3 steps to install OceanSpy.

1. First, download and install Miniconda_ or Anaconda_.

2. Second, OceanSpy and its dependencies need to be installed. The easiest way to install OceanSpy's dependencies is to use the conda-forge channel. 

  For optimal performance, OceanSpy requires the latest version of the following python dependencies to be installed : 

  - Dask.distributed_
  - bottleneck_
  - netCDF4_
  - xarray_
  - Cartopy_
  - ESMPy_
  - FFmpeg_
  - intake_xarray_
  - GeoPy_
  - xgcm_
  - xESMF_
  - IPython_
  - tqdm_

  To install OceanSpy and the above mentioned dependencies, open a terminal, then run the following commands:

  .. code-block:: bash
    
    $ conda update conda
    $ conda config --set channel_priority strict
    $ conda config --prepend channels conda-forge
    $ conda install -y oceanspy dask distributed bottleneck netCDF4 "xarray>=0.11.3" cartopy esmpy ffmpeg intake-xarray geopy "xgcm>=0.2" xesmf Ipython tqdm
    $ pip install "xmitgcm>=0.3" ffmpeg

  The commands above install the latest stable release of OceanSpy along with its dependencies.

3. Finally, in case you would like to use the latest version of OceanSpy under development rather than the stable release, add the following command:

  .. code-block:: bash

    $ pip install --upgrade git+https://github.com/malmans2/oceanspy.git


.. note::
		
	**For experts:** Use the following commands to `Create an environment`_ identical to the Oceanography environment available on SciServer:

	.. code-block:: bash

		$ conda update conda
		$ conda config --set channel_priority strict
		$ conda config --prepend channels conda-forge
		$ wget https://raw.githubusercontent.com/malmans2/oceanspy/master/sciserver_catalogs/environment.yml
		$ conda env create -f environment.yml

	Then, activate the Oceanography environment:

	.. code-block:: bash

		$ conda activate Oceanography

.. _Anaconda: https: //www.anaconda.com/
.. _Miniconda: https: //docs.conda.io/en/latest/miniconda.html
.. _Dask.distributed : http://distributed.dask.org/en/latest/
.. _bottleneck : https://github.com/kwgoodman/bottleneck
.. _netCDF4 : https://unidata.github.io/netcdf4-python/netCDF4/index.html
.. _xarray : http://xarray.pydata.org/en/stable/
.. _Cartopy : https://scitools.org.uk/cartopy/docs/latest/
.. _ESMPy : https://www.earthsystemcog.org/projects/esmpy/
.. _FFmpeg : https://ffmpeg.org/
.. _intake_xarray : https://github.com/intake/intake-xarray
.. _GeoPy : https://github.com/geopy/geopy
.. _xgcm : https://github.com/xgcm/xgcm
.. _xESMF : https://xesmf.readthedocs.io/en/latest/
.. _IPython : https://ipython.org/
.. _tqdm : https://tqdm.github.io/
.. _`Create an environment`: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
