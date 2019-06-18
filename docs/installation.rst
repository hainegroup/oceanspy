.. _installation:

============
Installation
============

If you would like to use OceanSpy for your own datasets and run it on a local machine or server, you will need to download and install it first.

Required dependencies
---------------------

* Python 3.6, 3.7
* dask_
* xarray_
* xgcm_

Optional dependencies
---------------------

For optimal performance and to enable all OceanSpy features, it is highly recommended that you install the following dependencies:

* bottleneck_  
* Cartopy_  
* Dask.distributed_  
* ESMPy_  
* FFmpeg_  
* GeoPy_  
* intake_xarray_  
* IPython_  
* netCDF4_  
* tqdm_  
* xESMF_  
* xmitgcm_  

Step-by-step instructions
-------------------------

1. First, download and install Miniconda_ or Anaconda_.

2. The easiest way to install OceanSpy and the above mentioned dependencies is to use the conda-forge channel. Open a terminal, then run the following commands:

  .. code-block:: bash
    
    $ conda update conda
    $ conda config --set channel_priority strict
    $ conda config --prepend channels conda-forge
    $ conda install -y oceanspy dask distributed bottleneck netCDF4 "xarray>=0.11.3" cartopy esmpy ffmpeg intake-xarray geopy "xgcm>=0.2" xesmf Ipython tqdm "xmitgcm>=0.3"
    $ pip install ffmpeg

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

Test suite
----------
Step-by-step instructions on how to run the test suite are available in :ref:`code`.

.. _xarray: http://xarray.pydata.org
.. _dask: https://dask.org
.. _xgcm: https://xgcm.readthedocs.io
.. _Anaconda: https: //www.anaconda.com/
.. _Miniconda: https: //docs.conda.io/
.. _bottleneck: https://github.com/kwgoodman/bottleneck
.. _Cartopy: https://scitools.org.uk/cartopy
.. _Dask.distributed: http://distributed.dask.org
.. _ESMPy: https://www.earthsystemcog.org/projects/esmpy/
.. _FFmpeg: https://ffmpeg.org/
.. _GeoPy: https://github.com/geopy/geopy
.. _intake_xarray: https://github.com/intake/intake-xarray
.. _IPython: https://ipython.org/
.. _netCDF4: https://unidata.github.io/netcdf4-python
.. _tqdm: https://tqdm.github.io/
.. _xESMF: https://xesmf.readthedocs.io/
.. _xmitgcm: https://xmitgcm.readthedocs.io/
.. _`Create an environment`: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
