.. _installation:

============
Installation
============

If you would like to use OceanSpy for your own datasets and run it on a local machine or server, you will need to download and install it first. Use the following 3 steps to install OceanSpy.

1. First, download and install Miniconda_ or Anaconda_.

2. Second, OceanSpy and its dependencies need to be installed. The easiest way to install OceanSpy's dependencies is to use the conda-forge channel. 

  Open a terminal, then run the following commands:

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

	Then, activate the Ocenography environment:

	.. code-block:: bash

		$ conda activate Oceanography

.. _Anaconda: https://www.anaconda.com/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _`Create an environment`: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
