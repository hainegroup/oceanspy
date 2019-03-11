.. _installation:

============
Installation
============

SciServer Access
----------------
SciServer_ optimizes Big Data science by allowing users to bring their analysis close to the data with Jupyter Notebooks deployed in server-side containers.
Several Apps_ are available on SciServer: use Compute to analyze data with interactive notebook, while use Compute Jobs to asynchronously run notebooks.

1. Go to Apps_ and register for a new account or log in to an existing account
2. Click on Compute
3. Create a new container and select
 
   .. list-table::
    :stub-columns: 0
    :widths: 60 60

    * - Compute Image:
      - Python + R
    * - Data volumes:
      - Ocean Circulation

4. Click on the container
5. Install OceanSpy and its dependencies

.. note::
    Users won't need to install OceanSpy and its dependencies on SciServer in the future.  

Install OceanSpy from Terminal
------------------------------
The easiest way to install most of OceanSpy's dependencies is to use conda-forge_.
First open a terminal (Jupyter Notebook: click on ``New`` + ``Terminal``), then run the following commands:

.. code-block:: bash

    conda config --remove channels defaults
    conda config --add channels conda-forge
    conda install -y dask distributed bottleneck netCDF4 xarray cartopy esmpy ffmpeg cmocean eofs
    pip install geopy xgcm xesmf xmitgcm oceanspy

This is the preferred method to install OceanSpy, as it will always install the most recent stable release.
Run the following command to install the latest version:

.. code-block:: bash

    pip install git+https://github.com/malmans2/oceanspy.git

Install from Jupyter Notebook
-----------------------------

This cell installs OceanSpy and its dependencies from a Jupyter Notebook:

.. code-block:: python
    :class: no-execute

    %%bash
    conda config --remove channels defaults
    conda config --add channels conda-forge
    conda install dask distributed bottleneck netCDF4 xarray cartopy esmpy ffmpeg cmocean eofs
    pip install geopy xgcm xesmf xmitgcm oceanspy

.. note::
    Users using Compute Jobs currently have to install OceanSpy and its dependencies in the first Notebook cell (this won't be necessary in the future).

.. _SciServer: http://www.sciserver.org
.. _Apps: https://apps.sciserver.org
.. _Conda: https://conda.io/docs
.. _conda-forge: https://conda-forge.org/
