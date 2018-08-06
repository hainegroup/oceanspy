.. highlight:: shell

============
Installation
============

SciServer Access
----------------
SciServer Compute_ optimizes Big Data science by allowing users to bring their analysis close to the data with Jupyter Notebooks deployed in server-side containers.

1. Register_  for a new account or `log in`_ to an existing account 
2. Create a new container and select
 
   .. list-table::
    :stub-columns: 0
    :widths: 60 60

    * - Image:
      - Python (astro)
    * - Public Volumes:
      - Ocean Circulation

3. Click on the green play button 

.. _dependencies:

Install Dependencies
--------------------
The easiest way to install most of OceanSpy's dependencies is to use Conda_.
First open a terminal by clicking on ``New`` + ``Terminal`` (top right), then run the following commands:

.. code-block:: bash

    conda install dask distributed bottleneck netCDF4
    conda install -c conda-forge xarray cartopy esmpy 
    conda install -c pyviz hvplot geoviews
    pip install xgcm xesmf

Install OceanSpy
----------------
To install OceanSpy, run this command in your terminal:

.. code-block:: bash

    pip install oceanspy

This is the preferred method to install OceanSpy, as it will always install the most recent stable release.

Install from Python
-------------------
To install OceanSpy and its dependencies from Python, use these commands::

    import sys
    !conda install --yes --prefix {sys.prefix} dask distributed bottleneck netCDF4
    !conda install --yes --prefix {sys.prefix} -c conda-forge xarray cartopy esmpy 
    !conda install --yes --prefix {sys.prefix} -c pyviz hvplot geoviews
    !{sys.executable} -m pip install xgcm xesmf

.. _SciServer: http://www.sciserver.org
.. _Compute: http://compute.sciserver.org/dashboard/Home/Index
.. _Register: http://portal.sciserver.org/login-portal/Account/Register
.. _log in: http://portal.sciserver.org/login-portal/Account/Login?callbackUrl=http:%2f%2fcompute.sciserver.org%2fdashboard
.. _Conda: https://conda.io/docs
