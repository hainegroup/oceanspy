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
      - Geo
    * - Data volumes:
      - Ocean Circulation

4. Click on the container
5. Install OceanSpy and its dependencies

Expert users can run notebooks using the Compute Jobs.

.. note::
    Users won't need to install OceanSpy and its dependencies on SciServer in the future.  

.. warning::
    OceanSpy's interactive plots are currently available for Classical Jupyter only, but they will be available on JupyterLab in the future.

Install OceanSpy from Terminal
------------------------------
The easiest way to install most of OceanSpy's dependencies is to use Conda_.
First open a terminal (SciServer: click on ``New`` + ``Terminal``), then run the following commands:

.. code-block:: bash

    conda install dask distributed bottleneck netCDF4
    conda install -c conda-forge xarray cartopy esmpy 
    conda install -c pyviz hvplot geoviews
    pip install xgcm xesmf

To install OceanSpy, run this command in your terminal:

.. code-block:: bash

    pip install oceanspy

This is the preferred method to install OceanSpy, as it will always install the most recent stable release.  
Run the following command to install the latest version:

.. code-block:: bash

    pip install git+https://github.com/malmans2/oceanspy.git

.. note::
    Use ``Shift``+right click to paste into JupyterLab terminal.
    
Install from Jupyter Notebook
-----------------------------
To install OceanSpy and its dependencies from Python, use these commands::

    import sys
    !conda install --yes --prefix {sys.prefix} dask distributed bottleneck netCDF4
    !conda install --yes --prefix {sys.prefix} -c conda-forge xarray cartopy esmpy 
    !conda install --yes --prefix {sys.prefix} -c pyviz hvplot geoviews
    !{sys.executable} -m pip install xgcm xesmf

.. note::
    Users using Compute Jobs currently have to install OceanSpy and its dependencies in the first Notebook cell (this won't be necessary in the future).

.. _SciServer: http://www.sciserver.org
.. _Apps: https://apps.sciserver.org
.. _Conda: https://conda.io/docs
