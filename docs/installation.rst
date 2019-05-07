.. _installation:

============
Installation
============
.. _from_terminal:

Install from terminal
---------------------
The easiest way to install OceanSpy's dependencies is to use conda-forge_.
First open a terminal, then run the following commands:

.. code-block:: bash

    conda config --remove channels defaults
    conda config --add channels conda-forge
    conda install -y dask distributed bottleneck netCDF4 xarray cartopy esmpy ffmpeg intake-xarray tqdm geopy xgcm xesmf oceanspy
    pip install --no-deps --force-reinstall git+https://github.com/xgcm/xmitgcm.git

Run the following command to install the latest version of OceanSpy:

.. code-block:: bash

    pip install git+https://github.com/malmans2/oceanspy.git

SciServer Access
----------------
SciServer_ optimizes Big Data science by allowing users to bring their analysis close to the data with Jupyter Notebooks deployed in server-side containers.
Several Apps_ are available on SciServer: use Compute Interact to analyze data with an interactive notebook; use Compute Jobs to run notebooks asynchronously.

Compute Interact:

1. Go to Apps_ and register for a new account or log in to an existing account
2. Click on Compute Interact
3. Create a new container and select
 
   .. list-table::
    :stub-columns: 0
    :widths: 60 60

    * - Compute Image:
      - Oceanography
    * - Data volumes:
      - Ocean Circulation

4. Click on the container
5. A stable release of OceanSpy and its dependencies are already installed in the Oceanography image. Run the following command to install the latest version of OceanSpy:

.. code-block:: bash

    pip install --no-deps --force-reinstall git+https://github.com/malmans2/oceanspy.git

Compute Jobs:

1. Go to Apps_ and register for a new account or log in to an existing account
2. Click on Compute Jobs
3. Click on Run Existing Notebook
4. Select a Compute Domain
5. Select the ``Oceanography`` image
6. Select the ``Ocean Circulation`` data volume
7. Select a User Volume and the Notebook to run

To install the latest version of OceanSpy in the Compute Jobs environment, add the following cell on top of the notebook that will be executed:

.. code-block:: ipython
    :class: no-execute

    import sys
    !{sys.executable} -m pip install --no-deps --force-reinstall git+https://github.com/malmans2/oceanspy.git
    
.. note::
    It is possible that the latest version of OceanSpy has different dependencies than the stable release. In that case, also install OceanSpy's dependencies using:

    * From terminal: ``conda install -y``
    * From notebook: ``!conda install --yes --prefix {sys.prefix}``

    OceanSpy's dependencies that need to be added to the commands above are listed here: :ref:`from_terminal`.

.. _SciServer: http://www.sciserver.org
.. _Apps: https://apps.sciserver.org
.. _Conda: https://conda.io/docs
.. _conda-forge: https://conda-forge.org/
