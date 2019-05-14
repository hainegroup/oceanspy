=========
SciServer
=========

Compute Interact
----------------

`Compute Interact`_ allows to analyze data with an interactive notebook. 
Step-by-step instructions for using the interactive mode are available in the `Quick Start section <quick.rst#quick-start>`_.
The interactive mode runs on a Virtual Machine with 16 cores shared between multiple users. 
Use it for notebooks that don’t require heavy computations, or to test and design notebooks.

Compute Jobs
------------

`Compute Jobs`_ allows to run notebooks asynchronously.
Use the job mode to fully exploit the computational power of SciServer. 
For larger jobs (8 hour maximum), you have exclusive access to 32 logical CPU cores and 240GiB of memory.

1. Go to `www.sciserver.org <http://www.sciserver.org/>`_
2. Log in or create a new account.
3. Click on ``Compute``.
4. Click on ``Run Existing Notebook``.
5. Select a ``Compute Domain`` between ``Large Jobs Domain``, or ``Small Jobs Domain``.
6. Click on ``Compute Image`` and select ``Oceanography``.
7. Click on ``Data Volumes`` and select ``Ocean Circulation``.
8. Click on ``User Volumes`` and select the volumes that are needed by the Job (e.g., ``persistent`` and/or ``scratch``).
9. Click on ``Notebook`` and select the Jupyter Notebook that you want to execute. 
10. Select a ``Working Directory``, which is the location where the executed notebook and its output will be stored (you can just use the default ``jobs`` directory that will be created in your ``Temporary volume``).

The ``Oceanography image`` will not include any extra-package installed in your interactive containers.
To install packages that are not available by default on the ``Oceanography image``, add the following lines in the first cell of your notebook:

.. code-block:: ipython
    :class: no-execute

    import sys
    !conda install --yes --prefix {sys.prefix} [list of packages to be installed using conda]
    !{sys.executable} -m pip install [list of packages to be installed using pip]

For example, to install the latest version of OceanSpy, use the following cell:

.. code-block:: ipython
    :class: no-execute

    import sys
    !{sys.executable} -m pip install --no-deps --force-reinstall git+https://github.com/malmans2/oceanspy.git

.. note::
    The ``Oceanography image`` has not been recently updated, so OceanSpy and its dependencies need to be updated.  
    Use the following cell to set up the latest OceanSpy environment:
            
    .. code-block:: ipython
        :class: no-execute
        
        import sys
        !conda config --set channel_priority strict
        !conda config --prepend channels conda-forge
        !conda install --yes --prefix {sys.prefix} dask distributed bottleneck netCDF4 xarray cartopy esmpy ffmpeg intake-xarray tqdm geopy xgcm xesmf
        !{sys.executable} -m pip install --no-deps --force-reinstall git+https://github.com/xgcm/xmitgcm.git        
        !{sys.executable} -m pip install --no-deps --force-reinstall git+https://github.com/malmans2/oceanspy.git

.. _`Compute Interact`: https://apps.sciserver.org/compute/
.. _`Compute Jobs`: https://apps.sciserver.org/compute/jobs
