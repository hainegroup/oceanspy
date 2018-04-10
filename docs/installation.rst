.. highlight:: shell

============
Installation
============

SciServer access
----------------
1. `Register <http://portal.sciserver.org/login-portal/Account/Register>`_ for a new account or `log in <http://portal.sciserver.org/login-portal/Account/Login?callbackUrl=http:%2f%2fcompute.sciserver.org%2fdashboard>`_ to an existing account 
2. Create a new container and choose
 
.. list-table::
    :stub-columns: 1
    :widths: 60 60

    * - Image:
      - Python (astro)
    * - Public Volumes:
      - Ocean Circulation

3. Click on the green play button 

Dependencies
------------
The easiest way to install all the dependencies is to use `Conda <https://conda.io/docs/>`_.
First open a terminal (click on New-->Terminal), and run the following commands:

.. code-block:: console

    $ conda install dask netCDF4 bottleneck
    $ conda install -c conda-forge xarray cartopy
    $ pip install xgcm

Stable release
--------------
To install OceanSpy, run this command in your terminal:

.. code-block:: console

    $ pip install oceanspy

This is the preferred method to install OceanSpy, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

From sources
------------
The sources for OceanSpy can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/malmans2/oceanspy

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/malmans2/oceanspy/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/malmans2/oceanspy
.. _tarball: https://github.com/malmans2/oceanspy/tarball/master
