.. _readme:

=====================================================================================
OceanSpy - A Python package to facilitate ocean model data analysis and visualization
=====================================================================================

.. list-table::
    :stub-columns: 1
    :widths: 10 90

    * - docs
      - |docs|
    * - tests
      - |travis| |codecov|
    * - package
      - |version| |supported-versions| |license|

.. |docs| image:: http://readthedocs.org/projects/oceanspy/badge/?version=latest
    :alt: Documentation Status
    :target: http://oceanspy.readthedocs.io/en/latest/?badge=latest

.. |travis| image:: https://travis-ci.org/malmans2/oceanspy.svg?branch=master
    :alt: Travis
    :target: https://travis-ci.org/malmans2/oceanspy
    
.. |codecov| image:: https://codecov.io/github/malmans2/oceanspy/coverage.svg?branch=master
    :alt: Coverage
    :target: https://codecov.io/github/malmans2/oceanspy?branch=master

.. |version| image:: https://img.shields.io/pypi/v/oceanspy.svg?style=flat
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/oceanspy

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/oceanspy.svg?style=flat
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/oceanspy
    
.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :alt: License
   :target: https://github.com/malmans2/oceanspy

**OceanSpy** is an open-source and user-friendly Python package that enables scientists and 
interested amateurs to analyze and visualize oceanographic data sets. 
OceanSpy builds on software packages developed by the Pangeo_ community, in particular xarray_, dask_, and xgcm. 
The integration of dask facilitates scalability, which is important for the petabyte-scale simulations that are becoming available. 
OceanSpy can be used as a standalone package for analysis of local circulation model output, 
or it can be run on a remote data-analysis cluster, such as the Johns Hopkins University SciServer_ system, 
which hosts several simulations (`here <api.rst#datasets-available-on-sciserver>`_ is the list of simulations publicly available).

OceanSpy enables extraction, processing, and visualization of model data to:

1. Compare with oceanographic observations.
2. Portray the kinematic and dynamic space-time properties of the circulation.

.. _Pangeo: http://pangeo-data.github.io
.. _xarray: http://xarray.pydata.org
.. _dask: https://dask.org
.. _xgcm: https://xgcm.readthedocs.io
.. _SciServer: http://www.sciserver.org
