.. _readme:

======================================================================================
OceanSpy - A Python package to facilitate ocean model data analysis and visualization.
======================================================================================

|OceanSpy|

|docs| |travis| |codecov| |version| |supported-versions| |license|

**OceanSpy** is an open-source and user-friendly Python package that enables scientists and 
interested amateurs to analyze and visualize oceanographic data sets. 
OceanSpy builds on software packages developed by the Pangeo_ community, 
in particular xarray_, dask_, and xgcm_. 
The integration of dask facilitates scalability, which is important for the petabyte-scale simulations that are becoming available. 

OceanSpy can be used as a standalone package for analysis of local circulation model output, 
or it can be run on a remote data-analysis cluster, 
such as the Johns Hopkins University SciServer_ system, 
which hosts several simulations and is publicly available (see :ref:`sciserver`, and :ref:`datasets`).

OceanSpy enables extraction, processing, and visualization of model data to 
(i) compare with oceanographic observations, and 
(ii) portray the kinematic and dynamic space-time properties of the circulation.

.. _Pangeo: http://pangeo-data.github.io
.. _xarray: http://xarray.pydata.org
.. _dask: https://dask.org
.. _xgcm: https://xgcm.readthedocs.io
.. _SciServer: http://www.sciserver.org


.. |OceanSpy| image:: https://github.com/malmans2/oceanspy/raw/master/docs/_static/oceanspy_logo_blue.png
   :alt: OceanSpy image
   :target: https://oceanspy.readthedocs.io

.. |docs| image:: http://readthedocs.org/projects/oceanspy/badge/?version=latest
    :alt: Documentation
    :target: http://oceanspy.readthedocs.io/en/latest/?badge=latest

.. |travis| image:: https://travis-ci.org/malmans2/oceanspy.svg?branch=master
    :alt: Travis
    :target: https://travis-ci.org/malmans2/oceanspy
    
.. |codecov| image:: https://codecov.io/github/malmans2/oceanspy/coverage.svg?branch=master
    :alt: Coverage
    :target: https://codecov.io/github/malmans2/oceanspy?branch=master

.. |version| image:: https://img.shields.io/pypi/v/oceanspy.svg?style=flat
    :alt: PyPI
    :target: https://pypi.python.org/pypi/oceanspy

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/oceanspy.svg?style=flat
    :alt: Python Version
    :target: https://pypi.python.org/pypi/oceanspy
    
.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :alt: License
   :target: https://github.com/malmans2/oceanspy
