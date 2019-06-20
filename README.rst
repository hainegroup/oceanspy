.. _readme:

======================================================================================
OceanSpy - A Python package to facilitate ocean model data analysis and visualization.
======================================================================================

|OceanSpy|

|version| |conda forge| |docs| |travis| |codecov| |license|

What is OceanSpy?
-----------------
**OceanSpy** is an open-source and user-friendly Python package that enables scientists and interested amateurs to analyze and visualize ocean model datasets. 
OceanSpy builds on software packages developed by the Pangeo_ community, in particular xarray_, dask_, and xgcm_. 
The integration of dask facilitates scalability, which is important for the petabyte-scale simulations that are becoming available. 

Why OceanSpy?
-------------
Simulations of ocean currents using numerical circulation models are becoming increasingly realistic.
At the same time, these models generate increasingly large volumes of model output data, making the analysis of model data harder.
Using OceanSpy, model data can be easily analyzed in the way observational oceanographers analyze field measurements.

How to use OceanSpy?
--------------------
OceanSpy can be used as a standalone package for analysis of local circulation model output, or it can be run on a remote data-analysis cluster, such as the Johns Hopkins University SciServer_ system, which hosts several simulations and is publicly available (see `SciServer Access`_, and `Datasets`_).

.. note::

   OceanSpy has been developed and tested using MITgcm output. However, it is designed to work with any (structured grid) ocean general circulation model. OceanSpy's architecture allows to easily implement model-specific features, such as different grids, numerical schemes for vector calculus, budget closures, and equations of state. We actively seek input and contributions from users of other ocean models (`feedback submission`_).




.. _Pangeo: http://pangeo-data.github.io
.. _xarray: http://xarray.pydata.org
.. _dask: https://dask.org
.. _xgcm: https://xgcm.readthedocs.io
.. _SciServer: http://www.sciserver.org
.. _`SciServer Access`: https://oceanspy.readthedocs.io/en/latest/sciserver.html
.. _Datasets: https://oceanspy.readthedocs.io/en/latest/datasets.html
.. _`feedback submission`: https://github.com/malmans2/oceanspy/issues

.. |OceanSpy| image:: https://github.com/malmans2/oceanspy/raw/master/docs/_static/oceanspy_logo_blue.png
   :alt: OceanSpy image
   :target: https://oceanspy.readthedocs.io

.. |version| image:: https://img.shields.io/pypi/v/oceanspy.svg?style=flat
    :alt: PyPI
    :target: https://pypi.python.org/pypi/oceanspy

.. |conda forge| image:: https://anaconda.org/conda-forge/oceanspy/badges/version.svg
   :alt: conda-forge
   :target: https://anaconda.org/conda-forge/xgcm

.. |docs| image:: http://readthedocs.org/projects/oceanspy/badge/?version=latest
    :alt: Documentation
    :target: http://oceanspy.readthedocs.io/en/latest/?badge=latest

.. |travis| image:: https://travis-ci.org/malmans2/oceanspy.svg?branch=master
    :alt: Travis
    :target: https://travis-ci.org/malmans2/oceanspy
    
.. |codecov| image:: https://codecov.io/github/malmans2/oceanspy/coverage.svg?branch=master
    :alt: Coverage
    :target: https://codecov.io/github/malmans2/oceanspy?branch=master

.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :alt: License
   :target: https://github.com/malmans2/oceanspy
