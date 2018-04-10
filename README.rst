============================================================
OceanSpy - A Python Package for Oceanographic Investigations
============================================================

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
      
**OceanSpy** is a python package that facilitates extracting information from numerical model output of Ocean General Circulation Models set up and run by the research group of `Prof. Thomas W. N. Haine <http://sites.krieger.jhu.edu/haine/>`_. The dynamics are simulated using the Massachussets Institute of Technology general circulation model (MITgcm), and high-resolution data are publicly available on `SciServer <http://www.sciserver.org/>`_. SciServer is a collaborative research environment for large-scale data-driven science administered by `IDIES <http://idies.jhu.edu/>`_ at  `Johns Hopkins University <https://www.jhu.edu/>`_.

The analysis of large datasets is often restricted by limited computation resources. Our goal is to build a collaborative sharing environment where users can access and process high-resolution datasets. OceanSpy aims to allow users to trace the physical evolution of ocean currents across orders of magnitude in space and time, and to quickly analyze important aspects of model events in conjunction with observational data.

Features
--------
* OceanSpy's documentation is available at: https://oceanspy.readthedocs.io/en/latest/ 
* OceanSpy is meant to be user-friendly and does not require advanced coding skills.
* SciServer users can either download subsets of data on their own machines, or run our tools online and store/visualize post-processing files on SciServer.

Credits
-------
OceanSpy is based on several tools and packages that are part of the `Pangeo <https://pangeo-data.github.io/>`_ initiative, such as `xarray <http://xarray.pydata.org/en/stable/>`_ and `dask <https://dask.pydata.org/en/latest/>`_.

This package was created with `Cookiecutter <https://github.com/audreyr/cookiecutter>`_ and the `audreyr/cookiecutter-pypackage <https://github.com/audreyr/cookiecutter-pypackage>`_ project template.
