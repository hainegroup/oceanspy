.. _readme:

================================================================================
OceanSpy - A Python package for easy ocean model data analysis and visualization
================================================================================

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

**OceanSpy** is an open-source and user-friendly Python package that aims to enable scientists and interested amateurs to use oceanographic data sets with out-of-the-box analysis tools. 
OceanSpy builds on software packages developed by the Pangeo_ community, in particular xarray_, dask_, and xgcm_. 
It can be used as a stand-alone package if the user has access to oceanographic model output, or it can be run on the Johns Hopkins University SciServer_ system, where a year-long high-resolution regional model solution is publicly available; 
moreover, the analysis can be done on the SciServer system, negating the need for the user to own a computing cluster or even download the data.   

OceanSpy aims to fill two needs:

1. Extraction of model data for direct comparison with observational programs.  
2. Facilitating a complete 4D analysis that complements in situ or remote observations, and enable a kinematic and dynamic analysis of diagnostics that cannot be obtained from observations directly.   


.. _Pangeo: http://pangeo-data.github.io
.. _xarray: http://xarray.pydata.org
.. _dask: https://dask.org
.. _xgcm: https://xgcm.readthedocs.io
.. _SciServer: http://www.sciserver.org
