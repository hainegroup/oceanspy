.. _readme:

======================================================================================
OceanSpy - A Python package to facilitate ocean model data analysis and visualization.
======================================================================================

|OceanSpy|

|version| |conda forge| |docs| |CI| |pre-commit| |codecov| |black| |license| |doi| |JOSS| |binder|

.. admonition:: Interactive Demo

   Check out the interactive demonstration of OceanSpy at `www.bndr.it/gfvgd <https://bndr.it/gfvgd>`_

For publications, please cite the following paper:

Almansi, M., R. Gelderloos, T. W. N. Haine, A. Saberi, and A. H. Siddiqui (2019). OceanSpy: A Python package to facilitate ocean model data analysis and visualization. *Journal of Open Source Software*, 4(39), 1506, doi: https://doi.org/10.21105/joss.01506 .

This material is based upon work supported by the National Science Foundation under Grant Numbers 1835640, 124330, 118123, and 1756863. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

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
.. _`feedback submission`: https://github.com/hainegroup/oceanspy/issues

.. |OceanSpy| image:: https://github.com/hainegroup/oceanspy/raw/main/docs/_static/oceanspy_logo_blue.png
   :alt: OceanSpy image
   :target: https://oceanspy.readthedocs.io

.. |version| image:: https://img.shields.io/pypi/v/oceanspy.svg?style=flat
    :alt: PyPI
    :target: https://pypi.python.org/pypi/oceanspy

.. |conda forge| image:: https://anaconda.org/conda-forge/oceanspy/badges/version.svg
   :alt: conda-forge
   :target: https://anaconda.org/conda-forge/oceanspy

.. |docs| image:: http://readthedocs.org/projects/oceanspy/badge/?version=latest
    :alt: Documentation
    :target: http://oceanspy.readthedocs.io/en/latest/?badge=latest

.. |CI| image:: https://img.shields.io/github/workflow/status/hainegroup/oceanspy/CI?logo=github
    :alt: CI
    :target: https://github.com/hainegroup/oceanspy/actions

.. |codecov| image:: https://codecov.io/github/hainegroup/oceanspy/coverage.svg?branch=main
    :alt: Coverage
    :target: https://codecov.io/github/hainegroup/oceanspy?branch=main

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :alt: black
    :target: https://github.com/psf/black

.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :alt: License
   :target: https://github.com/hainegroup/oceanspy

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3270646.svg
   :alt: doi
   :target: https://doi.org/10.5281/zenodo.3270646

.. |JOSS| image:: http://joss.theoj.org/papers/10.21105/joss.01506/status.svg
   :alt: JOSS
   :target: https://doi.org/10.21105/joss.01506

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :alt: binder
   :target: https://mybinder.org/v2/gh/hainegroup/oceanspy.git/main?filepath=binder

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/hainegroup/oceanspy/main.svg
   :target: https://results.pre-commit.ci/latest/github/hainegroup/oceanspy/main
   :alt: pre-commit.ci status
