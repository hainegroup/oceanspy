========
OceanSpy
========


.. image:: https://img.shields.io/pypi/v/oceanspy.svg
        :target: https://pypi.python.org/pypi/oceanspy

.. image:: https://img.shields.io/travis/malmans2/oceanspy.svg
        :target: https://travis-ci.org/malmans2/oceanspy

.. image:: https://readthedocs.org/projects/oceanspy/badge/?version=latest
        :target: https://oceanspy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/malmans2/oceanspy/shield.svg
     :target: https://pyup.io/repos/github/malmans2/oceanspy/
     :alt: Updates



**OceanSpy** is a python package that facilitates extracting information from numerical model output of Ocean General Circulation Models set up and run by the research group of `Prof. Thomas W. N. Haine <http://sites.krieger.jhu.edu/haine/>`_. The dynamics are simulated using the Massachussets Institute of Technology general circulation model (MITgcm), and our high-resolution datasets are publicly available on `SciServer <http://www.sciserver.org/>`_. SciServer is a collaborative research environment for large-scale data-driven science administered by `IDIES <http://idies.jhu.edu/>`_ at  `Johns Hopkins University <https://www.jhu.edu/>`_.

The analysis of large datasets is often restricted by limited computation resources. Our goal is to build a collaborative sharing environment where users can access and process high-resolution datasets. OceanSpy aims to allow users to trace the physical evolution of ocean currents across orders of magnitude in space and time, and to quickly analyze important aspects of model events in conjunction with observational data.

* Free software: MIT license
* Documentation: https://oceanspy.readthedocs.io.


Features
--------

* SciServer users can either download subsets of data on their own machines, or run our tools online and store/visualize post-processing files on SciServer.
* OceanSpy is meant to be user-friendly and can be easily run by non-python users.

Credits
-------
OceanSpy is based on several tools and packages involved in the `Pangeo <https://pangeo-data.github.io/>`_ community, such as `xarray <http://xarray.pydata.org/en/stable/>`_ and `dask <https://dask.pydata.org/en/latest/>`_.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
