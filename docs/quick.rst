.. _quick:

===========
Quick Start
===========

1. Go to `www.sciserver.org <http://www.sciserver.org/>`_.
2. Log in or create a new account.
3. Click on ``Compute``.
4. Click on ``Create container``, then use the following settings:

    .. list-table::
        :stub-columns: 0
        :widths: 60 60

        * - Domain:
          - Interactive Docker Compute Domain
        * - Compute Image:
          - Oceanography
        * - User volumes:
          - * persistent
            * scratch
        * - Data volumes:
          - Ocean Circulation

5. Click on ``Create``.
6. Click on the name of the new container.
7. Click on ``Storage`` >> ``your_username`` >> ``persistent``.

.. note::
    The ``Oceanography image`` has not been recently updated, so OceanSpy and its dependencies need to be updated.  
    Click on ``New`` >> ``Terminal``, then use the following commands:
    
    .. code-block:: bash

        $ conda config --set channel_priority strict
        $ conda config --prepend channels conda-forge
        $ conda install -y dask distributed bottleneck netCDF4 xarray cartopy esmpy ffmpeg intake-xarray tqdm geopy xgcm xesmf
        $ pip install --no-deps --force-reinstall git+https://github.com/xgcm/xmitgcm.git
        $ pip install --no-deps --force-reinstall git+https://github.com/malmans2/oceanspy.git

8. Click on ``New`` >> ``Python 3``.
9. Copy and paste the following lines in the first notebook cell to import OceanSpy, and open the get started dataset:

.. code-block:: ipython
    :class: no-execute
        
    import oceanspy as ospy
    od = ospy.open_oceandataset.from_catalog('get_started')

10. Use the following line to extract a cutout:

.. code-block:: ipython
    :class: no-execute

    od_cutout = od.subsample.cutout(YRange=[69.6, 71.4], XRange=[-21, -15], ZRange=[0, -100])

11. Use the following line to plot a map of weighted mean temperature:

.. code-block:: ipython
    :class: no-execute

    ax = od_cutout.plot.horizontal_section(varName='Temp', meanAxes=['Z', 'time'], center=False)

12. Use the following line to compute the potential density anomaly:

.. code-block:: ipython
    :class: no-execute
 
    od_cutout = od_cutout.compute.potential_density_anomaly()

13. Use the following line to store the cutout in netCDF format.

.. code-block:: ipython
    :class: no-execute

    od_cutout.to_netcdf('filename.nc')

14. You can either download the netCDF file and continue the post-processing offline, or keep it on SciServer. You can use any software to re-open the netCDF file. To re-open the file using OceanSpy, use the following command:

.. code-block:: ipython
    :class: no-execute

    od_cutout = ospy.open_oceandataset.from_netcdf('filename.nc')

15. Opening the netCDF file using OceanSpy will allow you to use OceanSpy's functions whether you are using SciServer or your own computer. For example, the following line plots an animation of mean potential density anomaly.

.. code-block:: ipython
    :class: no-execute

    anim = od_cutout.animate.horizontal_section(varName='Sigma0', plotType='contourf', meanAxes='Z', vmin=26.5, vmax=27.5)

The get started dataset is just a small cutout from a high-resolution realistic dataset. For a list of datasets available on SciServer, `see here <api.rst#datasets-available-on-sciserver>`_.  
Check out `Tutorial <Tutorial.ipynb#Tutorial>`_, Examples, and `API reference <api.rst>`_ to learn more about OceanSpy and its features.

Feel free to open an `issue here <https://github.com/malmans2/oceanspy/issues>`_, or to send an email to `mattia.almansi@jhu.edu <mattia.almansi@jhu.edu>`_ if you have any questions.
