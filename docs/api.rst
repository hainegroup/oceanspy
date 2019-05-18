.. _api:

.. currentmodule:: oceanspy

#############
API reference 
#############

OceanDataset
============

.. autosummary::
   :toctree: generated/
   
   OceanDataset

Import
------

.. autosummary::
   :toctree: generated/
   
    OceanDataset.shift_averages
    OceanDataset.manipulate_coords

Attributes
----------

.. autosummary::
   :toctree: generated/

    OceanDataset.name
    OceanDataset.description
    OceanDataset.dataset
    OceanDataset.grid
    OceanDataset.parameters
    OceanDataset.aliases
    OceanDataset.grid_coords
    OceanDataset.grid_periodic
    OceanDataset.projection

Set
----

.. autosummary::
   :toctree: generated/
   
    OceanDataset.set_name
    OceanDataset.set_description
    OceanDataset.set_parameters
    OceanDataset.set_aliases
    OceanDataset.set_grid_coords
    OceanDataset.set_grid_periodic
    OceanDataset.set_projection
    
Methods
-------

.. autosummary::
   :toctree: generated/
   
    OceanDataset.merge_into_oceandataset
    OceanDataset.to_netcdf
    OceanDataset.to_zarr
    OceanDataset.create_tree
    
Shortcuts
---------

.. autosummary::
   :toctree: generated/
   
   OceanDataset.subsample
   OceanDataset.compute
   OceanDataset.plot
   OceanDataset.animate
   
   

Opening 
=======

.. autosummary::
   :toctree: generated/
   
    open_oceandataset

.. autosummary::
   :toctree: generated/
   
    open_oceandataset.from_netcdf
    open_oceandataset.from_zarr
    open_oceandataset.from_catalog

The filename of catalogs must end with either `'xarray.yaml'` or `'xmitgcm.yaml'`.
Catalogs ending with `'xarray.yaml'` will be decoded using intake-xarray_, while catologs ending with `'xmitgcm.yaml'` will be decoded using xmitgcm.open_mdsdataset_. The arguments for xmitgcm.open_mdsdataset_ and  intake-xarray_ must be added under `args` (see `xmitgcm catalog`_ and `xarray catalog`_).  

Entries containing the requested `'name'` will be merged (e.g., grd_get_started, kpp_get_started, fld_get_started, avg_get_started. See `xarray catalog`_).
It is possible to select subdomains for each entry using xarray.isel_, and to rename variables using xarray.rename_ (see the metadata arguments of `get_started` entries in `xarray catalog`_). 
If an entry contains snapshots or averages only, the argument `original_output='snapshot'` or `original_output='average'` can be used (see get_started in `xarray catalog`_).

OceanSpy attributes and `Import`_ parameters can be provided adding arguments under `metadata`.
These attributes/parameters will be processed after merging the entries containing the requested `'name'`, so they must be provided for one entry only (see grd_get_started in `xarray catalog`_).

.. _intake-xarray: https://github.com/intake/intake-xarray
.. _xmitgcm.open_mdsdataset: https://xmitgcm.readthedocs.io/en/latest/usage.html#open-mdsdataset
.. _`xmitgcm catalog`: https://github.com/malmans2/oceanspy/blob/master/oceanspy/catalog_xmitgcm.yaml
.. _`xarray catalog`: https://github.com/malmans2/oceanspy/blob/master/oceanspy/catalog_xarray.yaml
.. _xarray.isel: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.isel.html
.. _xarray.rename: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.rename.html

Subsampling
===========

.. autosummary::
   :toctree: generated/
   
   subsample
   
Functions
---------
.. autosummary::
   :toctree: generated/
   
   subsample.cutout
   subsample.mooring_array
   subsample.survey_stations
   subsample.particle_properties
    
Computing
=========

.. autosummary::
   :toctree: generated/
   
   compute
   
Smart-name
----------
Computed variables are dynamically named. 
Names depend on input and operation.

.. autosummary::
   :toctree: generated/
   
   compute.gradient
   compute.divergence
   compute.curl
   compute.laplacian
   compute.integral
   compute.weighted_mean
   
Fixed-name
----------
Computed variables have a hard-coded name.

.. autosummary::
   :toctree: generated/
   
   compute.potential_density_anomaly
   compute.Brunt_Vaisala_frequency
   compute.velocity_magnitude
   compute.horizontal_velocity_magnitude
   compute.vertical_relative_vorticity
   compute.relative_vorticity
   compute.kinetic_energy
   compute.eddy_kinetic_energy
   compute.horizontal_divergence_velocity
   compute.shear_strain
   compute.normal_strain
   compute.Okubo_Weiss_parameter
   compute.Ertel_potential_vorticity
   compute.mooring_volume_transport
   compute.heat_budget
   compute.salt_budget
   compute.geographical_aligned_velocities
   compute.survey_aligned_velocities
   compute.missing_horizontal_spacing

   
Plotting
========

.. autosummary::
   :toctree: generated/
   
   plot
   
Functions
---------

.. autosummary::
   :toctree: generated/
   
   plot.TS_diagram
   plot.time_series
   plot.horizontal_section
   plot.vertical_section


Animating
=========

.. autosummary::
   :toctree: generated/
   
   animate
   
Functions
---------

.. autosummary::
   :toctree: generated/
   
   animate.TS_diagram
   animate.horizontal_section
   animate.vertical_section

Utilities
=========

.. autosummary::
   :toctree: generated/
   
   utils
   
Functions
---------

.. autosummary::
   :toctree: generated/
   
   utils.spherical2cartesian
   utils.great_circle_path
   utils.cartesian_path
   utils.densjmd95
   utils.densmdjwf
   utils.Coriolis_parameter
