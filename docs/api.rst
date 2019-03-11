.. currentmodule:: oceanspy

###
API 
###

OceanDataset
============

.. autosummary::
   :toctree: generated/
   
   OceanDataset
   
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

Import
------

.. autosummary::
   :toctree: generated/
   
    OceanDataset.import_MITgcm_rect_nc
    OceanDataset.import_MITgcm_rect_bin
    OceanDataset.import_MITgcm_curv_nc
    
Others
------

.. autosummary::
   :toctree: generated/
   
    OceanDataset.merge_into_oceandataset
    OceanDataset.to_netcdf
    OceanDataset.create_tree
    
Shortcuts
---------

.. autosummary::
   :toctree: generated/
   
   OceanDataset.subsample
   OceanDataset.compute
   
   

Opening 
=======

.. autosummary::
   :toctree: generated/
   
    open_oceandataset
   
SciServer
---------

.. autosummary::
   :toctree: generated/
   
    open_oceandataset.get_started
    open_oceandataset.EGshelfIIseas2km_ERAI
    open_oceandataset.EGshelfIIseas2km_ASR
    open_oceandataset.EGshelfSJsec500m
   
Others
------
   
.. autosummary::
   :toctree: generated/

    open_oceandataset.from_netcdf




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
   
Dynamic
-------

.. autosummary::
   :toctree: generated/
   
   compute.gradient
   compute.divergence
   compute.curl
   compute.laplacian
   compute.integral
   compute.weighted_mean
   
Static
------

.. autosummary::
   :toctree: generated/
   
   compute.potential_density_anomaly
   compute.Brunt_Vaisala_frequency
   compute.vertical_relative_vorticity
   compute.relative_vorticity
   compute.kinetic_energy
   compute.eddy_kinetic_energy
   compute.horizontal_divergence_velocity
   compute.shear_strain
   compute.normal_strain
   compute.Okubo_Weiss_parameter
   compute.Ertel_potential_vorticity
   compute.mooring_horizontal_volume_transport
   compute.heat_budget
   compute.salt_budget
   compute.geographical_aligned_velocities
   compute.survey_aligned_velocities
   
   
   
Plotting
========

.. autosummary::
   :toctree: generated/
   
   plot
   
Functions
---------

.. autosummary::
   :toctree: generated/
   
   plot.vertical_section
   plot.horizontal_section
   plot.time_series
   plot.TS_diagram
   


Animating
=========

.. autosummary::
   :toctree: generated/
   
   animate
   
Functions
---------

.. autosummary::
   :toctree: generated/
   
   animate.vertical_section
   animate.horizontal_section
   animate.TS_diagram

   
   
   
   
   
   
   
   
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
   
   
