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

[comment]: <>    OceanDataset.name
[comment]: <>    OceanDataset.description
[comment]: <>    OceanDataset.dataset
[comment]: <>    OceanDataset.grid
[comment]: <>    OceanDataset.parameters
[comment]: <>    OceanDataset.aliases
[comment]: <>    OceanDataset.grid_coords
[comment]: <>    OceanDataset.grid_periodic
[comment]: <>    OceanDataset.projection

Methods
-------

.. autosummary::
   :toctree: generated/
   
    OceanDataset.import_MITgcm_rect_nc
    OceanDataset.import_MITgcm_rect_bin
    OceanDataset.import_MITgcm_curv_nc
    
[comment]: <>    OceanDataset.add_DataArray
[comment]: <>    OceanDataset.merge_Dataset
[comment]: <>    OceanDataset.to_netcdf
[comment]: <>    OceanDataset.set_name
[comment]: <>    OceanDataset.set_description
[comment]: <>    OceanDataset.set_parameters
[comment]: <>    OceanDataset.set_aliases
[comment]: <>    OceanDataset.set_grid_coords
[comment]: <>    OceanDataset.set_grid_periodic
[comment]: <>    OceanDataset.set_projection
[comment]: <>    OceanDataset.create_tree

Shortcuts
---------


Subsample
^^^^^^^^^

.. autosummary::
   :toctree: generated/
   
[comment]: <>   OceanDataset.cutout
[comment]: <>   OceanDataset.mooring_array
[comment]: <>   OceanDataset.survey_stations
[comment]: <>   OceanDataset.particle_properties


Compute
^^^^^^^

.. autosummary::
   :toctree: generated/
   
[comment]: <>   OceanDataset.merge_gradient
[comment]: <>   OceanDataset.merge_divergence
[comment]: <>   OceanDataset.merge_curl
[comment]: <>   OceanDataset.merge_laplacian
[comment]: <>   OceanDataset.merge_volume_cells
[comment]: <>   OceanDataset.merge_volume_weighted_mean
[comment]: <>   OceanDataset.merge_potential_density_anomaly
[comment]: <>   OceanDataset.merge_Brunt_Vaisala_frequency
[comment]: <>   OceanDataset.merge_vertical_relative_vorticity
[comment]: <>   OceanDataset.merge_relative_vorticity
[comment]: <>   OceanDataset.merge_kinetic_energy
[comment]: <>   OceanDataset.merge_eddy_kinetic_energy
[comment]: <>   OceanDataset.merge_horizontal_divergence_velocity
[comment]: <>   OceanDataset.merge_shear_strain
[comment]: <>   OceanDataset.merge_normal_strain
[comment]: <>   OceanDataset.merge_Okubo_Weiss_parameter
[comment]: <>   OceanDataset.merge_Ertel_potential_vorticity
[comment]: <>   OceanDataset.merge_mooring_horizontal_volume_transport
[comment]: <>   OceanDataset.merge_heat_budget
[comment]: <>   OceanDataset.merge_salt_budget
[comment]: <>   OceanDataset.merge_geographical_aligned_velocities
[comment]: <>   OceanDataset.merge_survey_aligned_velocities
   
Plot
^^^^

.. autosummary::
   :toctree: generated/
   
[comment]: <>   OceanDataset.vertical_section
[comment]: <>   OceanDataset.horizontal_section
[comment]: <>   OceanDataset.time_series
[comment]: <>   OceanDataset.TS_diagram
   
   
   
   
   

Opening 
=======

.. autosummary::
   :toctree: generated/
   
   open_oceandataset
   
Functions
---------

.. autosummary::
   :toctree: generated/
   
   open_oceandataset.from_netcdf
   open_oceandataset.EGshelfIIseas2km_ERAI
   open_oceandataset.EGshelfIIseas2km_ASR
   open_oceandataset.EGshelfSJsec500m
   
   





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
   
Functions
---------

.. autosummary::
   :toctree: generated/
   
   compute.gradient
   compute.divergence
   compute.curl
   compute.laplacian
   compute.volume_cells
   compute.volume_weighted_mean
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
   
   
