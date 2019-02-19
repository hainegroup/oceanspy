.. currentmodule:: oceanspy

###
API 
###

Creating an OceanDataset
========================

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

Methods
-------

.. autosummary::
   :toctree: generated/
   
    OceanDataset.add_DataArray
    OceanDataset.merge_Dataset
    OceanDataset.to_netcdf
    OceanDataset.set_name
    OceanDataset.set_description
    OceanDataset.set_parameters
    OceanDataset.set_aliases
    OceanDataset.set_grid_coords
    OceanDataset.set_grid_periodic
    OceanDataset.set_coords
    OceanDataset.create_tree

Shortcuts
---------


Subsample
^^^^^^^^^

.. autosummary::
   :toctree: generated/
   
   OceanDataset.cutout
   OceanDataset.mooring_array
   OceanDataset.survey_stations
   OceanDataset.particle_properties


Compute
^^^^^^^

.. autosummary::
   :toctree: generated/
   
   OceanDataset.merge_gradient
   OceanDataset.merge_divergence
   OceanDataset.merge_curl
   OceanDataset.merge_laplacian
   OceanDataset.merge_volume_weighted_mean
   OceanDataset.merge_potential_density_anomaly
   OceanDataset.merge_Brunt_Vaisala_frequency
   OceanDataset.merge_vertical_relative_vorticity
   OceanDataset.merge_relative_vorticity
   OceanDataset.merge_kinetic_energy
   OceanDataset.merge_eddy_kinetic_energy
   OceanDataset.merge_horizontal_divergence_velocity
   OceanDataset.merge_shear_strain
   OceanDataset.merge_normal_strain
   OceanDataset.merge_Okubo_Weiss_parameter
   OceanDataset.merge_Ertel_potential_vorticity
   OceanDataset.merge_horizontal_volume_transport
   OceanDataset.merge_heat_budget
   OceanDataset.merge_salt_budget
   OceanDataset.merge_aligned_velocities
   

Opening an OceanDataset
=======================

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
   
   





Subsampling an OceanDataset
===========================

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
   
   
   
Computing from an OceanDataset
==============================

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
   compute.horizontal_volume_transport
   compute.heat_budget
   compute.salt_budget
   compute.aligned_velocities
   
OceanSpy utilities
==================

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