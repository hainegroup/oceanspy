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
   subsample.stations

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
   plot.faces_array


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
   utils.viewer2range
   utils._rel_lon
   utils._reset_range
   utils.circle_path_array
   utils.remove_repeated
   utils.reset_dim
   utils.diff_and_inds_where_insert
   utils.connector


LLC-transformation
==================
.. autosummary::
   :toctree: generated/

   llc_rearrange


Class
---------
.. autosummary::
   :toctree: generated/

   llc_rearrange.LLCtransformation

.. Classmethod
.. ---------
.. .. autosummary::
   :toctree: generated/

   llc_rearrange.LLCtransformation.arctic_crown


Functions
---------
.. autosummary::
   :toctree: generated/

   llc_rearrange.arct_connect
   llc_rearrange.mates
   llc_rearrange.rotate_vars
   llc_rearrange.shift_dataset
   llc_rearrange.reverse_dataset
   llc_rearrange.rotate_dataset
   llc_rearrange.shift_list_ds
   llc_rearrange.combine_list_ds
   llc_rearrange.flip_v
   llc_rearrange._edge_arc_data
   llc_rearrange.mask_var
   llc_rearrange.arc_limits_mask
   llc_rearrange._edge_facet_data
   llc_rearrange.slice_datasets
   llc_rearrange._LLC_check_sizes
   llc_rearrange._reorder_ds
   llc_rearrange.eval_dataset
   llc_rearrange.arctic_eval
   llc_rearrange.ds_edge_sametx
   llc_rearrange.ds_edge_samety
   llc_rearrange.ds_edge_difftx
   llc_rearrange.ds_edge_diffty
   llc_rearrange.ds_edge
   llc_rearrange.ds_arcedge
   llc_rearrange.face_direction
   llc_rearrange.splitter
   llc_rearrange.edge_completer
   llc_rearrange.edge_slider
   llc_rearrange.fill_path
   llc_rearrange.face_adjacent
   llc_rearrange.edgesid
   llc_rearrange.index_splitter
   llc_rearrange.order_from_indexing
   llc_rearrange.ds_splitarray
   llc_rearrange.fdir_completer
   llc_rearrange.mooring_singleface
   llc_rearrange.station_singleface
   llc_rearrange.cross_face_diffs
   llc_rearrange.arct_diffs
