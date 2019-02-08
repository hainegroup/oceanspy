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