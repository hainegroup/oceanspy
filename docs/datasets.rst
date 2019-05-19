.. _datasets:

========
Datasets
========

List of datasets available on SciServer.

.. _get_started:

-----------
get_started
-----------

Small cutout from EGshelfIIseas2km_ASR_crop_.
Citation:

* `Almansi et al., 2017 - JPO.`_

See also:

* EGshelfIIseas2km_ASR_full_: Full domain without variables to close budgets.
* EGshelfIIseas2km_ASR_crop_: Cropped domain with variables to close budgets.


Run the following code to open the dataset:

.. code-block:: ipython
    :class: no-execute

    import oceanspy as ospy
    od = ospy.open_oceandataset.from_catalog('get_started')

.. _EGshelfIIseas2km_ASR_full:

-------------------------
EGshelfIIseas2km_ASR_full
-------------------------

High-resolution (~2km) numerical simulation covering the east Greenland shelf (EGshelf), 
and the Iceland and Irminger Seas (IIseas) forced by the Arctic System Reanalysis (ASR). 
Citation:

* `Almansi et al., 2017 - JPO.`_

Characteristics:

* full: Full domain without variables to close budgets.

See also:

* EGshelfIIseas2km_ASR_crop_: Cropped domain with variables to close budgets.


Run the following code to open the dataset:

.. code-block:: ipython
    :class: no-execute

    import oceanspy as ospy
    od = ospy.open_oceandataset.from_catalog('EGshelfIIseas2km_ASR_full')

.. _EGshelfIIseas2km_ASR_crop:

-------------------------
EGshelfIIseas2km_ASR_crop
-------------------------

High-resolution (~2km) numerical simulation covering the east Greenland shelf (EGshelf), 
and the Iceland and Irminger Seas (IIseas) forced by the Arctic System Reanalysis (ASR). 
Citation:

* `Almansi et al., 2017 - JPO.`_

Characteristics:

* crop: Cropped domain with variables to close budgets.

See also:

* EGshelfIIseas2km_ASR_full_: Full domain without variables to close budgets.


Run the following code to open the dataset:

.. code-block:: ipython
    :class: no-execute

    import oceanspy as ospy
    od = ospy.open_oceandataset.from_catalog('EGshelfIIseas2km_ASR_crop')

.. _EGshelfIIseas2km_ERAI_6H:

------------------------
EGshelfIIseas2km_ERAI_6H
------------------------

High-resolution (~2km) numerical simulation covering the east Greenland shelf (EGshelf), 
and the Iceland and Irminger Seas (IIseas) forced by ERA-Interim. 
Citation:

* `Almansi et al., 2017 - JPO.`_

Characteristics:

* 6H: 6-hour resolution without sea ice and external forcing variables.

See also:

* EGshelfIIseas2km_ERAI_1D_: 1-day resolution with sea ice and external forcing variables.


Run the following code to open the dataset:

.. code-block:: ipython
    :class: no-execute

    import oceanspy as ospy
    od = ospy.open_oceandataset.from_catalog('EGshelfIIseas2km_ERAI_6H')

.. _EGshelfIIseas2km_ERAI_1D:

------------------------
EGshelfIIseas2km_ERAI_1D
------------------------

High-resolution (~2km) numerical simulation covering the east Greenland shelf (EGshelf), 
and the Iceland and Irminger Seas (IIseas) forced by ERA-Interim. 
Citation:

* `Almansi et al., 2017 - JPO.`_

Characteristics:

* 1D: 1-day resolution with sea ice and external forcing variables.

See also:

* EGshelfIIseas2km_ERAI_6H_: 6-hour resolution without sea ice and external forcing variables.


Run the following code to open the dataset:

.. code-block:: ipython
    :class: no-execute

    import oceanspy as ospy
    od = ospy.open_oceandataset.from_catalog('EGshelfIIseas2km_ERAI_1D')

.. _EGshelfSJsec500m_3H_hydro:

-------------------------
EGshelfSJsec500m_3H_hydro
-------------------------

Very high-resolution (500m) numerical simulation covering the east Greenland shelf (EGshelf) 
and the Spill Jet section (SJsec). Hydrostatic solutions.

Citation:

* `Magaldi and Haine, 2015 - DSR.`_

Characteristics:

* 3H:    3-hour resolution without external forcing variables.
* hydro: Hydrostatic solutions.

See also:

* EGshelfSJsec500m_6H_hydro_:    6-hour resolution with external forcing variables. Hydrostatic.
* EGshelfSJsec500m_6H_NONhydro_: 6-hour resolution with external forcing variables. Non-Hydrostatic.
* EGshelfSJsec500m_3H_NONhydro_: 3-hour resolution without external forcing variables. Non-Hydrostatic.


Run the following code to open the dataset:

.. code-block:: ipython
    :class: no-execute

    import oceanspy as ospy
    od = ospy.open_oceandataset.from_catalog('EGshelfSJsec500m_3H_hydro')

.. _EGshelfSJsec500m_6H_hydro:

-------------------------
EGshelfSJsec500m_6H_hydro
-------------------------

Very high-resolution (500m) numerical simulation covering the east Greenland shelf (EGshelf) 
and the Spill Jet section (SJsec). Hydrostatic solutions.

Citation:

* `Magaldi and Haine, 2015 - DSR.`_

Characteristics:

* 6H:    6-hour resolution with external forcing variables.
* hydro: Hydrostatic solutions.

See also:

* EGshelfSJsec500m_3H_hydro_:    3-hour resolution without external forcing variables. Hydrostatic.
* EGshelfSJsec500m_6H_NONhydro_: 6-hour resolution with external forcing variables. Non-Hydrostatic.
* EGshelfSJsec500m_3H_NONhydro_: 3-hour resolution without external forcing variables. Non-Hydrostatic.


Run the following code to open the dataset:

.. code-block:: ipython
    :class: no-execute

    import oceanspy as ospy
    od = ospy.open_oceandataset.from_catalog('EGshelfSJsec500m_6H_hydro')

.. _EGshelfSJsec500m_3H_NONhydro:

----------------------------
EGshelfSJsec500m_3H_NONhydro
----------------------------

Very high-resolution (500m) numerical simulation covering the east Greenland shelf (EGshelf) 
and the Spill Jet section (SJsec). Non-Hydrostatic solutions.

Citation:

* `Magaldi and Haine, 2015 - DSR.`_

Characteristics:

* 3H:       3-hour resolution without external forcing variables.
* NONhydro: Non-Hydrostatic solutions.

See also:

* EGshelfSJsec500m_6H_NONhydro_: 6-hour resolution with external forcing variables. Non-Hydrostatic.
* EGshelfSJsec500m_6H_hydro_:    6-hour resolution with external forcing variables. Hydrostatic.
* EGshelfSJsec500m_3H_hydro_:    3-hour resolution without external forcing variables. Hydrostatic.


Run the following code to open the dataset:

.. code-block:: ipython
    :class: no-execute

    import oceanspy as ospy
    od = ospy.open_oceandataset.from_catalog('EGshelfSJsec500m_3H_NONhydro')

.. _EGshelfSJsec500m_6H_NONhydro:

----------------------------
EGshelfSJsec500m_6H_NONhydro
----------------------------

Very high-resolution (500m) numerical simulation covering the east Greenland shelf (EGshelf) 
and the Spill Jet section (SJsec). Non-Hydrostatic solutions.

Citation:

* `Magaldi and Haine, 2015 - DSR.`_

Characteristics:

* 6H:       6-hour resolution with external forcing variables.
* NONhydro: NONHydrostatic solutions.

See also:

* EGshelfSJsec500m_3H_NONhydro_: 3-hour resolution without external forcing variables. Non-Hydrostatic.
* EGshelfSJsec500m_6H_hydro_:    6-hour resolution with external forcing variables. Hydrostatic.
* EGshelfSJsec500m_3H_hydro_:    3-hour resolution without external forcing variables. Hydrostatic.


Run the following code to open the dataset:

.. code-block:: ipython
    :class: no-execute

    import oceanspy as ospy
    od = ospy.open_oceandataset.from_catalog('EGshelfSJsec500m_6H_NONhydro')

.. _KangerFjord:

-----------
KangerFjord
-----------

A realistic numerical model constructed to simulate the oceanic conditions
and circulation in a large southeast Greenland fjord (Kangerdlugssuaq) and
the adjacent shelf sea region during winter 2007–2008. 

Citation:

* `Fraser et al., 2018 - JGR.`_


Run the following code to open the dataset:

.. code-block:: ipython
    :class: no-execute

    import oceanspy as ospy
    od = ospy.open_oceandataset.from_catalog('KangerFjord')

.. _`Almansi et al., 2017 - JPO.`: https://journals.ametsoc.org/doi/full/10.1175/JPO-D-17-0129.1
.. _`Magaldi and Haine, 2015 - DSR.`: https://www.sciencedirect.com/science/article/pii/S0967063714001915
.. _`Fraser et al., 2018 - JGR.`: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JC014435
