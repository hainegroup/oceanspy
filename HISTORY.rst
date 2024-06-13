.. _history:

=======
History
=======

v0.3.5 (2023-12-05)
-------------------

New Station subsampleMethod by (#377) by `Miguel Jimenez Urias`_. Improved
behavior of subsample.mooring_array (#354, #355, #361, #364, #386, #387, #398, #391
and #399), by `Miguel Jimenez Urias`_. Updating documentation (notebooks) by
`Miguel Jimenez Urias`_ in #340. Replaced rise with jupyterlab-rise to allow
binder to be build by `Miguel Jimenez Urias`_ in #400. Fix arctic_control
not opening by `Wenrui Jiang`_ (#396). Also fixed issues #321 and #269 by
`Wenrui Jiang`_. Allow trusted publishing by @malmans2 in #350. Ttest and
support python 3.11 by `Mattia Almansi`_ (#351). Added the ETOPO topography
to sciserver catalog by `Wenrui Jiang`_ (in #360).


v0.3.4 (2022-04-03)
-------------------
Fixed issues 322 (PR 325), 324 (PR 328), 332 (PR 324), and 312 (PR 337). Additional
grid files and removing (for the day) access to velocity LLC4320 data (PR 326) and update
environment (PR 335). All by `Miguel Jimenez Urias`_. Add daily mean ecco dataset to
catalog (PR 333) by


v0.3.3 (2022-02-07)
-------------------
Update binder environment, add llc4320 forcing files to catalog, fixed issue243, replace
deprecated cartopy property on notebook, Rename to Temp and S by, set persist as option via
argument, all by `Miguel Jimenez Urias`_. Fix toml prettifier by `Mattia Almansi`_.


v0.3.2 (2022-12-29)
-------------------
The new grid of the transformed dataset always has inner and outer as grid_coordinates. Improved documentation (api) for llc_rearrange and associated functions and classes. Unpins xesmf, improve functionality to :py:func:`~oceanspy.ospy_utils.viewer_to_range`. Improved (and fix) how vector fields are rotated from logical to geographical (lat-lon) coordinates. By `Miguel Jimenez Urias`_.


v0.3.1 (2022-12-14)
-------------------
Fix tarball and wheel for new release (PR 295 by `Filipe Fernandes`_).

v0.3.0 (2022-12-12)
-------------------
Migrated master to main (PR 239 by `Mattia Almansi`_). Allow oceandataset to have dict-style access to
variables (PR 262 by `Wenrui Jiang`_). Enhancement of oceanspy functionality to sample ocean data defined on llc-grids (PRs 214, 268 and 272, 284 by `Miguel Jimenez Urias`_). Integration with Poseidon viewer in Sciserver (PR 284 by `Miguel Jimenez Urias`_). HyCOM data can be accessed through SciServer (PR 206 by `Miguel Jimenez Urias`_). Bug and other fixes (by `Mattia Almansi`_, `Ali Siddiqui`_, `Tom Haine`_, `Wenrui Jiang`_ and `Miguel Jimenez Urias`_). See release notes for a full list of changes.

v0.2.0 (2020-10-17)
-------------------
Integration with LLC grid such as the ECCO data and the family of LLC simulations, while preserving the original (native) grid. This allows for the calculation (closure) of budgets. This new functionality was developed by `Miguel Jimenez Urias`_.

v0.1.0 (2019-07-06)
-------------------

Initial release published in the `Journal of Open Source Software`_.

`Mattia Almansi`_, `Renske Gelderloos`_, `Tom Haine`_, `Atousa Saberi`_, `Ali Siddiqui`_, `Miguel Jimenez Urias`_ `Wenrui Jiang`_ contributed to the development.

.. _`Mattia Almansi`: https://github.com/malmans2
.. _`Renske Gelderloos`: https://github.com/renskegelderloos
.. _`Tom Haine`: https://github.com/ThomasHaine
.. _`Atousa Saberi`: https://github.com/hooteoos-waltz
.. _`Ali Siddiqui`: https://github.com/asiddi24
.. _`Miguel Jimenez Urias`: https://github.com/Mikejmnez
.. _`Wenrui Jiang`: https://github.com/MaceKuailv
.. _`Filipe Fernandes`: https://github.com/ocefpaf
.. _`Journal of Open Source Software`: https://joss.theoj.org
