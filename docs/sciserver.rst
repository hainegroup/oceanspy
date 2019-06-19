.. _sciserver:

================
SciServer Access
================

`SciServer (www.sciserver.org)`_ is a data hosting service at the Johns Hopkins University which also provides tools for users to analyze the data. On this data hosting portal, a container can be made which allows the user to personalize the tools and look at specific data sets that they are interested in. For the purposes of OceanSpy, a number of data sets containing model output are already stored on Sciserver, which can be used to get started with OceanSpy as well as for research (see :ref:`datasets`).

After creating a user account on Sciserver, use one of the following useful features of Sciserver (always start with Compute Interact).

Compute Interact
----------------

SciServer allows to analyze data with an interactive notebook. 
Step-by-step instructions for using the interactive mode are available in :ref:`quick`.

The interactive mode runs on a Virtual Machine with 16 cores shared between multiple users. 
Use it for notebooks that don’t require heavy computations, or to test and design notebooks.

**For experts**: To install packages that are not available by default on the ``Oceanography image``, open a new terminal, then follow these step-by-step instructions:

1. Activate the Oceanography environment:

  .. code-block:: bash
		
		$ conda activate Oceanography

.. note::
		
	If you get a `CommandNotFoundError`, use the following command:

	.. code-block:: bash

		$ conda init bash		

	Then, open a new terminal and start from scratch:

	.. code-block:: bash

		$ conda activate Oceanography	

2. Optional: We suggest to use the following ``conda`` configuration commands:

  .. code-block:: bash

		$ conda config --set channel_priority strict
		$ conda config --prepend channels conda-forge

3. Use ``conda`` or ``pip`` to install new packages. For example:

  .. code-block:: bash

		$ conda install "name_of_packages_to_install"
		$ pip install "name_of_packages_to_install"


Compute Jobs
------------

`Compute Jobs`_ allows to run notebooks asynchronously.
Use the job mode to fully exploit the computational power of SciServer. 
For larger jobs (8 hour maximum), you have exclusive access to 32 logical CPU cores and 240GiB of memory.

.. note::

	If you're looking to work with the data interactively, then its better to use Compute Interact. However, if you're looking to run multiple notebooks simultaneously and want to exploit all CPU cores and memory available on Sciserver, Compute Jobs is the way to go.

1. Go to `SciServer (www.sciserver.org)`_.
2. Log in or create a new account.
3. Click on ``Compute Jobs``.
4. Click on ``Run Existing Notebook``.
5. Select a ``Compute Domain`` between ``Large Jobs Domain``, or ``Small Jobs Domain``.
6. Click on ``Compute Image`` and select ``Oceanography``.
7. Click on ``Data Volumes`` and select ``Ocean Circulation``.
8. Click on ``User Volumes`` and select the volumes that are needed by the Job (e.g., ``persistent`` and/or ``scratch``).
9. Click on ``Notebook`` and select the Jupyter Notebook that you want to execute. 
10. Select a ``Working Directory``, which is the location where the executed notebook and its output will be stored (you can just use the default ``jobs`` directory that will be created in your ``Temporary volume``).

The ``Oceanography image`` does not include any extra packages installed in your interactive containers.
To install packages that are not available by default on the ``Oceanography image``, add the following lines in the first cell of your notebook:

.. code-block:: ipython
    :class: no-execute

    import sys
    !conda install --yes --prefix {sys.prefix} [list of packages to be installed using conda]
    !{sys.executable} -m pip install [list of packages to be installed using pip]

For example, to install the latest version of OceanSpy, use the following cell:

.. code-block:: ipython
    :class: no-execute

    import sys
    !{sys.executable} -m pip install --upgrade git+https://github.com/malmans2/oceanspy.git


.. _`SciServer (www.sciserver.org)`: http://www.sciserver.org/
.. _`Compute Interact`: https://apps.sciserver.org/compute/
.. _`Compute Jobs`: https://apps.sciserver.org/compute/jobs
